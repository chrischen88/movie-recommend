"""Recommendation assembly: scores ① (taste profile), ② (embedding similarity)
and ③ (collaborative) combined by the blend (app/blend.py), then the watchlist
boost, filters (incl. the OMDb quality floor) and MMR re-ranking for variety.
M7 adds the LLM layer."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Literal

from sqlmodel import Session, col, select

import numpy as np

from app.blend import BlendModel, rank_with_fit, taste_fit
from app.collab import CollabScorer
from app.db import Candidate, Movie, UserFilm
from app.profile import Contribution, TasteProfile, score_profile, user_ratings
from app.taste import TasteModel, percentile_rank, score_embeddings
from app.vectorstore import VectorStore

log = logging.getLogger(__name__)


@dataclass
class Recommendation:
    tmdb_id: int
    title: str
    year: int | None
    directors: list[str]
    genres: list[str]
    poster_path: str | None
    overview: str | None
    runtime: int | None
    vote_average: float | None
    vote_count: int | None
    original_language: str | None
    in_watchlist: bool
    score: float  # blended score: percentile of the blend over the pool (+ watchlist boost), 0–1
    embedding_score: float  # score ②, percentile-normalized 0–1
    similarity: float  # raw best cosine similarity
    source: str  # "taste" or "cluster:<id>"
    source_label: str
    candidate_sources: list[str] = field(default_factory=list)
    profile_score: float | None = None  # score ①, percentile-normalized 0–1; None without a profile
    profile_raw: float | None = None
    profile_reasons: list[Contribution] = field(default_factory=list)  # strongest first
    collab_score: float | None = None  # score ③, percentile-normalized 0–1; None if not in MovieLens
    collab_predicted: float | None = None  # predicted rating, 0.5–5★
    predicted_rating: float | None = None  # the learned blend's predicted rating; None with fixed weights
    imdb_rating: float | None = None
    rt_score: int | None = None
    metacritic: int | None = None


@dataclass(frozen=True)
class RecFilters:
    """All optional. A film with an unknown value fails any filter on that field
    (e.g. no TMDB rating can't satisfy a minimum rating)."""

    min_rating: float | None = None  # on `rating_source`'s scale
    rating_source: RatingSource = "tmdb"
    genres: tuple[str, ...] = ()  # match any
    decade: int | None = None  # e.g. 1990
    max_runtime: int | None = None  # minutes
    language: str | None = None  # ISO 639-1, e.g. "ko"
    # Quality floor: Tomatometer ≥ quality_rt or IMDb ≥ quality_imdb. Films with
    # neither (OMDb is only fetched for the shortlist) fall back to TMDB ≥ quality_imdb.
    hide_low_quality: bool = False
    quality_rt: int = 60
    quality_imdb: float = 6.5

    @property
    def active(self) -> bool:
        return any(
            (self.min_rating is not None, self.genres, self.decade is not None,
             self.max_runtime is not None, self.language, self.hide_low_quality)
        )

    def matches(self, r: Recommendation) -> bool:
        if self.min_rating is not None:
            value = rating_value(r, self.rating_source)
            if value is None or value < self.min_rating:
                return False
        if self.genres and not set(self.genres) & set(r.genres):
            return False
        if self.decade is not None and (r.year is None or r.year // 10 * 10 != self.decade):
            return False
        if self.max_runtime is not None and (r.runtime is None or r.runtime > self.max_runtime):
            return False
        if self.language and r.original_language != self.language:
            return False
        if self.hide_low_quality and not self.good_enough(r):
            return False
        return True

    def good_enough(self, r: Recommendation) -> bool:
        if r.rt_score is not None or r.imdb_rating is not None:
            return (r.rt_score or 0) >= self.quality_rt or (r.imdb_rating or 0) >= self.quality_imdb
        return r.vote_average is not None and r.vote_average >= self.quality_imdb


RatingSource = Literal["tmdb", "imdb", "rt", "metacritic"]


def rating_value(r: Recommendation, source: RatingSource) -> float | None:
    """TMDB and IMDb are 0–10; Rotten Tomatoes and Metacritic are 0–100."""
    return {"tmdb": r.vote_average, "imdb": r.imdb_rating, "rt": r.rt_score, "metacritic": r.metacritic}[source]


@dataclass
class Facets:
    """What's available in the unfiltered pool, with counts, for the filter UI."""

    genres: dict[str, int]
    decades: dict[int, int]
    languages: dict[str, int]


@dataclass
class RecResult:
    items: list[Recommendation]
    total: int  # candidates before filters
    matching: int  # candidates passing the filters
    facets: Facets
    collab_films: int | None = None  # your rated films found in MovieLens; None = no score ③


def facets_for(recs: list[Recommendation]) -> Facets:
    genres = Counter(g for r in recs for g in r.genres)
    decades = Counter(r.year // 10 * 10 for r in recs if r.year)
    languages = Counter(r.original_language for r in recs if r.original_language)
    return Facets(
        genres=dict(genres.most_common()),
        decades=dict(sorted(decades.items())),
        languages=dict(languages.most_common()),
    )


def _seen_ids(session: Session) -> set[int]:
    return {
        i
        for i in session.exec(
            select(UserFilm.tmdb_id).where(col(UserFilm.watched).is_(True))
        )
        if i is not None
    }


def eligible_ids(session: Session) -> set[int]:
    """Films that can be recommended: current candidates and the user's unwatched
    (watchlist) films."""
    return current_film_ids(session) - _seen_ids(session)


def current_film_ids(session: Session) -> set[int]:
    """The user's own matched films plus the current candidates. Anything else in
    the index or `movie` table is left over from an earlier library."""
    own = {i for i in session.exec(select(UserFilm.tmdb_id)) if i is not None}
    return own | set(session.exec(select(Candidate.tmdb_id)).all())


def recommend(
    session: Session,
    store: VectorStore,
    model: TasteModel,
    *,
    limit: int = 40,
    mode: Literal["zmax", "max"] = "zmax",
    filters: RecFilters | None = None,
    profile: TasteProfile | None = None,
    feature_weights: Mapping[str, float] | None = None,
    reasons_per_film: int = 5,
    min_reason_stars: float = 0.05,
    collab: CollabScorer | None = None,
    collab_min_user_ratings: int = 5,
    blend: BlendModel | None = None,
    watchlist_boost: float = 0.0,
    mmr_lambda: float | None = None,
    min_runtime: int = 0,
    fit_weight: float = 0.0,
) -> RecResult:
    """`blend=None` averages the available scores equally. `mmr_lambda=None`
    skips MMR re-ranking. Films shorter than `min_runtime` minutes are left out
    unless they're on the watchlist. In learned mode, `fit_weight` mixes taste
    fit into the ranking (`blend.rank_with_fit`)."""
    filters = filters or RecFilters()
    # Every unseen film in the library or candidate set is scored: a few hundred
    # to a few thousand dot products, so no nearest-neighbour cutoff. (A k-NN
    # prefilter would drop films that ① or ③ rate highly just because ② doesn't.)
    # The DB, not the index, is the truth for what's seen and still in the library.
    pool = eligible_ids(session)
    embeddings = store.get_embeddings(pool)
    if len(embeddings) < len(pool):
        log.info("%d eligible films aren't embedded yet and are skipped", len(pool) - len(embeddings))
    if not embeddings:
        return RecResult(items=[], total=0, matching=0, facets=facets_for([]))

    scores = score_embeddings(model, embeddings, mode)
    movies = {m.tmdb_id: m for m in session.exec(select(Movie).where(col(Movie.tmdb_id).in_(scores)))}
    watchlist = {
        i
        for i in session.exec(
            select(UserFilm.tmdb_id).where(col(UserFilm.in_watchlist).is_(True))
        )
        if i is not None
    }
    provenance = {
        c.tmdb_id: c.sources
        for c in session.exec(select(Candidate).where(col(Candidate.tmdb_id).in_(scores)))
    }
    profile_scores = (
        score_profile(
            profile, movies.values(), feature_weights or {}, reasons_per_film, min_reason_stars
        )
        if profile is not None
        else {}
    )
    folded = collab.fold_in(user_ratings(session), collab_min_user_ratings) if collab is not None else None
    collab_scores = collab.score(folded, list(movies)) if collab is not None and folded is not None else {}

    blend = blend or BlendModel(mode="fixed", fixed_weights={"profile": 1.0, "embedding": 1.0, "collab": 1.0})
    recs: list[Recommendation] = []
    values: list[float] = []  # what the blend ranks by: predicted stars, or a weighted mean
    fits: list[float] = []  # taste fit, mixed into the ranking in learned mode
    for tid, sc in scores.items():
        m = movies.get(tid)
        if m is None:
            log.warning("tmdb %s is in the vector index but not the DB; skipped", tid)
            continue
        ps = profile_scores.get(tid)
        cs = collab_scores.get(tid)
        p_n = None if ps is None else ps.normalized
        c_n = None if cs is None else cs.normalized
        predicted = blend.predict(p_n, sc.normalized, c_n, m.vote_count)
        values.append(predicted if predicted is not None else blend.fixed_score(p_n, sc.normalized, c_n))
        fits.append(taste_fit(p_n, sc.normalized))
        recs.append(
            Recommendation(
                tmdb_id=tid,
                title=m.title,
                year=m.year,
                directors=m.directors,
                genres=m.genres,
                poster_path=m.poster_path,
                overview=m.overview,
                runtime=m.runtime,
                vote_average=m.vote_average,
                vote_count=m.vote_count,
                original_language=m.original_language,
                in_watchlist=tid in watchlist,
                score=0.0,  # set below, once the whole pool is scored
                embedding_score=sc.normalized,
                similarity=sc.raw,
                source=sc.source,
                source_label=model.label_for(sc.source),
                candidate_sources=provenance.get(tid, ["watchlist"] if tid in watchlist else []),
                profile_score=None if ps is None else ps.normalized,
                profile_raw=None if ps is None else ps.raw,
                profile_reasons=[] if ps is None else ps.contributions,
                collab_score=None if cs is None else cs.normalized,
                collab_predicted=None if cs is None else cs.predicted,
                predicted_rating=predicted,
                imdb_rating=m.imdb_rating,
                rt_score=m.rt_score,
                metacritic=m.metacritic,
            )
        )
    learned = blend.mode == "learned"
    ranking = rank_with_fit(np.array(values), np.array(fits), fit_weight if learned else 0.0)
    for r, pct in zip(recs, ranking):
        r.score = min(1.0, float(pct) + (watchlist_boost if r.in_watchlist else 0.0))

    # Scores are percentiles over the whole pool, so filtering never changes a
    # film's score, only which films are shown. Filter before applying the limit.
    recs.sort(key=lambda r: (r.score, r.similarity), reverse=True)
    kept = [r for r in recs if filters.matches(r) and not _is_short(r, min_runtime)]
    if filters.active:
        log.info("filters %s kept %d of %d candidates", filters, len(kept), len(recs))
    top = mmr(kept, embeddings, mmr_lambda, limit) if mmr_lambda is not None else kept[:limit]
    return RecResult(
        items=top,
        total=len(recs),
        matching=len(kept),
        facets=facets_for(recs),
        collab_films=None if folded is None else folded.n_mapped,
    )


def _is_short(r: Recommendation, min_runtime: int) -> bool:
    return not r.in_watchlist and r.runtime is not None and 0 < r.runtime < min_runtime


def mmr(
    ranked: list[Recommendation], embeddings: Mapping[int, np.ndarray], lam: float, k: int
) -> list[Recommendation]:
    """Maximal marginal relevance: repeatedly take the film with the best
    λ·score − (1−λ)·(max cosine to films already picked), so the top `k` aren't
    near-duplicates. Only the best 3k (at least 60) are considered; λ=1 keeps
    the ranking as is.

    Relevance is rescaled to 0–1 within that pool: the scores are percentiles
    over every candidate, so the top 60 all sit near 1 and would otherwise be
    swamped by the similarity term, ranking by novelty instead of fit."""
    pool = [r for r in ranked[: max(3 * k, 60)] if r.tmdb_id in embeddings]
    if not pool:
        return ranked[:k]
    vecs = np.stack([embeddings[r.tmdb_id] for r in pool]).astype(np.float32)
    vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    rel = np.array([r.score for r in pool])
    spread = rel.max() - rel.min()
    rel = (rel - rel.min()) / spread if spread > 0 else np.ones(len(pool))
    max_sim = np.zeros(len(pool))
    chosen: list[int] = []
    available = np.ones(len(pool), dtype=bool)
    for _ in range(min(k, len(pool))):
        gain = np.where(available, lam * rel - (1 - lam) * max_sim, -np.inf)
        i = int(np.argmax(gain))
        chosen.append(i)
        available[i] = False
        max_sim = np.maximum(max_sim, vecs @ vecs[i])
    return [pool[i] for i in chosen]
