"""Recommendation assembly. Milestone 3: embedding similarity (score ②) only;
later milestones add scores ① and ③, blending, MMR and the LLM layer."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from sqlmodel import Session, col, select

from app.db import Candidate, Movie, UserFilm
from app.taste import TasteModel, score_embeddings
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
    score: float  # blended score, 0–1 (for now: score ②)
    embedding_score: float  # score ②, percentile-normalized 0–1
    similarity: float  # raw best cosine similarity
    source: str  # "taste" or "cluster:<id>"
    source_label: str
    candidate_sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecFilters:
    """All optional. A film with an unknown value fails any filter on that field
    (e.g. no TMDB rating can't satisfy a minimum rating)."""

    min_rating: float | None = None  # TMDB user score, 0–10
    genres: tuple[str, ...] = ()  # match any
    decade: int | None = None  # e.g. 1990
    max_runtime: int | None = None  # minutes
    language: str | None = None  # ISO 639-1, e.g. "ko"

    @property
    def active(self) -> bool:
        return any(
            (self.min_rating is not None, self.genres, self.decade is not None,
             self.max_runtime is not None, self.language)
        )

    def matches(self, r: Recommendation) -> bool:
        if self.min_rating is not None and (r.vote_average is None or r.vote_average < self.min_rating):
            return False
        if self.genres and not set(self.genres) & set(r.genres):
            return False
        if self.decade is not None and (r.year is None or r.year // 10 * 10 != self.decade):
            return False
        if self.max_runtime is not None and (r.runtime is None or r.runtime > self.max_runtime):
            return False
        if self.language and r.original_language != self.language:
            return False
        return True


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


def embedding_pool(store: VectorStore, model: TasteModel, k: int) -> set[int]:
    """Union of the k nearest unseen films to the taste vector and each cluster centroid."""
    pool: set[int] = set()
    for _, vec in model.vectors():
        pool.update(i for i, _ in store.query(vec, k, where={"seen": False}))
    return pool


def recommend(
    session: Session,
    store: VectorStore,
    model: TasteModel,
    *,
    limit: int = 40,
    k_per_vector: int = 300,
    mode: Literal["zmax", "max"] = "zmax",
    filters: RecFilters | None = None,
) -> RecResult:
    filters = filters or RecFilters()
    pool = embedding_pool(store, model, k_per_vector)
    # The `seen` flag in the index can lag behind a fresh upload; the DB is the truth.
    pool -= _seen_ids(session)
    if not pool:
        return RecResult(items=[], total=0, matching=0, facets=facets_for([]))

    scores = score_embeddings(model, store.get_embeddings(pool), mode)
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

    recs: list[Recommendation] = []
    for tid, sc in scores.items():
        m = movies.get(tid)
        if m is None:
            log.warning("tmdb %s is in the vector index but not the DB; skipped", tid)
            continue
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
                score=sc.normalized,
                embedding_score=sc.normalized,
                similarity=sc.raw,
                source=sc.source,
                source_label=model.label_for(sc.source),
                candidate_sources=provenance.get(tid, ["watchlist"] if tid in watchlist else []),
            )
        )
    # Scores are percentiles over the whole pool, so filtering never changes a
    # film's score, only which films are shown. Filter before applying the limit.
    recs.sort(key=lambda r: (r.score, r.similarity), reverse=True)
    kept = [r for r in recs if filters.matches(r)]
    if filters.active:
        log.info("filters %s kept %d of %d candidates", filters, len(kept), len(recs))
    return RecResult(items=kept[:limit], total=len(recs), matching=len(kept), facets=facets_for(recs))
