"""Score ①: the content-based taste profile.

For every feature value (a director, genre, actor, keyword, decade, language or
country) the profile stores the user's average rating deviation (rating − μ)
over the films that have it, shrunk toward 0: `Σ(rating − μ) / (n + k)`. One
4.5★ film with an obscure keyword can't outweigh a director seen ten times.

A candidate's raw score is the weighted sum of its features' profile values.
Within each feature type the sum is divided by √(number of values of that type),
so a film with 40 keywords doesn't swamp one with 5 (see PROGRESS.md, decision 9).
Local data only: no API calls.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from sqlmodel import Session, col, select

from app.db import Movie, UserFilm
from app.taste import percentile_rank

FEATURE_TYPES = ("director", "genre", "actor", "keyword", "decade", "language", "country")
FEATURE_LABELS = {
    "director": "Director",
    "genre": "Genre",
    "actor": "Actor",
    "keyword": "Keyword",
    "decade": "Decade",
    "language": "Language",
    "country": "Country",
}
TOP_CAST = 5
_NOISE = 1e-9  # Σ(rating − μ) over every rated film is 0 up to float error

Feature = tuple[str, str]  # (type, value), e.g. ("director", "Denis Villeneuve")


def film_features(m: Movie) -> list[Feature]:
    """Every (type, value) pair a film has, deduplicated, in a stable order."""
    feats: list[Feature] = []
    feats += [("director", d) for d in m.directors]
    feats += [("genre", g) for g in m.genres]
    feats += [("actor", a) for a in m.cast[:TOP_CAST]]
    feats += [("keyword", k) for k in m.keywords]
    if m.year:
        feats.append(("decade", f"{m.year // 10 * 10}s"))
    if m.original_language:
        feats.append(("language", m.original_language))
    feats += [("country", c) for c in m.countries if c]
    return list(dict.fromkeys(feats))


@dataclass(frozen=True)
class FeatureStat:
    value: float  # shrunk mean deviation, in stars
    n: int  # rated films that have this feature


@dataclass
class TasteProfile:
    mean_rating: float
    n_rated: int
    stats: dict[Feature, FeatureStat] = field(default_factory=dict)

    def top(self, ftype: str, n: int = 10, *, min_films: int = 2, reverse: bool = False) -> list[tuple[str, FeatureStat]]:
        """Best (or with `reverse`, worst) values of one type, for a profile page."""
        rows = [(v, s) for (t, v), s in self.stats.items() if t == ftype and s.n >= min_films]
        rows.sort(key=lambda row: row[1].value, reverse=not reverse)
        return rows[:n]


def build_profile(
    ratings: Mapping[int, float], movies: Mapping[int, Movie], shrinkage_k: float
) -> TasteProfile | None:
    """`ratings`: tmdb_id → the user's rating. Films without metadata are skipped."""
    rated = [(movies[i], r) for i, r in ratings.items() if i in movies]
    if not rated:
        return None
    mu = sum(r for _, r in rated) / len(rated)
    sums: dict[Feature, float] = {}
    counts: dict[Feature, int] = {}
    for m, r in rated:
        for f in film_features(m):
            sums[f] = sums.get(f, 0.0) + (r - mu)
            counts[f] = counts.get(f, 0) + 1
    stats: dict[Feature, FeatureStat] = {}
    for f, total in sums.items():
        value = total / (counts[f] + shrinkage_k)
        stats[f] = FeatureStat(value=value if abs(value) > _NOISE else 0.0, n=counts[f])
    return TasteProfile(mean_rating=mu, n_rated=len(rated), stats=stats)


@dataclass(frozen=True)
class Contribution:
    type: str
    value: str
    label: str  # "Director Denis Villeneuve"
    stars: float  # the profile value: shrunk average of (rating − μ)
    n: int  # rated films behind it
    contribution: float  # weighted share of the raw score


@dataclass
class ProfileScore:
    raw: float
    normalized: float = 0.0  # percentile rank within the candidate set, 0–1
    contributions: list[Contribution] = field(default_factory=list)  # strongest first


def score_film(profile: TasteProfile, m: Movie, weights: Mapping[str, float]) -> ProfileScore:
    by_type: dict[str, list[str]] = {}
    for t, v in film_features(m):
        by_type.setdefault(t, []).append(v)
    contribs: list[Contribution] = []
    for t, values in by_type.items():
        w = weights.get(t, 0.0)
        if not w:
            continue
        scale = w / math.sqrt(len(values))
        for v in values:
            stat = profile.stats.get((t, v))
            if stat is None or stat.value == 0.0:
                continue
            contribs.append(
                Contribution(
                    type=t,
                    value=v,
                    label=f"{FEATURE_LABELS.get(t, t.title())} {v}",
                    stars=stat.value,
                    n=stat.n,
                    contribution=scale * stat.value,
                )
            )
    contribs.sort(key=lambda c: abs(c.contribution), reverse=True)
    return ProfileScore(raw=sum(c.contribution for c in contribs), contributions=contribs)


def score_profile(
    profile: TasteProfile,
    movies: Iterable[Movie],
    weights: Mapping[str, float],
    top_contributions: int = 5,
    min_reason_stars: float = 0.05,
) -> dict[int, ProfileScore]:
    """Score ① for each candidate, percentile-ranked across the set. Only the
    strongest contributions are kept for explanations, and none whose profile
    value is too small to mean anything (they still count toward `raw`)."""
    scores = {m.tmdb_id: score_film(profile, m, weights) for m in movies}
    ranked = percentile_rank(np.array([s.raw for s in scores.values()]))
    for s, pct in zip(scores.values(), ranked):
        s.normalized = float(pct)
        s.contributions = [c for c in s.contributions if abs(c.stars) >= min_reason_stars][:top_contributions]
    return scores


# ---------------------------------------------------------------- loading


def user_ratings(session: Session) -> dict[int, float]:
    """tmdb_id → rating. Two Letterboxd entries can map to one TMDB film: average them."""
    per_film: dict[int, list[float]] = {}
    rows = session.exec(
        select(UserFilm.tmdb_id, UserFilm.rating).where(
            col(UserFilm.tmdb_id).is_not(None), col(UserFilm.rating).is_not(None)
        )
    )
    for tid, rating in rows:
        if tid is not None and rating is not None:
            per_film.setdefault(tid, []).append(rating)
    return {tid: sum(rs) / len(rs) for tid, rs in per_film.items()}


def load_profile(session: Session, shrinkage_k: float) -> TasteProfile | None:
    ratings = user_ratings(session)
    movies = {m.tmdb_id: m for m in session.exec(select(Movie).where(col(Movie.tmdb_id).in_(ratings)))}
    return build_profile(ratings, movies, shrinkage_k)
