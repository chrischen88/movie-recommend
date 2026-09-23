from __future__ import annotations

import numpy as np
import pytest

from app.recommend import Recommendation, RecFilters, facets_for, _is_short, mmr


def rec(**kw: object) -> Recommendation:
    base: dict[str, object] = dict(
        tmdb_id=1, title="T", year=2001, directors=[], genres=["Drama"], poster_path=None,
        overview=None, runtime=100, vote_average=7.0, vote_count=500, original_language="en",
        in_watchlist=False, score=0.5, embedding_score=0.5, similarity=0.5,
        source="taste", source_label="x",
    )
    base.update(kw)
    return Recommendation(**base)  # type: ignore[arg-type]


def test_no_filters_match_everything() -> None:
    f = RecFilters()
    assert not f.active
    assert f.matches(rec(vote_average=None, year=None, runtime=None, original_language=None))


@pytest.mark.parametrize(
    ("filters", "film", "ok"),
    [
        (RecFilters(min_rating=7.0), rec(vote_average=7.0), True),
        (RecFilters(min_rating=7.0), rec(vote_average=6.9), False),
        (RecFilters(min_rating=7.0), rec(vote_average=None), False),  # unknown fails
        (RecFilters(genres=("Horror", "Drama")), rec(genres=["Drama", "Romance"]), True),
        (RecFilters(genres=("Horror",)), rec(genres=["Drama"]), False),
        (RecFilters(decade=1990), rec(year=1999), True),
        (RecFilters(decade=1990), rec(year=2000), False),
        (RecFilters(decade=1990), rec(year=None), False),
        (RecFilters(max_runtime=100), rec(runtime=100), True),
        (RecFilters(max_runtime=99), rec(runtime=100), False),
        (RecFilters(max_runtime=99), rec(runtime=None), False),
        (RecFilters(language="ko"), rec(original_language="ko"), True),
        (RecFilters(language="ko"), rec(original_language="en"), False),
        (RecFilters(min_rating=7.5, decade=2000), rec(vote_average=8.0, year=2004), True),
        (RecFilters(min_rating=7.5, decade=2000), rec(vote_average=8.0, year=1994), False),
    ],
)
def test_filter_matching(filters: RecFilters, film: Recommendation, ok: bool) -> None:
    assert filters.active
    assert filters.matches(film) is ok


def test_facets() -> None:
    f = facets_for([
        rec(genres=["Drama", "Romance"], year=1994, original_language="cn"),
        rec(genres=["Drama"], year=1997, original_language="en"),
        rec(genres=["Horror"], year=None, original_language=None),
    ])
    assert f.genres == {"Drama": 2, "Romance": 1, "Horror": 1}
    assert list(f.genres)[0] == "Drama"  # most common first
    assert f.decades == {1990: 2}
    assert f.languages == {"cn": 1, "en": 1}


# ---------------------------------------------------------------- M6: rating sources, quality floor, MMR


@pytest.mark.parametrize(
    ("filters", "film", "ok"),
    [
        (RecFilters(min_rating=7.0, rating_source="imdb"), rec(imdb_rating=7.2, vote_average=5.0), True),
        (RecFilters(min_rating=7.0, rating_source="imdb"), rec(imdb_rating=None), False),
        (RecFilters(min_rating=80, rating_source="rt"), rec(rt_score=85), True),
        (RecFilters(min_rating=80, rating_source="rt"), rec(rt_score=79), False),
        (RecFilters(min_rating=70, rating_source="metacritic"), rec(metacritic=70), True),
        (RecFilters(hide_low_quality=True), rec(rt_score=60, imdb_rating=5.0), True),  # either passes
        (RecFilters(hide_low_quality=True), rec(rt_score=40, imdb_rating=6.5), True),
        (RecFilters(hide_low_quality=True), rec(rt_score=40, imdb_rating=6.0), False),
        (RecFilters(hide_low_quality=True), rec(rt_score=None, imdb_rating=6.0), False),
        # No OMDb data at all: fall back to the TMDB score.
        (RecFilters(hide_low_quality=True), rec(vote_average=6.8), True),
        (RecFilters(hide_low_quality=True), rec(vote_average=6.0), False),
        (RecFilters(hide_low_quality=True), rec(vote_average=None), False),
    ],
)
def test_rating_sources_and_quality_floor(filters: RecFilters, film: Recommendation, ok: bool) -> None:
    assert filters.active
    assert filters.matches(film) is ok


def test_mmr_spreads_near_duplicates() -> None:
    e = {1: np.array([1.0, 0.0]), 2: np.array([0.99, 0.14]), 3: np.array([0.0, 1.0]), 4: np.array([-1.0, 0.0])}
    ranked = [rec(tmdb_id=1, score=1.0), rec(tmdb_id=2, score=0.98), rec(tmdb_id=3, score=0.9),
              rec(tmdb_id=4, score=0.0)]
    assert [r.tmdb_id for r in mmr(ranked, e, 1.0, 3)] == [1, 2, 3]  # λ=1: relevance only
    assert [r.tmdb_id for r in mmr(ranked, e, 0.7, 3)] == [1, 3, 2]  # the near-duplicate waits
    assert [r.tmdb_id for r in mmr(ranked, e, 0.7, 2)] == [1, 3]


def test_mmr_relevance_is_relative_to_the_pool() -> None:
    """Percentile scores bunch up near 1 at the top; a slightly worse but
    dissimilar film mustn't jump ahead of a good, somewhat similar one."""
    e = {1: np.array([1.0, 0.0, 0.0]), 2: np.array([0.9, 0.44, 0.0]), 3: np.array([0.0, 0.0, 1.0]),
         4: np.array([0.7, 0.71, 0.0])}
    ranked = [rec(tmdb_id=1, score=1.0), rec(tmdb_id=2, score=0.99), rec(tmdb_id=4, score=0.96),
              rec(tmdb_id=3, score=0.95)]
    assert [r.tmdb_id for r in mmr(ranked, e, 0.7, 4)][:2] == [1, 2]
    assert mmr([], e, 0.7, 3) == []


def test_shorts_hidden_unless_watchlisted() -> None:
    assert _is_short(rec(runtime=8), 40)
    assert not _is_short(rec(runtime=8, in_watchlist=True), 40)
    assert not _is_short(rec(runtime=90), 40)
    assert not _is_short(rec(runtime=None), 40)  # unknown runtime: keep
    assert not _is_short(rec(runtime=8), 0)  # disabled
