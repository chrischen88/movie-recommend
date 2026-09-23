from __future__ import annotations

import pytest

from app.recommend import Recommendation, RecFilters, facets_for


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
