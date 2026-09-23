from __future__ import annotations

from typing import Any

import pytest

from app import matching
from app.matching import match_film, normalize_for_match, parse_tmdb_ref, title_similarity
from tests.fake_tmdb import FakeTmdb, movie


class StubSearcher:
    def __init__(self, fake: FakeTmdb) -> None:
        self.fake = fake
        self.calls: list[tuple[str, int | None]] = []

    def search_movie(self, query: str, year: int | None = None) -> list[dict[str, Any]]:
        self.calls.append((query, year))
        import httpx

        params = httpx.QueryParams({"query": query, **({"year": year} if year else {})})
        return self.fake._search(params)["results"]


def searcher(*movies: dict[str, Any]) -> StubSearcher:
    return StubSearcher(FakeTmdb(list(movies)))


def test_normalize() -> None:
    assert normalize_for_match("Amélie") == "amelie"
    assert normalize_for_match("The Good, the Bad & the Ugly") == "good the bad and the ugly"
    assert normalize_for_match("Mad Max: Fury Road") == "mad max fury road"


def test_title_similarity() -> None:
    assert title_similarity("Amélie", "Amelie") == 1.0
    assert title_similarity("The Room", "Room") == matching.ARTICLE_ONLY_MATCH
    assert title_similarity("Arrival", "Survival") < 0.9


def test_exact_match() -> None:
    s = searcher(movie(329865, "Arrival", 2016), movie(9999, "Arrival", 1996, votes=50))
    r = match_film(s, "Arrival", 2016, 0.75)
    assert (r.tmdb_id, r.status, r.confidence) == (329865, matching.MATCHED, 1.0)
    assert s.calls == [("Arrival", 2016)]


def test_year_plus_minus_one_retry() -> None:
    s = searcher(movie(949, "Heat", 1995))
    r = match_film(s, "Heat", 1996, 0.75)
    assert r.tmdb_id == 949 and r.status == matching.MATCHED
    assert r.confidence == pytest.approx(0.9)
    assert s.calls == [("Heat", 1996), ("Heat", 1995)]
    assert "search year 1995" in r.note


def test_no_year_falls_back_to_plain_search() -> None:
    s = searcher(movie(1, "Heat", 1995))
    r = match_film(s, "Heat", None, 0.75)
    assert r.tmdb_id == 1 and r.confidence == pytest.approx(matching.YEAR_FACTOR_UNKNOWN)


def test_unmatched() -> None:
    s = searcher(movie(1, "Heat", 1995))
    r = match_film(s, "Home Movie Night", 2004, 0.75)
    assert r.tmdb_id is None and r.status == matching.UNMATCHED
    assert s.calls == [
        ("Home Movie Night", 2004), ("Home Movie Night", 2003),
        ("Home Movie Night", 2005), ("Home Movie Night", None),
    ]


def test_far_year_is_low_confidence() -> None:
    s = searcher(movie(1, "The Arrival", 1996))
    r = match_film(s, "Arrival", 2016, 0.75)
    assert r.tmdb_id == 1 and r.status == matching.LOW_CONFIDENCE
    assert r.confidence < 0.5


def test_ambiguous_same_title_same_year_flagged() -> None:
    s = searcher(movie(1, "Crash", 2004, votes=5000), movie(2, "Crash", 2004, votes=3000))
    r = match_film(s, "Crash", 2004, 0.75)
    assert r.tmdb_id == 1
    assert r.status == matching.LOW_CONFIDENCE
    assert "ambiguous" in r.note


def test_near_tie_with_obscure_film_prefers_popular_without_flag() -> None:
    s = searcher(movie(2, "Dune", 2021, votes=3), movie(1, "Dune", 2021, votes=12000))
    r = match_film(s, "Dune", 2021, 0.75)
    assert (r.tmdb_id, r.status) == (1, matching.MATCHED)
    assert "ambiguous" not in r.note


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("329865", 329865),
        (" 42 ", 42),
        ("https://www.themoviedb.org/movie/329865-arrival", 329865),
        ("themoviedb.org/movie/603?language=en", 603),
        ("0", None),
        ("arrival", None),
        (7, 7),
    ],
)
def test_parse_tmdb_ref(ref: str | int, expected: int | None) -> None:
    assert parse_tmdb_ref(ref) == expected
