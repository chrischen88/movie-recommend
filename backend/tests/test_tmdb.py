from __future__ import annotations

from sqlalchemy import Engine

from app.cache import ResponseCache
from app.config import Settings
from app.tmdb import TmdbClient, fetch_movie, parse_movie
from tests.fake_tmdb import FakeTmdb, movie


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, tmdb_requests_per_second=1000, **kw)  # type: ignore[call-arg]


def test_parse_full_details(engine: Engine) -> None:
    fake = FakeTmdb([movie(329865, "Arrival", 2016)])
    client = TmdbClient("v3key", ResponseCache(engine), settings(), transport=fake.transport)
    m = fetch_movie(client, 329865)
    assert m is not None
    assert m.title == "Arrival" and m.year == 2016
    assert m.directors == ["Director 329865"]
    assert m.cast == ["Actor 6", "Actor 5", "Actor 4", "Actor 3", "Actor 2"]
    assert m.genres == ["Drama", "Science Fiction"]
    assert m.keywords == ["alien", "language"]
    assert m.countries == ["US"]
    assert m.imdb_id == "tt0329865"
    assert m.runtime and m.overview and m.poster_path == "/p329865.jpg"
    assert len(m.reviews) == 2
    assert len(m.reviews[0]) <= 500 and m.reviews[0].endswith("…")
    assert fake.requests[0].url.params["api_key"] == "v3key"
    assert fake.requests[0].url.params["append_to_response"] == "credits,keywords,external_ids"


def test_fetch_missing_movie_returns_none(engine: Engine) -> None:
    fake = FakeTmdb([])
    client = TmdbClient("k", ResponseCache(engine), settings(), transport=fake.transport)
    assert fetch_movie(client, 123) is None


def test_bearer_token_auth(engine: Engine) -> None:
    fake = FakeTmdb([movie(1, "Heat", 1995)])
    client = TmdbClient("eyJhbGciOi.token", ResponseCache(engine), settings(), transport=fake.transport)
    client.search_movie("Heat", 1995)
    req = fake.requests[0]
    assert req.headers["Authorization"] == "Bearer eyJhbGciOi.token"
    assert "api_key" not in req.url.params


def test_parse_minimal_details_does_not_crash() -> None:
    m = parse_movie({"id": 5, "title": "Sparse"}, [])
    assert m.tmdb_id == 5 and m.year is None
    assert m.genres == [] and m.directors == [] and m.cast == [] and m.imdb_id is None
