from __future__ import annotations

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select

from app.cache import DailyBudgetExceeded, ResponseCache
from app.db import ApiCache, Movie
from app.library import Library
from app.omdb import OmdbClient, OmdbRatings, parse_ratings
from tests.fake_omdb import FakeOmdb
from tests.test_pipeline import fake_for_sample, make_pipeline, run, test_settings


def test_parse_ratings() -> None:
    body = {
        "imdbRating": "8.1", "Metascore": "90",
        "Ratings": [
            {"Source": "Internet Movie Database", "Value": "8.0/10"},
            {"Source": "Rotten Tomatoes", "Value": "93%"},
            {"Source": "Metacritic", "Value": "88/100"},
        ],
    }
    assert parse_ratings(body) == OmdbRatings(imdb_rating=8.0, rt_score=93, metacritic=88)
    # No Ratings array: fall back to the top-level fields; "N/A" is missing.
    assert parse_ratings({"imdbRating": "7.4", "Metascore": "N/A"}) == OmdbRatings(7.4, None, None)
    assert parse_ratings({"imdbRating": "N/A", "Ratings": []}) == OmdbRatings()
    assert parse_ratings({"Ratings": [{"Source": "Rotten Tomatoes", "Value": "garbage"}]}) == OmdbRatings()


def client(engine: Engine, fake: FakeOmdb, **settings: object) -> OmdbClient:
    s = test_settings()
    for k, v in settings.items():
        setattr(s, k, v)
    return OmdbClient("secret-key", ResponseCache(engine), s, transport=fake.transport)


def test_client_caches_and_hides_key(engine: Engine) -> None:
    fake = FakeOmdb()
    c = client(engine, fake)
    first = c.ratings("tt0000101")
    assert first.imdb_rating == 6.0 and first.rt_score is not None
    assert c.ratings("tt0000101") == first and len(fake.requests) == 1  # cached
    assert fake.requests[0].url.params["apikey"] == "secret-key"
    with Session(engine) as s:
        assert all("secret-key" not in row.params_json for row in s.exec(select(ApiCache)))


def test_unknown_id_is_empty_and_cached(engine: Engine) -> None:
    fake = FakeOmdb()
    c = client(engine, fake)
    assert c.ratings("nonsense") == OmdbRatings()
    assert c.ratings("nonsense") == OmdbRatings() and len(fake.requests) == 1


def test_daily_budget(engine: Engine) -> None:
    c = client(engine, FakeOmdb(), omdb_daily_limit=2)
    c.ratings("tt0000101")
    c.ratings("tt0000102")
    c.ratings("tt0000101")  # cached: doesn't count
    with pytest.raises(DailyBudgetExceeded):
        c.ratings("tt0000103")


# ---------------------------------------------------------------- pipeline stage


def test_pipeline_fetches_the_shortlist_once(engine: Engine, library: Library) -> None:
    fake = FakeOmdb()
    p = make_pipeline(engine, fake_for_sample(), omdb=fake)
    r = run(p, library)
    stats = r.stats["omdb"]
    assert stats["fetched"] > 0 and stats["fetched"] == stats["shortlist"] - stats["no_imdb_id"]
    with Session(engine) as s:
        rated = s.exec(select(Movie).where(Movie.omdb_fetched_at.is_not(None))).all()  # type: ignore[union-attr]
    assert len(rated) == stats["fetched"] and all(m.imdb_rating is not None for m in rated)
    n = len(fake.requests)
    again = run(p, library).stats["omdb"]
    assert again["fetched"] == 0 and len(fake.requests) == n  # already looked up


def test_pipeline_shortlist_is_capped(engine: Engine, library: Library) -> None:
    p = make_pipeline(engine, fake_for_sample(), omdb=FakeOmdb())
    p.settings.omdb_shortlist_size = 3
    assert run(p, library).stats["omdb"]["shortlist"] == 3


def test_pipeline_without_key(engine: Engine, library: Library) -> None:
    r = run(make_pipeline(engine, fake_for_sample()), library)
    assert r.status == "done" and "skipped" in r.stats["omdb"]


def test_pipeline_survives_omdb_401(engine: Engine, library: Library) -> None:
    r = run(make_pipeline(engine, fake_for_sample(), omdb=FakeOmdb(status=401)), library)
    assert r.status == "done"  # recommendations don't need OMDb
    assert "skipped" in r.stats["omdb"] and r.message and "OMDb" in r.message


def test_pipeline_stops_at_budget(engine: Engine, library: Library) -> None:
    p = make_pipeline(engine, fake_for_sample(), omdb=FakeOmdb())
    assert p.omdb is not None
    p.omdb.http.daily_limit = 2
    r = run(p, library)
    assert r.stats["omdb"]["fetched"] == 2 and r.stats["omdb"]["budget_exhausted"] > 0
    assert r.message and "budget" in r.message
