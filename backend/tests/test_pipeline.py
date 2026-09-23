from __future__ import annotations

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select

from app import matching
from app.cache import ResponseCache
from app.config import Settings
from app.db import IngestRun, Movie, UserFilm
from app.ingest import sync_export
from app.letterboxd import make_film_key, parse_export
from app.pipeline import Pipeline, TmdbUnavailable
from app.tmdb import TmdbClient
from tests.fake_tmdb import FakeTmdb, movie
from tests.fixtures.sample_export import WATCHED, WATCHLIST, build_zip


def fake_for_sample() -> FakeTmdb:
    movies = []
    seen = set()
    for i, (name, year, *_rest) in enumerate([*WATCHED, *WATCHLIST], start=100):
        if not year or (name, year) in seen:
            continue  # "Home Movie Night" isn't on TMDB
        seen.add((name, year))
        y = int(year)
        if name == "Heat":
            y = 1996  # TMDB disagrees by a year -> found via ±1 retry
        movies.append(movie(i, name, y))
    return FakeTmdb(movies)


def test_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        tmdb_api_key="k",
        tmdb_requests_per_second=1000,
        http_max_retries=1,
        http_backoff_base_seconds=0.001,
    )


test_settings.__test__ = False  # type: ignore[attr-defined]


def make_pipeline(engine: Engine, fake: FakeTmdb | None) -> Pipeline:
    s = test_settings()
    tmdb = TmdbClient("k", ResponseCache(engine), s, transport=fake.transport) if fake else None
    return Pipeline(engine, s, tmdb, max_workers=4)


def run(p: Pipeline, engine: Engine) -> IngestRun:
    with Session(engine) as s:
        r = IngestRun()
        s.add(r)
        s.commit()
        s.refresh(r)
    assert p.try_acquire()
    assert r.id is not None
    p.run(r.id)
    assert not p.busy
    with Session(engine) as s:
        out = s.get(IngestRun, r.id)
        assert out is not None
        return out


@pytest.fixture
def ingested(engine: Engine, sample_zip: bytes) -> Engine:
    with Session(engine) as s:
        sync_export(s, parse_export(sample_zip))
    return engine


def statuses(engine: Engine) -> dict[str, str | None]:
    with Session(engine) as s:
        return {f.name: f.match_status for f in s.exec(select(UserFilm))}


def test_full_run(ingested: Engine) -> None:
    fake = fake_for_sample()
    r = run(make_pipeline(ingested, fake), ingested)
    assert r.status == "done" and r.stage == "done"
    assert r.stats["matching"] == {"matched": 34, "low_confidence": 0, "unmatched": 1, "error": 0}
    assert r.stats["enrichment"]["enriched"] == 34
    assert r.progress_done == r.progress_total == 34

    st = statuses(ingested)
    assert st["Home Movie Night"] == matching.UNMATCHED
    with Session(ingested) as s:
        heat = s.get(UserFilm, make_film_key("Heat", 1995))
        assert heat and heat.match_confidence == pytest.approx(0.9)
        assert len(s.exec(select(Movie)).all()) == 34


def test_second_run_makes_no_requests(ingested: Engine) -> None:
    fake = fake_for_sample()
    p = make_pipeline(ingested, fake)
    run(p, ingested)
    n = len(fake.requests)
    r = run(p, ingested)
    assert len(fake.requests) == n
    assert r.stats["matching"]["matched"] == 0
    assert r.stats["enrichment"]["enriched"] == 0


def test_new_film_in_later_export_only_processes_that_film(ingested: Engine) -> None:
    fake = fake_for_sample()
    fake.movies[9001] = movie(9001, "Anora", 2024)
    p = make_pipeline(ingested, fake)
    run(p, ingested)
    n = len(fake.requests)

    from tests.fixtures.sample_export import ROOT, build_files

    files = build_files()
    files[f"{ROOT}/ratings.csv"] += b"2026-09-01,Anora,2024,https://boxd.it/n01,4.5\n"
    with Session(ingested) as s:
        sync_export(s, parse_export(build_zip(files)))
    r = run(p, ingested)
    assert r.stats["matching"]["matched"] == 1
    assert r.stats["enrichment"]["enriched"] == 1
    assert len(fake.requests) - n == 3  # search + details + reviews


def test_missing_key_skips_gracefully(ingested: Engine) -> None:
    r = run(make_pipeline(ingested, None), ingested)
    assert r.status == "done" and r.message and "TMDB_API_KEY" in r.message
    assert set(statuses(ingested).values()) == {None}


def test_bad_key_aborts_with_error(ingested: Engine) -> None:
    fake = fake_for_sample()
    fake.status_override = 401
    r = run(make_pipeline(ingested, fake), ingested)
    assert r.status == "error" and r.message and "401" in r.message


def test_enrichment_failure_marks_film_and_continues(ingested: Engine) -> None:
    fake = fake_for_sample()
    tenet_id = next(i for i, m in fake.movies.items() if m["title"] == "Tenet")
    fake.fail_ids.add(tenet_id)
    r = run(make_pipeline(ingested, fake), ingested)
    assert r.status == "done"
    assert r.stats["enrichment"] == {"enriched": 33, "not_found": 0, "errors": 1, "already_had": 0}
    assert statuses(ingested)["Tenet"] == matching.ERROR

    # Next run retries the errored film once TMDB recovers.
    fake.fail_ids.clear()
    r = run(make_pipeline(ingested, fake), ingested)
    assert statuses(ingested)["Tenet"] == matching.MATCHED
    assert r.stats["enrichment"]["enriched"] == 1


def test_manual_match_and_ignore(ingested: Engine) -> None:
    fake = fake_for_sample()
    fake.movies[777] = movie(777, "Some Home Video", 2004)
    p = make_pipeline(ingested, fake)
    run(p, ingested)

    key = make_film_key("Home Movie Night", None)
    film, mv = p.set_manual_match(key, 777)
    assert film.tmdb_id == 777 and film.match_status == matching.MANUAL
    assert mv.title == "Some Home Video"
    with pytest.raises(LookupError):
        p.set_manual_match(key, 123456)
    with pytest.raises(KeyError):
        p.set_manual_match("nope|", 777)

    film = p.set_status(key, matching.IGNORED, "ignored")
    assert film.tmdb_id is None and film.match_status == matching.IGNORED

    # Manual/ignored decisions survive a later run.
    run(p, ingested)
    assert statuses(ingested)["Home Movie Night"] == matching.IGNORED


def test_manual_match_requires_key(ingested: Engine) -> None:
    with pytest.raises(TmdbUnavailable):
        make_pipeline(ingested, None).set_manual_match(make_film_key("Heat", 1995), 1)
