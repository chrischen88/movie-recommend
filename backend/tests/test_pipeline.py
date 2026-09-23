from __future__ import annotations

import httpx
import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select

from app import matching
from app.cache import ApiError, ResponseCache
from app.config import Settings
from app.db import Candidate, IngestRun, Movie, UserFilm
from app.ingest import sync_export
from app.letterboxd import make_film_key, parse_export
from app.omdb import OmdbClient
from app.pipeline import Pipeline, TmdbUnavailable
from app.profile import FeatureStat, TasteProfile
from app.recommend import current_film_ids
from app.tmdb import TmdbClient
from pathlib import Path

from app.vectorstore import InMemoryStore
from tests.fake_embedder import HashEmbedder
from tests.fake_omdb import FakeOmdb
from tests.fake_movielens import fake_zip, write_fake_movielens
from tests.fake_tmdb import FakeTmdb, movie
from tests.fixtures.sample_export import WATCHED, WATCHLIST, build_files, build_zip


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
    # Films the user hasn't seen: only reachable through recommendations/similar.
    movies += [
        movie(501, "Contact", 1997, vote_average=7.4),
        movie(502, "Solaris", 1972, vote_average=8.0, original_language="ru"),
        movie(503, "Enemy", 2013, vote_average=6.9),
        movie(504, "Annihilation", 2018, vote_average=6.4),
        movie(505, "Chungking Express", 1994, vote_average=8.1, original_language="cn"),
        movie(506, "Obscure Short", 2020, votes=3),  # below candidate_min_votes
    ]
    return FakeTmdb(movies)


def test_settings(data_dir: Path | None = None) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        data_dir=data_dir or Path("."),
        tmdb_api_key="k",
        tmdb_requests_per_second=1000,
        http_max_retries=1,
        http_backoff_base_seconds=0.001,
        movielens_auto_download=False,  # tests never hit the network; see tests/fake_movielens.py
        candidate_collab_min_ratings=5,  # the fake MovieLens is tiny
    )


test_settings.__test__ = False  # type: ignore[attr-defined]


def make_pipeline(
    engine: Engine,
    fake: FakeTmdb | None,
    embedder: HashEmbedder | None = None,
    store: InMemoryStore | None = None,
    omdb: FakeOmdb | None = None,
) -> Pipeline:
    # Keep taste_model.json next to the test DB (a tmp dir).
    s = test_settings(Path(str(engine.url.database)).parent)
    tmdb = TmdbClient("k", ResponseCache(engine), s, transport=fake.transport) if fake else None
    return Pipeline(
        engine,
        s,
        tmdb,
        embedder=embedder or HashEmbedder(),
        store=store if store is not None else InMemoryStore(),
        max_workers=4,
        omdb=OmdbClient("secret", ResponseCache(engine), s, transport=omdb.transport) if omdb else None,
    )


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


def statuses(engine: Engine) -> dict[str, str | None]:
    with Session(engine) as s:
        return {f.name: f.match_status for f in s.exec(select(UserFilm))}


def test_full_run(ingested: Engine) -> None:
    fake = fake_for_sample()
    r = run(make_pipeline(ingested, fake), ingested)
    assert r.status == "done" and r.stage == "done"
    assert r.stats["matching"] == {"matched": 34, "low_confidence": 0, "unmatched": 1, "error": 0}
    assert r.stats["enrichment"]["enriched"] == 34

    st = statuses(ingested)
    assert st["Home Movie Night"] == matching.UNMATCHED
    with Session(ingested) as s:
        heat = s.get(UserFilm, make_film_key("Heat", 1995))
        assert heat and heat.match_confidence == pytest.approx(0.9)
        # 34 user films + 5 candidates (the 3-vote "Obscure Short" is filtered out)
        assert len(s.exec(select(Movie)).all()) == 39


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
    # Anora was already fetched as a candidate in the first run: no refetch.
    assert r.stats["enrichment"] == {"enriched": 0, "not_found": 0, "errors": 0, "already_had": 35}
    # Only: search for Anora + its recommendations/similar lists as a new 4.5★ seed.
    assert len(fake.requests) - n == 3


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


# ---------------------------------------------------------------- collaborative model (M5)

SAMPLE_TMDB = list(range(100, 140)) + [501, 502, 503, 504]  # not 505 (Chungking Express)


def test_collab_skipped_without_dataset(ingested: Engine) -> None:
    r = run(make_pipeline(ingested, fake_for_sample()), ingested)
    assert r.status == "done"  # score ③ is optional
    assert "skipped" in r.stats["collab"]
    assert r.message and "collaborative score unavailable" in r.message


def test_collab_trains_once(ingested: Engine) -> None:
    p = make_pipeline(ingested, fake_for_sample())
    write_fake_movielens(p.settings.movielens_dir, SAMPLE_TMDB)
    r = run(p, ingested)
    assert r.stats["collab"]["trained"] is True
    assert r.stats["collab"]["val_rmse"] < r.stats["collab"]["val_rmse_global_mean"]
    assert p.settings.collab_model_path.exists()

    assert run(p, ingested).stats["collab"]["trained"] is False  # settings unchanged: no-op
    assert p.train_collab(force=True)["trained"] is True
    p.settings.collab_factors = 4  # a new fingerprint retrains
    assert p.train_collab()["trained"] is True


def test_collab_downloads_when_allowed(ingested: Engine) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=fake_zip("ml-latest-small", SAMPLE_TMDB))

    p = make_pipeline(ingested, fake_for_sample())
    p.settings.movielens_auto_download = True
    p.movielens_transport = httpx.MockTransport(handler)
    assert p.train_collab()["trained"] is True
    assert len(calls) == 1 and calls[0].endswith("/ml-latest-small.zip")
    assert p.train_collab()["trained"] is False and len(calls) == 1


def test_collab_disabled(ingested: Engine) -> None:
    p = make_pipeline(ingested, fake_for_sample())
    p.settings.collab_enabled = False
    assert "disabled" in str(p.train_collab()["skipped"])


# ---------------------------------------------------------------- replacing the library


def _seeds(engine: Engine) -> set[int]:
    with Session(engine) as s:
        return {
            f.tmdb_id for f in s.exec(select(UserFilm))
            if f.tmdb_id is not None and (f.rating or 0) >= 4.0
        }


def test_candidates_are_replaced_not_accumulated(ingested: Engine) -> None:
    p = make_pipeline(ingested, fake_for_sample())
    run(p, ingested)
    with Session(ingested) as s:
        s.add(Candidate(tmdb_id=424243, sources=["similar:424242"]))  # 424242 isn't a seed
        stale = s.get(Candidate, 503)
        assert stale is not None
        stale.sources = [*stale.sources, "recommendations:424242"]
        s.add(stale)
        s.commit()
    r = run(p, ingested)
    assert r.stats["candidates"]["removed"] == 1 and r.stats["candidates"]["failed_sources"] == 0
    seeds = _seeds(ingested)
    with Session(ingested) as s:
        for c in s.exec(select(Candidate)):
            assert c.sources and all(int(src.split(":")[1]) in seeds for src in c.sources), c


def test_failed_seed_keeps_its_candidates(ingested: Engine) -> None:
    p = make_pipeline(ingested, fake_for_sample())
    run(p, ingested)
    assert p.tmdb is not None
    seed = min(_seeds(ingested))
    with Session(ingested) as s:
        s.add(Candidate(tmdb_id=424242, sources=[f"similar:{seed}"]))
        s.commit()
    original = p.tmdb.similar

    def flaky(tmdb_id: int) -> list[dict]:
        if tmdb_id == seed:
            raise ApiError("boom", 500)
        return original(tmdb_id)

    p.tmdb.similar = flaky  # type: ignore[method-assign]
    r = run(p, ingested)
    assert r.stats["candidates"]["failed_sources"] == 1
    with Session(ingested) as s:
        kept = s.get(Candidate, 424242)
        assert kept is not None and kept.sources == [f"similar:{seed}"]


def test_index_only_holds_current_library(ingested: Engine) -> None:
    store = InMemoryStore()
    p = make_pipeline(ingested, fake_for_sample(), store=store)
    run(p, ingested)
    # A film left over from an earlier library: cached metadata plus an index entry.
    with Session(ingested) as s:
        s.add(Movie(tmdb_id=424244, title="Someone Else's Favourite", genres=["Drama"]))
        s.commit()
    store.upsert([424244], HashEmbedder().embed(["x"]), [{"tmdb_id": 424244, "seen": False}], ["x"])
    r = run(p, ingested)
    assert r.stats["embedding"]["removed"] == 1
    with Session(ingested) as s:
        assert set(store.get_metadata()) == current_film_ids(s) & {m.tmdb_id for m in s.exec(select(Movie))}
        assert s.get(Movie, 424244) is not None  # kept as cached metadata, just not indexed
    assert 424244 not in store.get_metadata()


# ---------------------------------------------------------------- candidate sources (M6)


def test_discover_jobs_follow_the_profile(ingested: Engine) -> None:
    p = make_pipeline(ingested, fake_for_sample())
    profile = TasteProfile(mean_rating=3.5, n_rated=30, stats={
        ("genre", "Drama"): FeatureStat(0.4, 10),
        ("genre", "Science Fiction"): FeatureStat(-0.2, 8),  # disliked: no query
        ("genre", "Western"): FeatureStat(0.9, 2),  # too few films: no query
        ("language", "ru"): FeatureStat(0.5, 4),
        ("language", "en"): FeatureStat(0.6, 20),  # English is the default pool anyway
    })
    jobs = p._discover_jobs(profile, lambda src, results: [(src, r["id"], r["vote_count"]) for r in results])
    assert sorted(jobs) == ["discover:genre:Drama", "discover:lang:ru"]
    drama = jobs["discover:genre:Drama"]()
    assert drama and {src for src, _, _ in drama} == {"discover:genre:Drama"}
    assert all(votes >= p.settings.candidate_discover_min_votes for _, _, votes in drama)
    assert [tid for _, tid, _ in jobs["discover:lang:ru"]()] == [502]  # Solaris
    assert p._discover_jobs(None, lambda *_: []) == {}


def test_collab_candidates(ingested: Engine) -> None:
    p = make_pipeline(ingested, fake_for_sample())
    write_fake_movielens(p.settings.movielens_dir, SAMPLE_TMDB + [506])
    r = run(p, ingested)
    assert r.stats["candidates"]["by_source"].get("collab", 0) > 0
    with Session(ingested) as s:
        sources = {c.tmdb_id: c.sources for c in s.exec(select(Candidate))}
    assert any("collab" in src for src in sources.values())
    assert 506 not in sources  # "Obscure Short": 3 TMDB votes, dropped after enrichment
    assert not _seeds(ingested) & set(sources)  # never a film the user has seen


def test_collab_source_needs_a_model(ingested: Engine) -> None:
    p = make_pipeline(ingested, fake_for_sample())
    assert p._collab_job({100: 4.0}, set()) is None
