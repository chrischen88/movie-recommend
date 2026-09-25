from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine, inspect, text
from sqlmodel import Session, select

from app import matching
from app.cache import ApiError, ResponseCache
from app.config import Settings
from app.db import LEGACY_USER_FILES, LEGACY_USER_TABLES, Movie
from app.letterboxd import make_film_key, parse_export
from app.library import InvalidFixes, Library, library_from_export
from app.omdb import OmdbClient
from app.pipeline import Pipeline, TmdbUnavailable
from app.profile import FeatureStat, TasteProfile
from app.sessions import RunState
from app.tmdb import TmdbClient

from app.vectorstore import InMemoryStore
from tests.fake_embedder import HashEmbedder
from tests.fake_omdb import FakeOmdb
from tests.fake_movielens import fake_zip, write_fake_movielens
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
    # data_dir next to the test DB (a tmp dir): the MovieLens model goes there.
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




def run(p: Pipeline, lib: Library) -> RunState:
    r = RunState(status="running")
    p.run(lib, r)
    return r


def statuses(lib: Library) -> dict[str, str | None]:
    return {f.name: f.match_status for f in lib.films.values()}


def fresh(sample_zip: bytes, fixes: dict | None = None) -> Library:
    return library_from_export(parse_export(sample_zip), fixes)


def test_full_run(engine: Engine, library: Library) -> None:
    fake = fake_for_sample()
    r = run(make_pipeline(engine, fake), library)
    assert r.status == "done" and r.stage == "done"
    assert r.stats["matching"] == {"matched": 34, "low_confidence": 0, "unmatched": 1, "error": 0}
    assert r.stats["enrichment"]["enriched"] == 34

    assert statuses(library)["Home Movie Night"] == matching.UNMATCHED
    heat = library.films[make_film_key("Heat", 1995)]
    assert heat.match_confidence == pytest.approx(0.9)
    assert library.taste is not None and library.blend is not None
    assert library.candidates and not set(library.candidates) & library.seen_ids()
    with Session(engine) as s:
        # 34 user films + 5 candidates (the 3-vote "Obscure Short" is filtered out)
        assert len(s.exec(select(Movie)).all()) == 39


def test_second_user_makes_no_requests(engine: Engine, sample_zip: bytes) -> None:
    """Film data is shared: a second session with the same films is served
    entirely from the API cache, the Movie table and the index."""
    fake = fake_for_sample()
    store = InMemoryStore()
    p = make_pipeline(engine, fake, store=store)
    first = fresh(sample_zip)
    run(p, first)
    n, indexed = len(fake.requests), store.count()

    second = fresh(sample_zip)
    r = run(p, second)
    assert r.status == "done"
    assert len(fake.requests) == n
    assert r.stats["matching"]["matched"] == 34  # re-matched, from cached searches
    assert r.stats["enrichment"] == {"enriched": 0, "not_found": 0, "errors": 0, "already_had": 34}
    assert r.stats["embedding"]["embedded"] == 0 and store.count() == indexed
    assert second.candidates == first.candidates


def test_missing_key_skips_gracefully(engine: Engine, library: Library) -> None:
    r = run(make_pipeline(engine, None), library)
    assert r.status == "done" and r.message and "TMDB_API_KEY" in r.message
    assert set(statuses(library).values()) == {None}


def test_bad_key_aborts_with_error(engine: Engine, library: Library) -> None:
    fake = fake_for_sample()
    fake.status_override = 401
    r = run(make_pipeline(engine, fake), library)
    assert r.status == "error" and r.message and "401" in r.message


def test_enrichment_failure_marks_film_and_continues(engine: Engine, library: Library) -> None:
    fake = fake_for_sample()
    tenet_id = next(i for i, m in fake.movies.items() if m["title"] == "Tenet")
    fake.fail_ids.add(tenet_id)
    p = make_pipeline(engine, fake)
    r = run(p, library)
    assert r.status == "done"
    assert r.stats["enrichment"] == {"enriched": 33, "not_found": 0, "errors": 1, "already_had": 0}
    assert statuses(library)["Tenet"] == matching.ERROR

    # Reprocessing retries the errored film once TMDB recovers.
    fake.fail_ids.clear()
    r = run(p, library)
    assert statuses(library)["Tenet"] == matching.MATCHED
    assert r.stats["enrichment"]["enriched"] == 1


def test_match_fixes_are_applied_and_skip_matching(engine: Engine, sample_zip: bytes) -> None:
    fake = fake_for_sample()
    fake.movies[777] = movie(777, "Some Home Video", 2004)
    home, heat = make_film_key("Home Movie Night", None), make_film_key("Heat", 1995)
    lib = fresh(sample_zip, {home: {"tmdb_id": 777}, heat: {"ignored": True}, "not|in export": {"tmdb_id": 1}})
    r = run(make_pipeline(engine, fake), lib)
    assert r.stats["matching"]["matched"] == 33  # 35 films − 2 fixed
    assert lib.films[home].match_status == matching.MANUAL and lib.films[home].tmdb_id == 777
    assert lib.films[heat].match_status == matching.IGNORED and lib.films[heat].tmdb_id is None
    with Session(engine) as s:
        assert s.get(Movie, 777) is not None  # a fixed film is enriched like any other

    for bad in ({home: {"tmdb_id": "777"}}, {home: {"tmdb_id": 0}}, {home: {}}, {home: 5}):
        with pytest.raises(InvalidFixes):
            fresh(sample_zip, bad)


def test_library_edits() -> None:
    lib = library_from_export(parse_export(build_zip()))
    key = make_film_key("Arrival", 2016)
    film = lib.set_manual(key, 42, "set by hand")
    assert film.tmdb_id == 42 and film.match_confidence == 1.0 and film.match_status == matching.MANUAL
    film = lib.set_status(key, matching.IGNORED, "ignored")
    assert film.tmdb_id is None and film.match_confidence is None
    with pytest.raises(KeyError):
        lib.film("nope|")


def test_lookup_movie(engine: Engine) -> None:
    fake = fake_for_sample()
    fake.movies[777] = movie(777, "Some Home Video", 2004)
    p = make_pipeline(engine, fake)
    assert p.lookup_movie(777).title == "Some Home Video"
    with pytest.raises(LookupError):
        p.lookup_movie(123456)
    keyless = make_pipeline(engine, None)
    assert keyless.lookup_movie(777).title == "Some Home Video"  # cached: no key needed
    with pytest.raises(TmdbUnavailable):
        keyless.lookup_movie(778)


def test_cancelled_run_stops(engine: Engine, library: Library) -> None:
    fake = fake_for_sample()
    r = RunState(status="running", cancelled=True)
    make_pipeline(engine, fake).run(library, r)
    assert r.status == "cancelled" and r.finished_at is not None
    assert fake.requests == []


# ---------------------------------------------------------------- collaborative model (M5)

SAMPLE_TMDB = list(range(100, 140)) + [501, 502, 503, 504]  # not 505 (Chungking Express)


def test_collab_skipped_without_dataset(engine: Engine, library: Library) -> None:
    r = run(make_pipeline(engine, fake_for_sample()), library)
    assert r.status == "done"  # score ③ is optional
    assert "skipped" in r.stats["collab"]
    assert r.message and "collaborative score unavailable" in r.message


def test_collab_trains_once(engine: Engine, library: Library) -> None:
    p = make_pipeline(engine, fake_for_sample())
    write_fake_movielens(p.settings.movielens_dir, SAMPLE_TMDB)
    r = run(p, library)
    assert r.stats["collab"]["trained"] is True
    assert r.stats["collab"]["val_rmse"] < r.stats["collab"]["val_rmse_global_mean"]
    assert p.settings.collab_model_path.exists()

    assert run(p, library).stats["collab"]["trained"] is False  # settings unchanged: no-op
    assert p.train_collab(force=True)["trained"] is True
    p.settings.collab_factors = 4  # a new fingerprint retrains
    assert p.train_collab()["trained"] is True


def test_collab_downloads_when_allowed(engine: Engine) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=fake_zip("ml-latest-small", SAMPLE_TMDB))

    p = make_pipeline(engine, fake_for_sample())
    p.settings.movielens_auto_download = True
    p.movielens_transport = httpx.MockTransport(handler)
    assert p.train_collab()["trained"] is True
    assert len(calls) == 1 and calls[0].endswith("/ml-latest-small.zip")
    assert p.train_collab()["trained"] is False and len(calls) == 1


def test_collab_disabled(engine: Engine) -> None:
    p = make_pipeline(engine, fake_for_sample())
    p.settings.collab_enabled = False
    assert "disabled" in str(p.train_collab()["skipped"])


# ---------------------------------------------------------------- candidates


def _seeds(lib: Library) -> set[int]:
    return {tid for tid, r in lib.ratings().items() if r >= 4.0}


def test_candidates_come_from_the_library_seeds(engine: Engine, library: Library) -> None:
    r = run(make_pipeline(engine, fake_for_sample()), library)
    assert r.stats["candidates"]["failed_sources"] == 0
    assert r.stats["candidates"]["total"] == len(library.candidates) > 0
    seeds = _seeds(library)
    for tid, sources in library.candidates.items():
        seed_sources = [src for src in sources if src.split(":")[0] in ("recommendations", "similar")]
        assert all(int(src.split(":")[1]) in seeds for src in seed_sources), (tid, sources)


def test_failed_seed_is_reported_and_the_run_continues(engine: Engine, library: Library) -> None:
    p = make_pipeline(engine, fake_for_sample())
    assert p.tmdb is not None
    run(p, library)  # learn the seeds
    seed = min(_seeds(library))
    original = p.tmdb.similar

    def flaky(tmdb_id: int) -> list[dict]:
        if tmdb_id == seed:
            raise ApiError("boom", 500)
        return original(tmdb_id)

    p.tmdb.similar = flaky  # type: ignore[method-assign]
    r = run(p, library)
    assert r.status == "done" and r.stats["candidates"]["failed_sources"] == 1
    assert library.candidates


def test_index_is_shared_and_holds_no_user_flags(engine: Engine, library: Library) -> None:
    store = InMemoryStore()
    p = make_pipeline(engine, fake_for_sample(), store=store)
    # A film another user's session embedded: stays in the shared index.
    with Session(engine) as s:
        s.add(Movie(tmdb_id=424244, title="Someone Else's Favourite", genres=["Drama"]))
        s.commit()
    store.upsert([424244], HashEmbedder().embed(["x"]), [{"tmdb_id": 424244}], ["x"])
    run(p, library)
    meta = store.get_metadata()
    assert 424244 in meta
    assert library.current_ids() <= set(meta)  # everything this library needs is indexed
    assert not any("seen" in m for m in meta.values())


def test_purge_user_data(engine: Engine, tmp_path: Path) -> None:
    """Data older versions stored about the user is removed at startup."""
    store = InMemoryStore()
    p = make_pipeline(engine, None, store=store)
    with engine.begin() as conn:
        for table in LEGACY_USER_TABLES:
            conn.execute(text(f'CREATE TABLE "{table}" (id INTEGER PRIMARY KEY, secret TEXT)'))
            conn.execute(text(f"INSERT INTO \"{table}\" (secret) VALUES ('my rating')"))
    for name in LEGACY_USER_FILES:
        (p.settings.data_dir / name).write_text("{}")
    store.upsert([1, 2], HashEmbedder().embed(["a", "b"]), [{"tmdb_id": 1, "seen": True}, {"tmdb_id": 2}], ["a", "b"])

    removed = p.purge_user_data()
    assert set(LEGACY_USER_TABLES) | set(LEGACY_USER_FILES) <= set(removed)
    assert set(inspect(engine).get_table_names()) == {"apicache", "movie"}
    assert not any((p.settings.data_dir / name).exists() for name in LEGACY_USER_FILES)
    assert store.get_metadata() == {1: {"tmdb_id": 1}, 2: {"tmdb_id": 2}}
    assert p.purge_user_data() == []  # idempotent


# ---------------------------------------------------------------- candidate sources (M6)


def test_discover_jobs_follow_the_profile(engine: Engine) -> None:
    p = make_pipeline(engine, fake_for_sample())
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


def test_collab_candidates(engine: Engine, library: Library) -> None:
    p = make_pipeline(engine, fake_for_sample())
    write_fake_movielens(p.settings.movielens_dir, SAMPLE_TMDB + [506])
    r = run(p, library)
    assert r.stats["candidates"]["by_source"].get("collab", 0) > 0
    sources = library.candidates
    assert any("collab" in src for src in sources.values())
    assert 506 not in sources  # "Obscure Short": 3 TMDB votes, dropped after enrichment
    assert not library.seen_ids() & set(sources)  # never a film the user has seen


def test_collab_source_needs_a_model(engine: Engine) -> None:
    p = make_pipeline(engine, fake_for_sample())
    assert p._collab_job({100: 4.0}, set()) is None
