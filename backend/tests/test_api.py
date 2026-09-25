from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, inspect
from sqlmodel import Session

from app import demo
from app.config import get_settings
from app.db import get_session
from app.letterboxd import make_film_key, parse_export
from app.library import Library
from app.main import app
from app.pipeline import get_pipeline
from app.sessions import InlineExecutor, SessionStore, get_session_store
from tests.fake_movielens import write_fake_movielens
from tests.fake_omdb import FakeOmdb
from tests.fake_tmdb import FakeTmdb, movie
from tests.fixtures.sample_export import WATCHED, build_files, build_zip
from tests.test_pipeline import SAMPLE_TMDB, fake_for_sample, make_pipeline


def make_client(engine: Engine, fake: FakeTmdb | None, omdb: FakeOmdb | None = None) -> TestClient:
    """A client on a fresh app state: its own DB, pipeline and session store.
    Pipeline runs execute inline, so an upload returns after processing."""

    def _session() -> Iterator[Session]:
        with Session(engine) as s:
            yield s

    pipeline = make_pipeline(engine, fake, omdb=omdb)
    store = SessionStore(pipeline.settings, executor=InlineExecutor())
    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_session_store] = lambda: store
    return TestClient(app)


def store_of() -> SessionStore:
    return app.dependency_overrides[get_session_store]()


@pytest.fixture
def client(engine: Engine) -> Iterator[TestClient]:
    fake = fake_for_sample()
    fake.movies[777] = movie(777, "Some Home Video", 2004)
    with make_client(engine, fake) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def keyless_client(engine: Engine) -> Iterator[TestClient]:
    with make_client(engine, None) as c:
        yield c
    app.dependency_overrides.clear()


def upload(c: TestClient, data: bytes, fixes: dict | None = None) -> dict:
    """Upload an export and make `c` send the new session's id from now on."""
    form = {"fixes": json.dumps(fixes)} if fixes is not None else {}
    resp = c.post("/api/sessions", files={"file": ("export.zip", data, "application/zip")}, data=form)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    c.headers["X-Session-Id"] = body["session_id"]
    return body


def library_of(c: TestClient) -> Library:
    session = store_of().get(c.headers["X-Session-Id"])
    assert session is not None
    return session.library


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body["features"]) == {"tmdb", "omdb", "openai"}


def test_upload_runs_pipeline(client: TestClient, sample_zip: bytes) -> None:
    body = upload(client, sample_zip)
    assert len(body["session_id"]) >= 40 and body["expires_in"] > 0
    assert body["stats"] == {"films": 35, "rated": 28, "watchlist": 6, "warnings": len(body["warnings"]),
                             "fixes_applied": 0}
    assert any("Heat" in w for w in body["warnings"])

    # The inline executor runs the pipeline before the upload returns.
    out = client.get("/api/session").json()
    assert out["stages"] == ["parsing", "matching", "enrichment", "collab", "candidates", "embedding", "taste", "blend", "omdb"]
    assert out["run"]["status"] == "done" and out["queue_position"] is None
    assert out["run"]["stats"]["matching"]["matched"] == 34
    assert out["run"]["stats"]["films"] == 35  # upload counts carried into the run
    assert "cancelled" not in out["run"]


def test_upload_rejects_garbage(client: TestClient, sample_zip: bytes) -> None:
    resp = client.post("/api/sessions", files={"file": ("x.zip", b"nope", "application/zip")})
    assert resp.status_code == 400 and "ZIP" in resp.json()["detail"]
    for bad in ("not json", "[1]", json.dumps({"x|": {"tmdb_id": "7"}})):
        resp = client.post("/api/sessions", files={"file": ("e.zip", sample_zip, "application/zip")},
                           data={"fixes": bad})
        assert resp.status_code == 400, bad
    assert len(store_of()) == 0
    upload(client, sample_zip)


def test_upload_without_tmdb_key(keyless_client: TestClient, sample_zip: bytes) -> None:
    upload(keyless_client, sample_zip)
    run = keyless_client.get("/api/session").json()["run"]
    assert run["status"] == "done" and "TMDB_API_KEY" in run["message"]


def test_endpoints_need_a_live_session(client: TestClient) -> None:
    for path in ("/api/session", "/api/matches", "/api/recommendations", "/api/taste", "/api/metrics"):
        assert client.get(path).status_code == 410, path
        assert client.get(path, headers={"X-Session-Id": "made-up"}).status_code == 410, path
    assert client.post("/api/session/reprocess").status_code == 410
    assert client.post("/api/matches/ignore", json={"film_key": "x"}).status_code == 410


def test_match_review_flow(client: TestClient, sample_zip: bytes) -> None:
    upload(client, sample_zip)
    review = client.get("/api/matches").json()
    assert review["counts"] == {"matched": 34, "unmatched": 1}
    [row] = review["rows"]
    assert row["name"] == "Home Movie Night" and row["movie"] is None

    all_rows = client.get("/api/matches", params={"filter": "all"}).json()["rows"]
    arrival = next(r for r in all_rows if r["name"] == "Arrival")
    assert arrival["movie"]["title"] == "Arrival" and arrival["movie"]["directors"]

    key = row["film_key"]
    bad = client.post("/api/matches/set", json={"film_key": key, "tmdb_ref": "not an id"})
    assert bad.status_code == 400
    missing = client.post("/api/matches/set", json={"film_key": key, "tmdb_ref": "424242"})
    assert missing.status_code == 404
    assert client.post("/api/matches/set", json={"film_key": "nope|", "tmdb_ref": "777"}).status_code == 404

    fixed = client.post(
        "/api/matches/set",
        json={"film_key": key, "tmdb_ref": "https://www.themoviedb.org/movie/777-some-home-video"},
    ).json()
    assert fixed["status"] == "manual" and fixed["tmdb_id"] == 777
    assert fixed["movie"]["title"] == "Some Home Video"
    assert client.get("/api/matches").json()["rows"] == []

    ignored = client.post("/api/matches/ignore", json={"film_key": key}).json()
    assert ignored["status"] == "ignored" and ignored["tmdb_id"] is None

    no_candidate = client.post("/api/matches/accept", json={"film_key": key})
    assert no_candidate.status_code == 400
    accepted = client.post("/api/matches/accept", json={"film_key": arrival["film_key"]}).json()
    assert accepted["status"] == "manual" and accepted["confidence"] == 1.0
    assert accepted["movie"]["title"] == "Arrival"


def test_matches_are_locked_while_processing(client: TestClient, sample_zip: bytes) -> None:
    upload(client, sample_zip)
    session = store_of().get(client.headers["X-Session-Id"])
    assert session is not None
    session.run.status = "running"
    resp = client.post("/api/matches/ignore", json={"film_key": make_film_key("Arrival", 2016)})
    assert resp.status_code == 409


def test_fixes_sent_with_the_upload_are_applied(client: TestClient, sample_zip: bytes) -> None:
    home = make_film_key("Home Movie Night", None)
    body = upload(client, sample_zip, {home: {"tmdb_id": 777}, "gone|1999": {"ignored": True}})
    assert body["stats"]["fixes_applied"] == 1
    assert client.get("/api/matches").json()["rows"] == []
    row = next(r for r in client.get("/api/matches", params={"filter": "all"}).json()["rows"]
               if r["film_key"] == home)
    assert row["status"] == "manual" and row["movie"]["title"] == "Some Home Video"


def test_reprocess_after_fixing(client: TestClient, sample_zip: bytes) -> None:
    upload(client, sample_zip)
    first = client.get("/api/session").json()["run"]
    home = make_film_key("Home Movie Night", None)
    client.post("/api/matches/set", json={"film_key": home, "tmdb_ref": "777"})
    out = client.post("/api/session/reprocess").json()
    assert out["run"]["id"] == first["id"] + 1 and out["run"]["status"] == "done"
    assert out["run"]["stats"]["matching"]["matched"] == 0  # only unresolved films are re-matched
    assert 777 in library_of(client).own_ids()


def test_recommendations(client: TestClient, sample_zip: bytes) -> None:
    upload(client, sample_zip)
    body = client.get("/api/recommendations", params={"limit": 100}).json()
    assert body["ready"] is True
    items = body["items"]
    titles = {i["title"] for i in items}

    assert not library_of(client).seen_ids() & {i["tmdb_id"] for i in items}
    assert {"Contact", "Solaris", "Enemy", "Annihilation", "Chungking Express"} <= titles
    assert "Obscure Short" not in titles  # filtered by candidate_min_votes
    past_lives = next(i for i in items if i["title"] == "Past Lives")
    assert past_lives["in_watchlist"] and past_lives["candidate_sources"]

    scores = [i["score"] for i in items]
    assert items[0]["score"] == max(scores)  # MMR reorders for variety, but always starts at the top
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert all(i["source_label"] for i in items)
    assert max(i["profile_score"] for i in items) == 1.0
    # 28 ratings < 50: fixed weights 0.3/0.4/0.3, re-weighted without ③; the score is
    # the percentile of that blend over the pool (+ a boost for watchlist films).
    assert body["blend_mode"] == "fixed"
    plain = [i for i in items if not i["in_watchlist"]]
    blend = {i["tmdb_id"]: (0.3 * i["profile_score"] + 0.4 * i["embedding_score"]) / 0.7 for i in plain}
    for a in plain:
        for b in plain:
            if blend[a["tmdb_id"]] > blend[b["tmdb_id"]] + 1e-9:
                assert a["score"] >= b["score"]
    assert all(i["predicted_rating"] is None for i in items)  # no learned model in fixed mode
    reasons = [r for i in items for r in i["profile_reasons"]]
    assert reasons and all({"label", "stars", "n", "contribution"} <= r.keys() for r in reasons)
    # Every sample film is Drama/Sci-Fi, so those carry no signal and aren't shown.
    assert all(abs(r["stars"]) >= 0.05 for r in reasons)
    assert any(r["label"].startswith("Decade ") for r in reasons)
    assert not any(r["label"].startswith("Genre ") for r in reasons)

    assert len(client.get("/api/recommendations", params={"limit": 3}).json()["items"]) == 3


def test_recommendations_with_collab(client: TestClient, sample_zip: bytes, tmp_path: Path) -> None:
    write_fake_movielens(tmp_path / "movielens" / "ml-latest-small", SAMPLE_TMDB)
    upload(client, sample_zip)
    body = client.get("/api/recommendations", params={"limit": 100}).json()
    assert body["collab_films"] == 28  # every rated sample film is in the fake MovieLens
    items = {i["title"]: i for i in body["items"]}
    contact = items["Contact"]
    assert contact["collab_score"] is not None and 0.5 <= contact["collab_predicted"] <= 5.0
    chungking = items["Chungking Express"]  # not in MovieLens: ③ is null and the blend uses ① and ②
    assert chungking["collab_score"] is None and chungking["collab_predicted"] is None
    assert 0.0 <= chungking["score"] <= 1.0


def test_recommendations_without_collab_model(client: TestClient, sample_zip: bytes) -> None:
    upload(client, sample_zip)
    body = client.get("/api/recommendations").json()
    assert body["ready"] and body["collab_films"] is None
    assert all(i["collab_score"] is None for i in body["items"])


def test_taste_endpoint(client: TestClient, sample_zip: bytes) -> None:
    upload(client, sample_zip)
    taste = client.get("/api/taste").json()
    assert taste["ready"] and taste["n_rated"] == 28  # every rated film matched
    # The fake hash embedder has little cluster structure; whatever k-means
    # finds must respect the minimum cluster size.
    assert len(taste["clusters"]) <= 6
    assert all(c["size"] >= 3 and c["examples"] for c in taste["clusters"])


def test_not_ready_without_a_taste_model(keyless_client: TestClient, sample_zip: bytes) -> None:
    upload(keyless_client, sample_zip)  # no TMDB: nothing matched, so no taste model
    assert keyless_client.get("/api/taste").json()["ready"] is False
    rec = keyless_client.get("/api/recommendations").json()
    assert rec["ready"] is False and "taste model" in rec["message"]


def test_recommendation_filters(client: TestClient, sample_zip: bytes) -> None:
    upload(client, sample_zip)
    base = client.get("/api/recommendations", params={"limit": 200}).json()
    assert base["matching"] == base["total"] == len(base["items"])
    assert base["facets"]["languages"]["ru"] == 1 and "Drama" in base["facets"]["genres"]

    def titles(**params: object) -> set[str]:
        body = client.get("/api/recommendations", params={"limit": 200, **params}).json()
        assert body["matching"] == len(body["items"]) <= body["total"] == base["total"]
        return {i["title"] for i in body["items"]}

    high = titles(min_rating=7.5)
    assert {"Solaris", "Chungking Express"} <= high
    assert not {"Contact", "Enemy", "Annihilation"} & high
    assert titles(language="ru") == {"Solaris"}
    assert titles(decade=1990, min_rating=8) == {"Chungking Express"}
    assert "Solaris" in titles(genre=["Horror", "Drama"])
    assert titles(genre=["Western"]) == set()

    # Scores don't change under filtering: they're ranks over the whole pool.
    by_id = {i["tmdb_id"]: i["score"] for i in base["items"]}
    filtered = client.get("/api/recommendations", params={"min_rating": 7.5}).json()["items"]
    assert all(by_id[i["tmdb_id"]] == i["score"] for i in filtered)

    # A limit applies after filtering, so strict filters still fill the page.
    assert len(client.get("/api/recommendations", params={"limit": 2, "min_rating": 7.5}).json()["items"]) == 2

    assert client.get("/api/recommendations", params={"min_rating": 11}).status_code == 422


def test_every_eligible_film_is_scored(client: TestClient, sample_zip: bytes) -> None:
    """No nearest-neighbour cutoff: every unseen candidate or watchlist film gets
    a score, however far it is from the taste vectors."""
    upload(client, sample_zip)
    eligible = library_of(client).eligible_ids()
    body = client.get("/api/recommendations", params={"limit": 200}).json()
    assert body["total"] == len(eligible) > 0
    assert {i["tmdb_id"] for i in body["items"]} == eligible


def test_omdb_ratings_and_filters(engine: Engine, sample_zip: bytes) -> None:
    with make_client(engine, fake_for_sample(), FakeOmdb()) as c:
        upload(c, sample_zip)
        items = c.get("/api/recommendations", params={"limit": 200}).json()["items"]
        assert all(i["imdb_rating"] is not None for i in items)  # the whole small pool is the shortlist
        assert any(i["rt_score"] is None for i in items) and any(i["metacritic"] is None for i in items)

        def ids(**params: object) -> set[int]:
            body = c.get("/api/recommendations", params={"limit": 200, **params})
            assert body.status_code == 200, body.text
            return {i["tmdb_id"] for i in body.json()["items"]}

        by_id = {i["tmdb_id"]: i for i in items}
        assert ids(rating_source="rt", min_rating=60) == {
            t for t, i in by_id.items() if i["rt_score"] is not None and i["rt_score"] >= 60}
        assert ids(rating_source="imdb", min_rating=8) == {
            t for t, i in by_id.items() if i["imdb_rating"] >= 8}
        assert ids(hide_low_quality=True) == {
            t for t, i in by_id.items() if (i["rt_score"] or 0) >= 60 or i["imdb_rating"] >= 6.5}
        assert c.get("/api/recommendations", params={"min_rating": 50}).status_code == 422  # TMDB is 0–10
        assert c.get("/api/recommendations", params={"rating_source": "letterboxd"}).status_code == 422
    app.dependency_overrides.clear()


def test_metrics(client: TestClient, sample_zip: bytes) -> None:
    upload(client, sample_zip)
    m = client.get("/api/metrics").json()
    assert m["ready"] and m["mode"] == "fixed" and m["n_ratings"] == 28  # < 50: fixed weights
    assert m["fixed_weights"] == {"profile": 0.3, "embedding": 0.4, "collab": 0.3}
    assert {"baseline_mean", "profile", "embedding", "fixed_blend", "learned_blend"} <= m["metrics"].keys()
    assert all(v["rmse"] > 0 for k, v in m["metrics"].items() if k != "ranking")
    assert m["metrics"]["ranking"]["rmse"] is None  # a ranking, not a rating prediction
    assert m["partial"]["features"] == ["profile", "embedding", "votes"]
    assert m["collab_model"] is None and m["omdb"]["enabled"] is False
    assert m["candidates"]["total"] > 0 and sum(m["candidates"]["by_source"].values()) > 0


# ---------------------------------------------------------------- sessions and privacy


def test_sessions_are_isolated(client: TestClient, sample_zip: bytes) -> None:
    upload(client, sample_zip)
    before = client.get("/api/recommendations", params={"limit": 200}).json()["items"]

    other = TestClient(app)
    upload(other, build_zip(build_files(username="someoneelse", watched=WATCHED[:12])))
    assert other.headers["X-Session-Id"] != client.headers["X-Session-Id"]
    theirs = {r["film_key"] for r in other.get("/api/matches", params={"filter": "all"}).json()["rows"]}
    mine = {r["film_key"] for r in client.get("/api/matches", params={"filter": "all"}).json()["rows"]}
    assert theirs < mine

    # Another user's upload, fix or deletion never changes this session.
    other.post("/api/matches/ignore", json={"film_key": make_film_key("Arrival", 2016)})
    other.delete("/api/session")
    after = client.get("/api/recommendations", params={"limit": 200}).json()["items"]
    assert [(i["tmdb_id"], i["score"]) for i in after] == [(i["tmdb_id"], i["score"]) for i in before]
    assert library_of(client).films[make_film_key("Arrival", 2016)].match_status == "matched"


def test_delete_session(client: TestClient, sample_zip: bytes) -> None:
    upload(client, sample_zip)
    assert client.delete("/api/session").json() == {"deleted": True}
    assert client.get("/api/recommendations").status_code == 410
    assert client.delete("/api/session").json() == {"deleted": False}
    assert len(store_of()) == 0


def test_nothing_about_the_user_is_stored(client: TestClient, engine: Engine, sample_zip: bytes, tmp_path: Path) -> None:
    upload(client, sample_zip)
    client.post("/api/matches/set", json={"film_key": make_film_key("Home Movie Night", None), "tmdb_ref": "777"})
    client.post("/api/session/reprocess")
    client.delete("/api/session")

    assert set(inspect(engine).get_table_names()) == {"apicache", "movie"}
    # data_dir is tmp_path: only the shared film database is on disk, no model files.
    stored = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert stored <= {"test.sqlite3", "test.sqlite3-wal", "test.sqlite3-shm"}


def test_upload_limits(engine: Engine, sample_zip: bytes) -> None:
    with make_client(engine, fake_for_sample()) as c:
        store = store_of()
        store.settings.max_export_films = 10
        resp = c.post("/api/sessions", files={"file": ("e.zip", sample_zip, "application/zip")})
        assert resp.status_code == 413 and "35 films" in resp.json()["detail"]
        store.settings.max_export_films = 5000

        store.settings.max_sessions = 1
        upload(c, sample_zip)
        busy = TestClient(app).post("/api/sessions", files={"file": ("e.zip", sample_zip, "application/zip")})
        assert busy.status_code == 503
        store.settings.max_sessions = 20

        store.settings.uploads_per_ip_per_hour = 4  # 3 used above
        upload(c, sample_zip)
        limited = c.post("/api/sessions", files={"file": ("e.zip", sample_zip, "application/zip")})
        assert limited.status_code == 429
    app.dependency_overrides.clear()


def test_demo_export_is_clean_and_big_enough_to_learn() -> None:
    export = parse_export(demo.demo_export_zip())
    assert export.warnings == []
    assert export.username == demo.USERNAME
    assert len(export.rated) == len(demo.RATED) >= get_settings().blend_min_ratings_for_learning
    assert len(export.watchlist) == len(demo.WATCHLIST)


def test_demo_session(client: TestClient) -> None:
    resp = client.post("/api/sessions/demo")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stats"]["films"] == len(demo.RATED) + len(demo.WATCHLIST)
    assert body["stats"]["fixes_applied"] == 0
    client.headers["X-Session-Id"] = body["session_id"]
    assert client.get("/api/session").json()["run"]["status"] == "done"


def test_demo_sessions_have_their_own_rate_limit(client: TestClient, sample_zip: bytes) -> None:
    settings = store_of().settings
    settings.uploads_per_ip_per_hour = 1
    settings.demo_sessions_per_ip_per_hour = 2
    upload(client, sample_zip)
    # Uploads used up; demos still work, up to their own limit.
    assert client.post("/api/sessions/demo").status_code == 200
    assert client.post("/api/sessions/demo").status_code == 200
    limited = client.post("/api/sessions/demo")
    assert limited.status_code == 429 and "Sample profile" in limited.json()["detail"]
    # Demos don't count toward the upload limit either way.
    settings.uploads_per_ip_per_hour = 2
    upload(client, sample_zip)
