from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlmodel import Session, select

from app.db import UserFilm, get_session, make_engine
from app.main import app
from app.pipeline import get_pipeline
from app.recommend import current_film_ids
from tests.fake_movielens import write_fake_movielens
from tests.fake_omdb import FakeOmdb
from tests.fake_tmdb import FakeTmdb, movie
from tests.fixtures.sample_export import WATCHED, build_files, build_zip
from tests.test_pipeline import SAMPLE_TMDB, fake_for_sample, make_pipeline


def make_client(engine: Engine, fake: FakeTmdb | None, omdb: FakeOmdb | None = None) -> TestClient:
    def _session() -> Iterator[Session]:
        with Session(engine) as s:
            yield s

    pipeline = make_pipeline(engine, fake, omdb=omdb)
    app.dependency_overrides[get_session] = _session
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


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


def upload(c: TestClient, data: bytes) -> dict:
    resp = c.post("/api/upload", files={"file": ("export.zip", data, "application/zip")})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body["features"]) == {"tmdb", "omdb", "openai"}


def test_upload_runs_pipeline(client: TestClient, sample_zip: bytes) -> None:
    body = upload(client, sample_zip)
    assert body["stats"]["added"] == 35
    assert body["stats"]["rated"] == 28
    assert any("Heat" in w for w in body["warnings"])

    # TestClient runs background tasks before returning.
    out = client.get(f"/api/ingest/{body['run_id']}").json()
    assert out["stages"] == ["parsing", "matching", "enrichment", "collab", "candidates", "embedding", "taste", "blend", "omdb"]
    assert out["run"]["status"] == "done"
    assert out["run"]["stats"]["matching"]["matched"] == 34
    assert client.get("/api/ingest/latest").json()["run"]["id"] == body["run_id"]

    assert len(client.get("/api/films").json()) == 35
    assert upload(client, sample_zip)["stats"]["unchanged"] == 35


def test_upload_rejects_garbage_and_releases_lock(client: TestClient, sample_zip: bytes) -> None:
    resp = client.post("/api/upload", files={"file": ("x.zip", b"nope", "application/zip")})
    assert resp.status_code == 400
    assert "ZIP" in resp.json()["detail"]
    upload(client, sample_zip)  # lock was released


def test_upload_without_tmdb_key(keyless_client: TestClient, sample_zip: bytes) -> None:
    body = upload(keyless_client, sample_zip)
    run = keyless_client.get(f"/api/ingest/{body['run_id']}").json()["run"]
    assert run["status"] == "done" and "TMDB_API_KEY" in run["message"]


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

    fixed = client.post(
        "/api/matches/set",
        json={"film_key": key, "tmdb_ref": "https://www.themoviedb.org/movie/777-some-home-video"},
    ).json()
    assert fixed["status"] == "manual" and fixed["movie"]["title"] == "Some Home Video"
    assert client.get("/api/matches").json()["rows"] == []

    ignored = client.post("/api/matches/ignore", json={"film_key": key}).json()
    assert ignored["status"] == "ignored" and ignored["tmdb_id"] is None

    no_candidate = client.post("/api/matches/accept", json={"film_key": key})
    assert no_candidate.status_code == 400
    accepted = client.post("/api/matches/accept", json={"film_key": arrival["film_key"]}).json()
    assert accepted["status"] == "manual" and accepted["confidence"] == 1.0


def test_recommendations(client: TestClient, sample_zip: bytes) -> None:
    before = client.get("/api/recommendations").json()
    assert before["ready"] is False and "taste model" in before["message"]

    upload(client, sample_zip)
    body = client.get("/api/recommendations", params={"limit": 100}).json()
    assert body["ready"] is True
    items = body["items"]
    titles = {i["title"] for i in items}

    watched = {f["tmdb_id"] for f in client.get("/api/films").json() if f["watched"]}
    assert not watched & {i["tmdb_id"] for i in items}
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
    assert client.get("/api/taste").json()["ready"] is False
    upload(client, sample_zip)
    taste = client.get("/api/taste").json()
    assert taste["ready"] and taste["n_rated"] == 28  # every rated film matched
    # The fake hash embedder has little cluster structure; whatever k-means
    # finds must respect the minimum cluster size.
    assert len(taste["clusters"]) <= 6
    assert all(c["size"] >= 3 and c["examples"] for c in taste["clusters"])


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


def _snapshot(c: TestClient) -> dict[str, object]:
    films = sorted(
        (f["film_key"], f["tmdb_id"], f["match_status"], f["rating"], f["watched"])
        for f in c.get("/api/films").json()
    )
    recs = c.get("/api/recommendations", params={"limit": 200}).json()
    taste = c.get("/api/taste").json()
    return {
        "films": films,
        "recs": [(i["tmdb_id"], round(i["score"], 9), i["candidate_sources"]) for i in recs["items"]],
        "total": recs["total"],
        "taste": (taste["n_rated"], [(cl["label"], cl["size"]) for cl in taste["clusters"]]),
    }


def test_other_account_upload_equals_fresh_install(engine: Engine, tmp_path: Path, sample_zip: bytes) -> None:
    """Uploading someone else's export leaves exactly the state a fresh install
    would have after uploading it: nothing from the previous account survives."""
    other_zip = build_zip(build_files(username="someoneelse", watched=WATCHED[:12]))

    (tmp_path / "fresh").mkdir()
    fresh_engine = make_engine(tmp_path / "fresh" / "db.sqlite3")
    def tmdb() -> FakeTmdb:
        fake = fake_for_sample()
        fake.movies[777] = movie(777, "Some Home Video", 2004)  # target of the manual fix below
        return fake

    with make_client(fresh_engine, tmdb()) as fresh:
        upload(fresh, other_zip)
        expected = _snapshot(fresh)
    app.dependency_overrides.clear()

    with make_client(engine, tmdb()) as c:
        upload(c, sample_zip)
        arrival = next(f for f in c.get("/api/films").json() if f["name"] == "Arrival")  # in both exports
        fixed = c.post("/api/matches/set", json={"film_key": arrival["film_key"], "tmdb_ref": "777"}).json()
        assert fixed["status"] == "manual"

        body = upload(c, other_zip)
        assert body["stats"]["reset"] is True
        assert _snapshot(c) == expected
    app.dependency_overrides.clear()


def test_every_eligible_film_is_scored(client: TestClient, engine: Engine, sample_zip: bytes) -> None:
    """No nearest-neighbour cutoff: every unseen candidate or watchlist film gets
    a score, however far it is from the taste vectors."""
    upload(client, sample_zip)
    with Session(engine) as s:
        eligible = current_film_ids(s) - {f.tmdb_id for f in s.exec(select(UserFilm)) if f.watched}
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
    assert client.get("/api/metrics").json()["ready"] is False
    upload(client, sample_zip)
    m = client.get("/api/metrics").json()
    assert m["ready"] and m["mode"] == "fixed" and m["n_ratings"] == 28  # < 50: fixed weights
    assert m["fixed_weights"] == {"profile": 0.3, "embedding": 0.4, "collab": 0.3}
    assert {"baseline_mean", "profile", "embedding", "fixed_blend", "learned_blend"} <= m["metrics"].keys()
    assert all(v["rmse"] > 0 for v in m["metrics"].values())
    assert m["partial"]["features"] == ["profile", "embedding", "votes"]
    assert m["collab_model"] is None and m["omdb"]["enabled"] is False
    assert m["candidates"]["total"] > 0 and sum(m["candidates"]["by_source"].values()) > 0
