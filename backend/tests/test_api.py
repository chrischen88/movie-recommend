from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlmodel import Session

from app.db import get_session
from app.main import app
from app.pipeline import get_pipeline
from tests.fake_tmdb import FakeTmdb, movie
from tests.test_pipeline import fake_for_sample, make_pipeline


def make_client(engine: Engine, fake: FakeTmdb | None) -> TestClient:
    def _session() -> Iterator[Session]:
        with Session(engine) as s:
            yield s

    pipeline = make_pipeline(engine, fake)
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
    assert out["stages"] == ["parsing", "matching", "enrichment", "candidates", "embedding", "taste"]
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
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores) and scores[0] == 1.0
    assert all(i["source_label"] for i in items)

    assert len(client.get("/api/recommendations", params={"limit": 3}).json()["items"]) == 3


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
