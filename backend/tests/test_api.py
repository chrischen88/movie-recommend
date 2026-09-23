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
    assert out["stages"] == ["parsing", "matching", "enrichment"]
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
