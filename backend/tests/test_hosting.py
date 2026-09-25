"""What a hosted deployment relies on: the API serving the built UI and the
optional Basic-auth gate."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine

from app.config import get_settings
from app.main import app
from tests.test_api import make_client


@pytest.fixture
def client(engine: Engine) -> Iterator[TestClient]:
    with make_client(engine, None) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def dist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<div id=root></div>")
    (root / "assets" / "index-abc123.js").write_text("console.log(1)")
    (root / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("outside dist")
    monkeypatch.setattr(get_settings(), "frontend_dist", root)
    return root


def basic(user: str, pw: str) -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


# ---------------------------------------------------------------- frontend


def test_serves_index_for_client_routes(client: TestClient, dist: Path) -> None:
    for path in ("/", "/recommendations", "/matches"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert resp.text == "<div id=root></div>"
        assert resp.headers["cache-control"] == "no-cache"


def test_serves_static_files_and_caches_hashed_assets(client: TestClient, dist: Path) -> None:
    js = client.get("/assets/index-abc123.js")
    assert js.text == "console.log(1)"
    assert "immutable" in js.headers["cache-control"]
    assert client.get("/favicon.svg").text == "<svg/>"
    assert "immutable" not in client.get("/favicon.svg").headers.get("cache-control", "")


def test_unknown_api_paths_stay_404(client: TestClient, dist: Path) -> None:
    assert client.get("/api/nope").status_code == 404
    assert client.get("/api").status_code == 404
    assert client.get("/api/health").json()["status"] == "ok"


def test_does_not_serve_files_outside_dist(client: TestClient, dist: Path) -> None:
    resp = client.get("/%2E%2E/secret.txt")
    assert "outside dist" not in resp.text


def test_no_build_means_no_frontend(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "frontend_dist", tmp_path / "missing")
    assert client.get("/recommendations").status_code == 404


# ---------------------------------------------------------------- auth


def test_auth_off_without_password(client: TestClient) -> None:
    assert client.get("/api/matches").status_code == 410  # reached the app: no session yet


def test_auth_gates_everything_but_health(client: TestClient, dist: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "auth_password", "hunter2")

    for path in ("/api/matches", "/", "/assets/index-abc123.js"):
        resp = client.get(path)
        assert resp.status_code == 401, path
        assert resp.headers["www-authenticate"].startswith("Basic ")
    assert client.get("/api/health").status_code == 200

    assert client.get("/api/matches", headers=basic("letterboxd", "hunter2")).status_code == 410
    assert client.get("/", headers=basic("letterboxd", "hunter2")).status_code == 200
    for bad in (basic("letterboxd", "wrong"), basic("admin", "hunter2"),
                {"Authorization": "Basic not-base64!"}, {"Authorization": "Bearer hunter2"}):
        assert client.get("/api/matches", headers=bad).status_code == 401
