from __future__ import annotations

from pathlib import Path

import httpx
import numpy as np
import pytest

from app import movielens
from app.movielens import MovieLensError, download_dataset, is_downloaded, load_dataset
from tests.fake_movielens import DUP_MOVIE, NO_TMDB_MOVIE, fake_zip, write_fake_movielens

TMDB = list(range(100, 120))


def test_load_dataset(tmp_path: Path) -> None:
    d = write_fake_movielens(tmp_path / "ml", TMDB, n_users=20)
    assert is_downloaded(d) and not is_downloaded(tmp_path / "missing")
    data = load_dataset(d)
    assert data.n_users == 20
    assert len(data.user_idx) == len(data.item_idx) == len(data.rating)
    assert data.rating.dtype == np.float32 and data.rating.min() >= 0.5 and data.rating.max() <= 5.0
    by_movie = dict(zip(data.movie_ids.tolist(), data.tmdb_ids.tolist()))
    assert by_movie[1] == 100 and by_movie[20] == 119
    assert by_movie[NO_TMDB_MOVIE] == -1  # blank tmdbId
    assert by_movie[DUP_MOVIE] == 100  # shares a tmdb id; resolved by the scorer


def test_load_dataset_missing_files(tmp_path: Path) -> None:
    with pytest.raises(MovieLensError):
        load_dataset(tmp_path)


def serve(body: bytes, status: int = 200, calls: list[str] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(status, content=body)

    return httpx.MockTransport(handler)


def test_download_extracts_required_files(tmp_path: Path) -> None:
    calls: list[str] = []
    progress: list[tuple[int, int]] = []
    out = download_dataset(
        "ml-latest-small", tmp_path, transport=serve(fake_zip("ml-latest-small", TMDB), calls=calls),
        on_progress=lambda done, total: progress.append((done, total)),
    )
    assert calls == ["https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"]
    assert out == tmp_path / "ml-latest-small"
    assert sorted(p.name for p in out.iterdir()) == ["links.csv", "ratings.csv"]  # not tags/README
    assert load_dataset(out).n_users == 80
    assert progress and progress[-1][0] > 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ml-latest-small"]  # no .part / .tmp left


def test_download_rejects_bad_zip(tmp_path: Path) -> None:
    with pytest.raises(MovieLensError, match="not a valid ZIP"):
        download_dataset("ml-latest-small", tmp_path, transport=serve(b"<html>oops</html>"))
    assert list(tmp_path.iterdir()) == []


def test_download_rejects_zip_without_ratings(tmp_path: Path) -> None:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ml-latest-small/links.csv", "movieId,imdbId,tmdbId\n")
    with pytest.raises(MovieLensError, match="no ratings.csv"):
        download_dataset("ml-latest-small", tmp_path, transport=serve(buf.getvalue()))
    assert list(tmp_path.iterdir()) == []


def test_download_retries_then_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(movielens.time, "sleep", lambda _s: None)
    calls: list[str] = []
    with pytest.raises(MovieLensError, match="could not download"):
        download_dataset("ml-latest-small", tmp_path, transport=serve(b"", 503, calls), attempts=3)
    assert len(calls) == 3
    assert not is_downloaded(tmp_path / "ml-latest-small")
