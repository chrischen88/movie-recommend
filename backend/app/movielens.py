"""MovieLens: download, load, and the movieId → tmdbId mapping.

The dataset is a static ZIP (≈1 MB for ml-latest-small, ≈240 MB for ml-32m), not
a JSON API, so it's streamed straight to disk with httpx rather than through
`CachedHttpClient` (which caches JSON bodies in SQLite). It's fetched once and
then read from `data/movielens/<name>/` (see PROGRESS.md, decision 12).
"""

from __future__ import annotations

import logging
import shutil
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://files.grouplens.org/datasets/movielens"
REQUIRED_FILES = ("ratings.csv", "links.csv")


class MovieLensError(RuntimeError):
    pass


def is_downloaded(dataset_dir: Path) -> bool:
    return all((dataset_dir / f).is_file() for f in REQUIRED_FILES)


def download_dataset(
    name: str,
    root: Path,
    *,
    transport: httpx.BaseTransport | None = None,
    attempts: int = 3,
    timeout: float = 60.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Fetch `<name>.zip` and extract just the files we need into `root/<name>/`.

    Writes to temporary paths and renames at the end, so an interrupted
    download never leaves a half-written dataset that looks complete.
    """
    root.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{name}.zip"
    part = root / f"{name}.zip.part"
    for attempt in range(1, attempts + 1):
        try:
            log.info("downloading MovieLens %s from %s", name, url)
            with httpx.Client(transport=transport, timeout=timeout, follow_redirects=True) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length") or 0)
                    done = 0
                    with part.open("wb") as f:
                        for chunk in resp.iter_bytes(1 << 20):
                            f.write(chunk)
                            done += len(chunk)
                            if on_progress:
                                on_progress(done, total)
            break
        except httpx.HTTPError as exc:
            part.unlink(missing_ok=True)
            if attempt == attempts:
                raise MovieLensError(f"could not download MovieLens {name}: {exc}") from exc
            log.warning("MovieLens download attempt %d failed (%s); retrying", attempt, exc)
            time.sleep(2**attempt)

    out, staging = root / name, root / f"{name}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        with zipfile.ZipFile(part) as zf:
            members = {Path(m).name: m for m in zf.namelist() if not m.endswith("/")}
            for fname in REQUIRED_FILES:
                if fname not in members:
                    raise MovieLensError(f"{name}.zip has no {fname}")
                with zf.open(members[fname]) as src, (staging / fname).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise MovieLensError(f"{name}.zip is not a valid ZIP: {exc}") from exc
    except MovieLensError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        part.unlink(missing_ok=True)
    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    log.info("MovieLens %s ready in %s", name, out)
    return out


@dataclass
class MovieLensData:
    user_idx: np.ndarray  # int32, one per rating
    item_idx: np.ndarray  # int32, one per rating
    rating: np.ndarray  # float32, 0.5–5.0
    movie_ids: np.ndarray  # item index → MovieLens movieId
    tmdb_ids: np.ndarray  # item index → tmdbId, or -1 when links.csv has none
    n_users: int

    @property
    def n_items(self) -> int:
        return len(self.movie_ids)


def load_dataset(dataset_dir: Path) -> MovieLensData:
    if not is_downloaded(dataset_dir):
        raise MovieLensError(f"MovieLens files missing in {dataset_dir}")
    ratings = pd.read_csv(
        dataset_dir / "ratings.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
    )
    # tmdbId is blank for some films, so it can't be read as an integer column.
    links = pd.read_csv(dataset_dir / "links.csv", usecols=["movieId", "tmdbId"], dtype={"movieId": "int32"})
    user_idx, users = pd.factorize(ratings["userId"])
    item_idx, movie_ids = pd.factorize(ratings["movieId"])
    tmdb = links.dropna(subset=["tmdbId"]).set_index("movieId")["tmdbId"].astype("int64")
    tmdb_ids = pd.Series(movie_ids).map(tmdb).fillna(-1).astype("int64").to_numpy()
    log.info(
        "MovieLens: %d ratings, %d users, %d films (%d with a TMDB id)",
        len(ratings), len(users), len(movie_ids), int((tmdb_ids >= 0).sum()),
    )
    return MovieLensData(
        user_idx=user_idx.astype(np.int32),
        item_idx=item_idx.astype(np.int32),
        rating=ratings["rating"].to_numpy(np.float32),
        movie_ids=np.asarray(movie_ids, dtype=np.int64),
        tmdb_ids=tmdb_ids,
        n_users=len(users),
    )
