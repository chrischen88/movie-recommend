"""A tiny synthetic MovieLens with real structure, so ALS has something to learn.

Two taste groups: "A" users love even-positioned films and dislike odd ones,
"B" users the reverse. Extra rows cover the awkward cases in real links.csv:
a film with no tmdbId, two movieIds sharing one tmdbId, and a thinly rated film.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

NO_TMDB_MOVIE = 9001  # movieId with a blank tmdbId
DUP_MOVIE = 9002  # second movieId for the first tmdb id, with fewer ratings
THIN_MOVIE = 9003  # rated by a single user


def fake_ratings(tmdb_ids: list[int], n_users: int = 80, seed: int = 0) -> tuple[str, str]:
    """Returns (ratings.csv, links.csv) contents. movieId = position + 1."""
    rng = np.random.default_rng(seed)
    rows = ["userId,movieId,rating,timestamp"]
    for u in range(1, n_users + 1):
        likes_even = u % 2 == 0
        for pos in range(len(tmdb_ids)):
            if rng.random() > 0.6:
                continue
            liked = (pos % 2 == 0) == likes_even
            r = rng.choice([4.0, 4.5, 5.0]) if liked else rng.choice([1.0, 1.5, 2.0])
            rows.append(f"{u},{pos + 1},{r},0")
        if u <= 10:
            rows.append(f"{u},{NO_TMDB_MOVIE},3.0,0")
        if u <= 3:
            rows.append(f"{u},{DUP_MOVIE},1.0,0")
    rows.append(f"1,{THIN_MOVIE},5.0,0")
    links = ["movieId,imdbId,tmdbId"]
    links += [f"{pos + 1},{pos + 1:07d},{tid}" for pos, tid in enumerate(tmdb_ids)]
    links += [f"{NO_TMDB_MOVIE},0009001,", f"{DUP_MOVIE},0009002,{tmdb_ids[0]}", f"{THIN_MOVIE},0009003,999999"]
    return "\n".join(rows) + "\n", "\n".join(links) + "\n"


def write_fake_movielens(dataset_dir: Path, tmdb_ids: list[int], **kw: int) -> Path:
    ratings, links = fake_ratings(tmdb_ids, **kw)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "ratings.csv").write_text(ratings)
    (dataset_dir / "links.csv").write_text(links)
    return dataset_dir


def fake_zip(name: str, tmdb_ids: list[int]) -> bytes:
    """The download's layout: everything under a `<name>/` folder, plus files we ignore."""
    ratings, links = fake_ratings(tmdb_ids)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{name}/ratings.csv", ratings)
        zf.writestr(f"{name}/links.csv", links)
        zf.writestr(f"{name}/tags.csv", "userId,movieId,tag,timestamp\n")
        zf.writestr(f"{name}/README.txt", "fake")
    return buf.getvalue()
