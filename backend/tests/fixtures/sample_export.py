"""Builds a synthetic ~30-film Letterboxd export for tests and manual UI testing.

Deliberate edge cases:
  * titles with commas / accents / colons, a BOM on ratings.csv
  * a film in watched.csv with no year
  * a diary-only rating (Uncut Gems), a diary row with an invalid rating (Heat)
  * a rewatch with multiple diary entries (Arrival) + tags
  * an extra unexpected column in diary.csv
  * a film both watched and on the watchlist (Stalker)
  * decoy `deleted/` and `likes/` folders that must be ignored

Run as a script to write a zip:  python -m tests.fixtures.sample_export out.zip
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile

ROOT = "letterboxd-sampleuser-2026-09-01-00-00-utc"

# (name, year, rating or None, letterboxd id)
WATCHED: list[tuple[str, str, float | None, str]] = [
    ("Arrival", "2016", 5.0, "a01"),
    ("Blade Runner 2049", "2017", 4.5, "a02"),
    ("Sicario", "2015", 4.0, "a03"),
    ("Dune", "2021", 4.5, "a04"),
    ("Prisoners", "2013", 4.0, "a05"),
    ("Parasite", "2019", 5.0, "a06"),
    ("Crouching Tiger, Hidden Dragon", "2000", 4.0, "a07"),
    ("Amélie", "2001", 3.5, "a08"),
    ("The Grand Budapest Hotel", "2014", 4.0, "a09"),
    ("Mad Max: Fury Road", "2015", 4.5, "a10"),
    ("Transformers: Age of Extinction", "2014", 1.0, "a11"),
    ("Cats", "2019", 0.5, "a12"),
    ("The Room", "2003", 1.5, "a13"),
    ("Inception", "2010", 3.5, "a14"),
    ("Interstellar", "2014", 4.0, "a15"),
    ("Her", "2013", 4.5, "a16"),
    ("Lost in Translation", "2003", 4.0, "a17"),
    ("In the Mood for Love", "2000", 5.0, "a18"),
    ("Spirited Away", "2001", 4.5, "a19"),
    ("The Emoji Movie", "2017", 0.5, "a20"),
    ("Everything Everywhere All at Once", "2022", 4.5, "a21"),
    ("Moonlight", "2016", 4.0, "a22"),
    ("Tenet", "2020", 2.5, "a23"),
    ("The Good, the Bad and the Ugly", "1966", 4.0, "a24"),
    ("Portrait of a Lady on Fire", "2019", 4.5, "a25"),
    ("Stalker", "1979", 5.0, "a26"),
    ("Heat", "1995", 4.0, "a27"),
    ("Paddington 2", "2017", None, "a28"),
    ("Uncut Gems", "2019", None, "a29"),  # rated only in diary
    ("Home Movie Night", "", None, "a30"),  # no year
]

WATCHLIST: list[tuple[str, str, str]] = [
    ("Past Lives", "2023", "w01"),
    ("Oppenheimer", "2023", "w02"),
    ("Drive My Car", "2021", "w03"),
    ("Aftersun", "2022", "w04"),
    ("Decision to Leave", "2022", "w05"),
    ("Stalker", "1979", "a26"),
]

DIARY_HEADER = [
    "Date", "Name", "Year", "Letterboxd URI", "Rating", "Rewatch", "Tags",
    "Watched Date", "Extra Column",
]
DIARY: list[list[str]] = [
    ["2016-11-21", "Arrival", "2016", "https://boxd.it/e01", "4.5", "", "", "2016-11-20", "x"],
    ["2024-01-05", "Arrival", "2016", "https://boxd.it/e02", "5", "Yes", "sci-fi, cinema", "2024-01-04", ""],
    ["2021-10-23", "Dune", "2021", "https://boxd.it/e03", "4.5", "", "imax", "2021-10-22", ""],
    ["2019-11-02", "Parasite", "2019", "https://boxd.it/e04", "5", "", "", "2019-11-01", ""],
    ["2020-09-05", "Tenet", "2020", "https://boxd.it/e05", "2.5", "", "cinema", "2020-09-04", ""],
    ["2019-12-25", "Uncut Gems", "2019", "https://boxd.it/e06", "3", "", "", "2019-12-24", ""],
    ["2019-12-21", "Cats", "2019", "https://boxd.it/e07", "0.5", "", "hate-watch", "2019-12-20", ""],
    ["2022-06-01", "The Room", "2003", "https://boxd.it/e08", "1.5", "Yes", "hate-watch", "2022-05-31", ""],
    ["2023-02-11", "Heat", "1995", "https://boxd.it/e09", "abc", "", "", "2023-02-10", ""],
    ["2023-07-15", "Paddington 2", "2017", "https://boxd.it/e10", "", "", "", "2023-07-14", ""],
]

REVIEWS_HEADER = [
    "Date", "Name", "Year", "Letterboxd URI", "Rating", "Rewatch", "Review", "Tags", "Watched Date",
]
REVIEWS: list[list[str]] = [
    ["2016-11-21", "Arrival", "2016", "https://boxd.it/e01", "4.5", "",
     "Quietly devastating. The linguistics, the score, \"Hannah\" — all of it.", "", "2016-11-20"],
    ["2019-12-21", "Cats", "2019", "https://boxd.it/e07", "0.5", "",
     "Why.\nJust why.", "hate-watch", "2019-12-20"],
    ["2023-07-15", "Paddington 2", "2017", "https://boxd.it/e10", "", "",
     "If we're kind and polite, the world will be right.", "", "2023-07-14"],
]


def _csv(header: list[str], rows: list[list[str]], bom: bool = False) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return (("﻿" if bom else "") + buf.getvalue()).encode("utf-8")


def build_files() -> dict[str, bytes]:
    watched_rows = [["2024-01-01", n, y, f"https://boxd.it/{i}"] for n, y, _, i in WATCHED]
    rating_rows = [
        ["2024-01-01", n, y, f"https://boxd.it/{i}", f"{r:g}"]
        for n, y, r, i in WATCHED
        if r is not None
    ]
    watchlist_rows = [["2024-02-01", n, y, f"https://boxd.it/{i}"] for n, y, i in WATCHLIST]
    return {
        f"{ROOT}/watched.csv": _csv(["Date", "Name", "Year", "Letterboxd URI"], watched_rows),
        f"{ROOT}/ratings.csv": _csv(
            ["Date", "Name", "Year", "Letterboxd URI", "Rating"], rating_rows, bom=True
        ),
        f"{ROOT}/diary.csv": _csv(DIARY_HEADER, DIARY),
        f"{ROOT}/watchlist.csv": _csv(["Date", "Name", "Year", "Letterboxd URI"], watchlist_rows),
        f"{ROOT}/reviews.csv": _csv(REVIEWS_HEADER, REVIEWS),
        f"{ROOT}/profile.csv": _csv(["Username"], [["sampleuser"]]),
        # Decoys: same file names in folders that must be ignored.
        f"{ROOT}/deleted/ratings.csv": _csv(
            ["Date", "Name", "Year", "Letterboxd URI", "Rating"],
            [["2020-01-01", "Deleted Film", "1999", "https://boxd.it/zz", "5"]],
        ),
        f"{ROOT}/likes/films.csv": _csv(
            ["Date", "Name", "Year", "Letterboxd URI"],
            [["2020-01-01", "Liked Film", "1999", "https://boxd.it/zy"]],
        ),
    }


def build_zip(files: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, data in (files or build_files()).items():
            zf.writestr(path, data)
    return buf.getvalue()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "sample_letterboxd_export.zip"
    with open(out, "wb") as fh:
        fh.write(build_zip())
    print(f"wrote {out}")
