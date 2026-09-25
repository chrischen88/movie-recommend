"""A built-in sample profile, so visitors without a Letterboxd export can try the app.

A made-up viewer with a clear taste: slow-burn international and arthouse films
and cerebral sci-fi rated high, franchise blockbusters rated low. It has more
than `blend_min_ratings_for_learning` ratings, so the learned blend and the
metrics page have something to show. Titles are unambiguous (well-known, with
their release year) so they all match on TMDB without review.

It's built as a real export ZIP and goes through the normal upload path. Once
one demo run has brought its films into the shared film cache, later demo
sessions cost no API calls or embedding.
"""

from __future__ import annotations

import csv
import io
import zipfile
from functools import lru_cache

USERNAME = "sample-profile"
ROOT = f"letterboxd-{USERNAME}-2026-09-01-00-00-utc"

# (name, year, rating)
RATED: list[tuple[str, int, float]] = [
    ("In the Mood for Love", 2000, 5.0),
    ("Parasite", 2019, 5.0),
    ("Arrival", 2016, 5.0),
    ("Portrait of a Lady on Fire", 2019, 5.0),
    ("Spirited Away", 2001, 5.0),
    ("Stalker", 1979, 5.0),
    ("Yi Yi", 2000, 5.0),
    ("Blade Runner 2049", 2017, 4.5),
    ("Moonlight", 2016, 4.5),
    ("Her", 2013, 4.5),
    ("Past Lives", 2023, 4.5),
    ("Aftersun", 2022, 4.5),
    ("Drive My Car", 2021, 4.5),
    ("Burning", 2018, 4.5),
    ("Memories of Murder", 2003, 4.5),
    ("The Handmaiden", 2016, 4.5),
    ("Decision to Leave", 2022, 4.5),
    ("Paterson", 2016, 4.5),
    ("Phantom Thread", 2017, 4.5),
    ("There Will Be Blood", 2007, 4.5),
    ("No Country for Old Men", 2007, 4.5),
    ("Children of Men", 2006, 4.5),
    ("Eternal Sunshine of the Spotless Mind", 2004, 4.5),
    ("Mulholland Drive", 2001, 4.5),
    ("Shoplifters", 2018, 4.5),
    ("Tokyo Story", 1953, 4.5),
    ("Chungking Express", 1994, 4.5),
    ("Princess Mononoke", 1997, 4.5),
    ("Perfect Days", 2023, 4.5),
    ("Oldboy", 2003, 4.0),
    ("Columbus", 2017, 4.0),
    ("Lost in Translation", 2003, 4.0),
    ("Sicario", 2015, 4.0),
    ("Prisoners", 2013, 4.0),
    ("Ex Machina", 2014, 4.0),
    ("Annihilation", 2018, 4.0),
    ("Under the Skin", 2013, 4.0),
    ("Roma", 2018, 4.0),
    ("Seven Samurai", 1954, 4.0),
    ("My Neighbor Totoro", 1988, 4.0),
    ("The Zone of Interest", 2023, 4.0),
    ("Anatomy of a Fall", 2023, 4.0),
    ("The Worst Person in the World", 2021, 4.0),
    ("Minari", 2020, 4.0),
    ("Everything Everywhere All at Once", 2022, 4.0),
    ("Dune", 2021, 4.0),
    ("Mad Max: Fury Road", 2015, 4.0),
    ("The Grand Budapest Hotel", 2014, 3.5),
    ("Amélie", 2001, 3.5),
    ("Inception", 2010, 3.5),
    ("Interstellar", 2014, 3.5),
    ("La La Land", 2016, 3.5),
    ("Whiplash", 2014, 3.5),
    ("The Irishman", 2019, 3.5),
    ("Joker", 2019, 2.5),
    ("Tenet", 2020, 2.5),
    ("Don't Look Up", 2021, 2.5),
    ("Avatar: The Way of Water", 2022, 2.5),
    ("Bohemian Rhapsody", 2018, 2.0),
    ("Jurassic World", 2015, 2.0),
    ("Venom", 2018, 1.5),
    ("Black Adam", 2022, 1.5),
    ("Justice League", 2017, 1.5),
    ("Transformers: Age of Extinction", 2014, 1.0),
    ("Fast X", 2023, 1.0),
    ("Morbius", 2022, 1.0),
    ("The Emoji Movie", 2017, 0.5),
    ("Cats", 2019, 0.5),
]

WATCHLIST: list[tuple[str, int]] = [
    ("Days of Heaven", 1978),
    ("Tampopo", 1985),
    ("The Tree of Life", 2011),
    ("Evil Does Not Exist", 2023),
    ("Fallen Leaves", 2023),
    ("All of Us Strangers", 2023),
]


def _csv(header: list[str], rows: list[list[str]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


@lru_cache(maxsize=1)
def demo_export_zip() -> bytes:
    """The sample profile as a Letterboxd export ZIP."""
    header = ["Date", "Name", "Year", "Letterboxd URI"]
    uri = lambda prefix, i: f"https://boxd.it/{prefix}{i:03d}"  # noqa: E731
    watched = [["2024-01-01", n, str(y), uri("d", i)] for i, (n, y, _) in enumerate(RATED)]
    ratings = [row + [f"{r:g}"] for row, (_, _, r) in zip(watched, RATED)]
    watchlist = [["2024-02-01", n, str(y), uri("w", i)] for i, (n, y) in enumerate(WATCHLIST)]
    files = {
        "watched.csv": _csv(header, watched),
        "ratings.csv": _csv(header + ["Rating"], ratings),
        "watchlist.csv": _csv(header, watchlist),
        "profile.csv": _csv(["Date Joined", "Username"], [["2020-01-01", USERNAME]]),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(f"{ROOT}/{name}", data)
    return buf.getvalue()
