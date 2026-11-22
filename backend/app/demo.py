"""Built-in sample profiles, so visitors without a Letterboxd export can try the app.

Each is a made-up viewer with a clear taste, picked from a dropdown:
- arthouse: slow-burn international and arthouse films and cerebral sci-fi rated
  high, franchise blockbusters low;
- family: Pixar, Ghibli, Aardman and Laika rated high, grim adult films and weak
  cash-in animation low;
- action: John Wick-style choreographed action rated high, romance and slow
  dramas low.

Each has more than `blend_min_ratings_for_learning` ratings, so the learned blend
and the metrics page have something to show. Titles are unambiguous (well-known,
with their release year) so they all match on TMDB without review.

A profile is built as a real export ZIP and goes through the normal upload path.
Once one run has brought its films into the shared film cache, later sessions on
it cost no API calls or embedding.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class DemoProfile:
    id: str
    label: str
    rated: list[tuple[str, int, float]]  # (name, year, rating)
    watchlist: list[tuple[str, int]]

    @property
    def username(self) -> str:
        return f"sample-{self.id}"


ARTHOUSE_RATED: list[tuple[str, int, float]] = [
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

ARTHOUSE_WATCHLIST: list[tuple[str, int]] = [
    ("Days of Heaven", 1978),
    ("Tampopo", 1985),
    ("The Tree of Life", 2011),
    ("Evil Does Not Exist", 2023),
    ("Fallen Leaves", 2023),
    ("All of Us Strangers", 2023),
]

FAMILY_RATED: list[tuple[str, int, float]] = [
    ("Toy Story", 1995, 5.0),
    ("Spirited Away", 2001, 5.0),
    ("Up", 2009, 5.0),
    ("Coco", 2017, 5.0),
    ("WALL·E", 2008, 5.0),
    ("Spider-Man: Into the Spider-Verse", 2018, 5.0),
    ("Inside Out", 2015, 5.0),
    ("How to Train Your Dragon", 2010, 5.0),
    ("Ratatouille", 2007, 4.5),
    ("Finding Nemo", 2003, 4.5),
    ("Toy Story 3", 2010, 4.5),
    ("The Incredibles", 2004, 4.5),
    ("Monsters, Inc.", 2001, 4.5),
    ("My Neighbor Totoro", 1988, 4.5),
    ("Kiki's Delivery Service", 1989, 4.5),
    ("Zootopia", 2016, 4.5),
    ("The Lion King", 1994, 4.5),
    ("Paddington 2", 2017, 4.5),
    ("Wallace & Gromit: The Curse of the Were-Rabbit", 2005, 4.5),
    ("Kubo and the Two Strings", 2016, 4.5),
    ("The Iron Giant", 1999, 4.5),
    ("Spider-Man: Across the Spider-Verse", 2023, 4.5),
    ("Klaus", 2019, 4.5),
    ("The Mitchells vs. the Machines", 2021, 4.5),
    ("Puss in Boots: The Last Wish", 2022, 4.5),
    ("Wolfwalkers", 2020, 4.5),
    ("Shrek", 2001, 4.5),
    ("Moana", 2016, 4.0),
    ("Encanto", 2021, 4.0),
    ("Frozen", 2013, 4.0),
    ("Tangled", 2010, 4.0),
    ("Big Hero 6", 2014, 4.0),
    ("Kung Fu Panda", 2008, 4.0),
    ("Fantastic Mr. Fox", 2009, 4.0),
    ("Coraline", 2009, 4.0),
    ("Ponyo", 2008, 4.0),
    ("Paddington", 2014, 4.0),
    ("Chicken Run", 2000, 4.0),
    ("The Lego Movie", 2014, 4.0),
    ("Beauty and the Beast", 1991, 4.0),
    ("Aladdin", 1992, 4.0),
    ("Howl's Moving Castle", 2004, 4.0),
    ("Song of the Sea", 2014, 4.0),
    ("Luca", 2021, 4.0),
    ("Soul", 2020, 4.0),
    ("Turning Red", 2022, 4.0),
    ("The Wild Robot", 2024, 4.0),
    ("Mary Poppins", 1964, 4.0),
    ("E.T. the Extra-Terrestrial", 1982, 4.0),
    ("Despicable Me", 2010, 3.5),
    ("Finding Dory", 2016, 3.5),
    ("Cars", 2006, 3.5),
    ("Ice Age", 2002, 3.5),
    ("Sing", 2016, 3.5),
    ("Elemental", 2023, 3.5),
    ("Minions: The Rise of Gru", 2022, 2.5),
    ("The Wolf of Wall Street", 2013, 2.0),
    ("Cars 2", 2011, 2.0),
    ("Uncut Gems", 2019, 1.5),
    ("Joker", 2019, 1.5),
    ("Space Jam: A New Legacy", 2021, 1.5),
    ("Midsommar", 2019, 1.0),
    ("Hereditary", 2018, 1.0),
    ("Requiem for a Dream", 2000, 1.0),
    ("The Emoji Movie", 2017, 1.0),
    ("Saw", 2004, 0.5),
]

FAMILY_WATCHLIST: list[tuple[str, int]] = [
    ("The Boy and the Heron", 2023),
    ("Flow", 2024),
    ("Inside Out 2", 2024),
    ("The Tale of the Princess Kaguya", 2013),
    ("Robot Dreams", 2023),
    ("My Life as a Zucchini", 2016),
]

ACTION_RATED: list[tuple[str, int, float]] = [
    ("John Wick", 2014, 5.0),
    ("John Wick: Chapter 2", 2017, 5.0),
    ("John Wick: Chapter 3 - Parabellum", 2019, 5.0),
    ("The Raid", 2011, 5.0),
    ("The Raid 2", 2014, 5.0),
    ("Mad Max: Fury Road", 2015, 5.0),
    ("Mission: Impossible - Fallout", 2018, 5.0),
    ("John Wick: Chapter 4", 2023, 4.5),
    ("Nobody", 2021, 4.5),
    ("Atomic Blonde", 2017, 4.5),
    ("Die Hard", 1988, 4.5),
    ("Heat", 1995, 4.5),
    ("The Villainess", 2017, 4.5),
    ("Kill Bill: Vol. 1", 2003, 4.5),
    ("Oldboy", 2003, 4.5),
    ("Extraction", 2020, 4.5),
    ("Monkey Man", 2024, 4.5),
    ("Upgrade", 2018, 4.5),
    ("The Night Comes for Us", 2018, 4.5),
    ("Hard Boiled", 1992, 4.5),
    ("The Man from Nowhere", 2010, 4.5),
    ("Terminator 2: Judgment Day", 1991, 4.5),
    ("Dredd", 2012, 4.5),
    ("The Bourne Ultimatum", 2007, 4.5),
    ("Top Gun: Maverick", 2022, 4.5),
    ("Mission: Impossible - Ghost Protocol", 2011, 4.0),
    ("Mission: Impossible - Dead Reckoning Part One", 2023, 4.0),
    ("Kill Bill: Vol. 2", 2004, 4.0),
    ("The Equalizer", 2014, 4.0),
    ("Taken", 2008, 4.0),
    ("Collateral", 2004, 4.0),
    ("Sicario", 2015, 4.0),
    ("Baby Driver", 2017, 4.0),
    ("Casino Royale", 2006, 4.0),
    ("Skyfall", 2012, 4.0),
    ("The Dark Knight", 2008, 4.0),
    ("Ip Man", 2008, 4.0),
    ("Crouching Tiger, Hidden Dragon", 2000, 4.0),
    ("Equilibrium", 2002, 4.0),
    ("The Matrix", 1999, 4.0),
    ("Hardcore Henry", 2015, 4.0),
    ("Bullet Train", 2022, 4.0),
    ("Edge of Tomorrow", 2014, 4.0),
    ("Kingsman: The Secret Service", 2014, 4.0),
    ("Face/Off", 1997, 4.0),
    ("Speed", 1994, 4.0),
    ("Point Break", 1991, 3.5),
    ("The Accountant", 2016, 3.5),
    ("Jack Reacher", 2012, 3.5),
    ("Fast Five", 2011, 3.5),
    ("The Gray Man", 2022, 3.0),
    ("La La Land", 2016, 2.5),
    ("Fast X", 2023, 2.5),
    ("Red Notice", 2021, 2.0),
    ("Gemini Man", 2019, 2.0),
    ("Lost in Translation", 2003, 2.0),
    ("Love Actually", 2003, 2.0),
    ("Pride & Prejudice", 2005, 2.0),
    ("The Tree of Life", 2011, 1.5),
    ("The Notebook", 2004, 1.5),
    ("Mamma Mia!", 2008, 1.0),
    ("Twilight", 2008, 1.0),
    ("Fifty Shades of Grey", 2015, 0.5),
]

ACTION_WATCHLIST: list[tuple[str, int]] = [
    ("The Killer", 1989),
    ("Furiosa: A Mad Max Saga", 2024),
    ("Headshot", 2016),
    ("Sisu", 2022),
    ("The Shadow Strays", 2024),
    ("Ballerina", 2025),
]

PROFILES: dict[str, DemoProfile] = {
    p.id: p
    for p in (
        DemoProfile("arthouse", "Arthouse & world cinema", ARTHOUSE_RATED, ARTHOUSE_WATCHLIST),
        DemoProfile("family", "Family animation", FAMILY_RATED, FAMILY_WATCHLIST),
        DemoProfile("action", "Action, John Wick-style", ACTION_RATED, ACTION_WATCHLIST),
    )
}
DEFAULT_PROFILE = "arthouse"


def _csv(header: list[str], rows: list[list[str]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


@lru_cache(maxsize=len(PROFILES))
def demo_export_zip(profile_id: str = DEFAULT_PROFILE) -> bytes:
    """A sample profile as a Letterboxd export ZIP. Raises KeyError for an unknown id."""
    profile = PROFILES[profile_id]
    root = f"letterboxd-{profile.username}-2026-09-01-00-00-utc"
    header = ["Date", "Name", "Year", "Letterboxd URI"]
    uri = lambda prefix, i: f"https://boxd.it/{prefix}{i:03d}"  # noqa: E731
    watched = [["2024-01-01", n, str(y), uri("d", i)] for i, (n, y, _) in enumerate(profile.rated)]
    ratings = [row + [f"{r:g}"] for row, (_, _, r) in zip(watched, profile.rated)]
    watchlist = [["2024-02-01", n, str(y), uri("w", i)] for i, (n, y) in enumerate(profile.watchlist)]
    files = {
        "watched.csv": _csv(header, watched),
        "ratings.csv": _csv(header + ["Rating"], ratings),
        "watchlist.csv": _csv(header, watchlist),
        "profile.csv": _csv(["Date Joined", "Username"], [["2020-01-01", profile.username]]),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(f"{root}/{name}", data)
    return buf.getvalue()
