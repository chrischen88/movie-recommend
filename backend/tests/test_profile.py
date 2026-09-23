from __future__ import annotations

import math

import pytest
from sqlmodel import Session

from app.db import Movie, UserFilm, make_engine
from app.profile import build_profile, film_features, load_profile, score_film, score_profile, user_ratings

WEIGHTS = {"director": 1.0, "genre": 0.6, "actor": 0.5, "keyword": 0.4, "decade": 0.3, "language": 0.3, "country": 0.2}


def mv(tmdb_id: int, **kw: object) -> Movie:
    base: dict[str, object] = dict(
        tmdb_id=tmdb_id, title=f"Film {tmdb_id}", year=None, original_language=None,
        genres=[], directors=[], cast=[], keywords=[], countries=[],
    )
    base.update(kw)
    return Movie(**base)  # type: ignore[arg-type]


def test_film_features() -> None:
    m = mv(
        1, year=1997, original_language="en", genres=["Drama", "Drama"], directors=["A"],
        cast=[f"C{i}" for i in range(8)], keywords=["space"], countries=["US", ""],
    )
    feats = film_features(m)
    assert ("genre", "Drama") in feats and len([f for f in feats if f[0] == "genre"]) == 1
    assert [v for t, v in feats if t == "actor"] == ["C0", "C1", "C2", "C3", "C4"]  # top 5 cast
    assert ("decade", "1990s") in feats and ("language", "en") in feats
    assert [v for t, v in feats if t == "country"] == ["US"]
    assert film_features(mv(2)) == []


def test_build_profile_shrinkage() -> None:
    movies = {
        1: mv(1, directors=["Villeneuve"], genres=["Sci-Fi"]),
        2: mv(2, directors=["Villeneuve"], genres=["Sci-Fi"]),
        3: mv(3, directors=["Bay"], genres=["Action"]),
        4: mv(4, directors=["Once"], genres=["Drama"]),
    }
    ratings = {1: 5.0, 2: 4.5, 3: 1.5, 4: 5.0, 99: 3.0}  # 99 has no metadata: ignored
    p = build_profile(ratings, movies, shrinkage_k=3.0)
    assert p is not None
    assert p.n_rated == 4 and p.mean_rating == pytest.approx(4.0)
    vil = p.stats[("director", "Villeneuve")]
    assert vil.n == 2 and vil.value == pytest.approx((1.0 + 0.5) / (2 + 3))
    assert p.stats[("director", "Bay")].value == pytest.approx(-2.5 / 4)
    # One +1★ film is shrunk harder than two films averaging +0.75★.
    once = p.stats[("director", "Once")].value
    assert once == pytest.approx(0.25) and once < vil.value
    assert [v for v, _ in p.top("director", min_films=1)] == ["Villeneuve", "Once", "Bay"]
    assert [v for v, _ in p.top("director", min_films=1, reverse=True)][0] == "Bay"
    assert [v for v, _ in p.top("director")] == ["Villeneuve"]  # min_films=2 by default


def test_one_film_cannot_dominate() -> None:
    # One 5★ film vs. a director with five films at +1★ each.
    movies = {1: mv(1, directors=["One Hit"])} | {i: mv(i, directors=["Reliable"]) for i in range(2, 7)}
    movies |= {i: mv(i, directors=[f"Filler {i}"]) for i in range(7, 15)}
    ratings = {1: 5.0} | {i: 4.0 for i in range(2, 7)} | {i: 2.5 for i in range(7, 15)}
    p = build_profile(ratings, movies, shrinkage_k=3.0)
    assert p is not None
    assert p.stats[("director", "Reliable")].value > p.stats[("director", "One Hit")].value


def test_no_ratings() -> None:
    assert build_profile({}, {}, 3.0) is None
    assert build_profile({1: 4.0}, {}, 3.0) is None


def test_score_film_contributions() -> None:
    movies = {
        1: mv(1, directors=["Villeneuve"], genres=["Sci-Fi"], keywords=["desert"]),
        2: mv(2, directors=["Villeneuve"], genres=["Sci-Fi"]),
        3: mv(3, directors=["Bay"], genres=["Action"], keywords=["explosion"]),
    }
    p = build_profile({1: 5.0, 2: 5.0, 3: 2.0}, movies, shrinkage_k=1.0)
    assert p is not None
    cand = mv(10, directors=["Villeneuve"], genres=["Sci-Fi", "Action"], keywords=["unknown"])
    s = score_film(p, cand, WEIGHTS)
    assert [c.label for c in s.contributions] == ["Director Villeneuve", "Genre Action", "Genre Sci-Fi"]
    top = s.contributions[0]
    assert top.stars == pytest.approx(2.0 / 3) and top.n == 2 and top.contribution == pytest.approx(2.0 / 3)
    # Two genres share the genre weight: each is scaled by 0.6 / √2.
    assert s.contributions[2].contribution == pytest.approx(0.6 / math.sqrt(2) * (2.0 / 3))
    assert s.contributions[1].contribution == pytest.approx(0.6 / math.sqrt(2) * -1.0)  # disliked genre
    assert s.raw == pytest.approx(sum(c.contribution for c in s.contributions))


def test_many_keywords_dont_swamp() -> None:
    kws = [f"k{i}" for i in range(16)]
    movies = {1: mv(1, keywords=kws, directors=["X"]), 2: mv(2, keywords=kws, directors=["X"]), 3: mv(3)}
    p = build_profile({1: 5.0, 2: 5.0, 3: 2.0}, movies, shrinkage_k=1.0)
    assert p is not None
    s = score_film(p, mv(10, keywords=kws), WEIGHTS)
    # 16 identical keyword values count as √16 = 4, not 16.
    assert s.raw == pytest.approx(0.4 * 16 / 4 * p.stats[("keyword", "k0")].value)


def test_score_profile_ranks_and_truncates() -> None:
    movies = {
        1: mv(1, directors=["Good"], genres=["Drama"], keywords=["a", "b", "c"]),
        2: mv(2, directors=["Good"], genres=["Drama"], keywords=["a", "b", "c"]),
        3: mv(3, directors=["Bad"], genres=["Horror"]),
    }
    p = build_profile({1: 5.0, 2: 4.5, 3: 1.0}, movies, shrinkage_k=3.0)
    assert p is not None
    cands = [
        mv(10, directors=["Good"], genres=["Drama"], keywords=["a", "b", "c"]),
        mv(11, directors=["Bad"], genres=["Horror"]),
        mv(12, directors=["Nobody"]),
    ]
    scores = score_profile(p, cands, WEIGHTS, top_contributions=2)
    assert scores[10].normalized == 1.0 and scores[11].normalized == 0.0
    assert scores[12].raw == 0.0 and scores[12].normalized == 0.5 and scores[12].contributions == []
    assert len(scores[10].contributions) == 2
    assert scores[10].contributions[0].label == "Director Good"
    assert scores[11].contributions[0].label == "Director Bad" and scores[11].contributions[0].stars < 0


def test_weak_reasons_are_hidden_but_still_scored() -> None:
    movies = {1: mv(1, directors=["A"], genres=["Drama"]), 2: mv(2, directors=["B"], genres=["Drama"])}
    movies |= {3: mv(3, directors=["C"], genres=["Comedy"]), 4: mv(4, directors=["D"], genres=["Horror"])}
    p = build_profile({1: 5.0, 2: 1.0, 3: 3.1, 4: 2.9}, movies, shrinkage_k=3.0)  # μ = 3.0
    assert p is not None
    assert p.stats[("genre", "Drama")].value == 0.0  # +2 and −2 cancel exactly
    comedy = p.stats[("genre", "Comedy")].value
    assert 0 < abs(comedy) < 0.05
    s = score_profile(p, [mv(10, directors=["A"], genres=["Comedy"])], WEIGHTS)[10]
    assert [c.label for c in s.contributions] == ["Director A"]
    assert s.raw == pytest.approx(1.0 * p.stats[("director", "A")].value + 0.6 * comedy)


def test_zero_weight_type_is_ignored() -> None:
    movies = {1: mv(1, genres=["Drama"]), 2: mv(2, genres=["Horror"])}
    p = build_profile({1: 5.0, 2: 1.0}, movies, shrinkage_k=1.0)
    assert p is not None
    s = score_film(p, mv(10, genres=["Drama"]), {**WEIGHTS, "genre": 0.0})
    assert s.raw == 0.0 and s.contributions == []


def test_user_ratings_and_load_profile() -> None:
    engine = make_engine(":memory:")
    with Session(engine) as s:
        s.add_all([
            UserFilm(film_key="a|2000", name="A", rating=4.0, tmdb_id=1, content_hash="x"),
            UserFilm(film_key="a2|2000", name="A2", rating=5.0, tmdb_id=1, content_hash="x"),  # same film
            UserFilm(film_key="b|2000", name="B", rating=2.0, tmdb_id=2, content_hash="x"),
            UserFilm(film_key="c|2000", name="C", rating=None, tmdb_id=3, content_hash="x"),  # unrated
            UserFilm(film_key="d|2000", name="D", rating=3.0, tmdb_id=None, content_hash="x"),  # unmatched
            mv(1, directors=["X"]), mv(2, directors=["Y"]), mv(3, directors=["X"]),
        ])
        s.commit()
        assert user_ratings(s) == {1: 4.5, 2: 2.0}
        p = load_profile(s, shrinkage_k=1.0)
        assert p is not None and p.n_rated == 2 and p.mean_rating == pytest.approx(3.25)
        assert p.stats[("director", "X")].n == 1
