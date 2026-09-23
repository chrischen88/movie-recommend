from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.blend import BlendModel, LinearBlend, OOFRow, fit_blend, fixed_model, out_of_fold
from app.config import Settings
from app.db import Movie


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def rows(n: int, seed: int = 0, collab_every: int = 1) -> list[OOFRow]:
    """Ratings driven mostly by ③, a bit by ①, not at all by ②."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        p, e, c = rng.random(3)
        rating = float(np.clip(1.0 + 1.0 * p + 3.0 * c + rng.normal(0, 0.2), 0.5, 5.0))
        out.append(OOFRow(i, rating, float(p), float(e), float(c) if i % collab_every == 0 else None,
                          float(rng.normal(8, 1))))
    return out


def test_fixed_score_reweights_missing_scores() -> None:
    m = fixed_model(settings())
    assert m.fixed_weights == {"profile": 0.3, "embedding": 0.4, "collab": 0.3}
    assert m.fixed_score(1.0, 0.0, 1.0) == pytest.approx(0.6)
    assert m.fixed_score(1.0, 0.0, None) == pytest.approx(0.3 / 0.7)  # ③ missing: weight spread over ①②
    assert m.fixed_score(None, 0.5, None) == pytest.approx(0.5)
    assert m.predict(1.0, 1.0, 1.0, 100) is None  # fixed mode has no predicted rating


def test_learns_which_score_matters() -> None:
    m = fit_blend(rows(120), settings())
    assert m.mode == "learned" and m.n_ratings == 120 and m.n_with_collab == 120
    assert m.full is not None and m.full.features == ["profile", "embedding", "collab", "votes"]
    coef = dict(zip(m.full.features, m.full.coef))
    assert coef["collab"] > coef["profile"] > coef["embedding"] - 1e-9
    assert all(c >= 0 for c in m.full.coef)  # non-negative weights
    mt = m.metrics
    assert mt["learned_blend"]["rmse"] < mt["fixed_blend"]["rmse"] < mt["baseline_mean"]["rmse"]
    assert mt["collab"]["spearman"] > mt["embedding"]["spearman"]
    good = m.predict(0.9, 0.5, 0.95, 1000)
    bad = m.predict(0.1, 0.5, 0.05, 1000)
    assert good is not None and bad is not None and good > bad + 2


def test_partial_model_for_films_without_collab() -> None:
    m = fit_blend(rows(120, collab_every=2), settings())
    assert m.n_with_collab == 60 and m.full is not None and m.partial is not None
    assert "collab" not in m.partial.features
    assert m.predict(0.5, 0.5, None, 100) == pytest.approx(
        m.partial.predict({"profile": 0.5, "embedding": 0.5, "votes": m.vote_feature(100)})
    )


def test_few_ratings_use_fixed_weights() -> None:
    m = fit_blend(rows(30), settings())
    assert m.mode == "fixed" and "learned_blend" in m.metrics  # metrics still reported
    tiny = fit_blend(rows(5), settings())
    assert tiny.mode == "fixed" and tiny.metrics == {}


def test_without_vote_feature() -> None:
    m = fit_blend(rows(80), settings(blend_use_vote_count=False))
    assert m.full is not None and "votes" not in m.full.features


def test_save_load(tmp_path: Path) -> None:
    m = fit_blend(rows(80), settings())
    m.fingerprint = "abc"
    m.save(tmp_path / "b.json")
    loaded = BlendModel.load(tmp_path / "b.json")
    assert loaded is not None and loaded.fingerprint == "abc"
    assert isinstance(loaded.full, LinearBlend)
    assert loaded.predict(0.3, 0.4, 0.5, 10) == pytest.approx(m.predict(0.3, 0.4, 0.5, 10))
    assert BlendModel.load(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert BlendModel.load(tmp_path / "bad.json") is None


# ---------------------------------------------------------------- out-of-fold


def library(n: int = 40, dim: int = 16) -> tuple[dict[int, Movie], dict[int, np.ndarray]]:
    rng = np.random.default_rng(1)
    movies, emb = {}, {}
    for t in range(1, n + 1):
        movies[t] = Movie(tmdb_id=t, title=f"F{t}", genres=["Drama" if t % 2 else "Comedy"],
                          directors=[f"Director {t}"], vote_count=100 * t)
        v = rng.normal(size=dim)
        emb[t] = (v / np.linalg.norm(v)).astype(np.float32)
    return movies, emb


def test_out_of_fold_has_no_leakage() -> None:
    movies, emb = library()
    rated = {t: (4.5 if t % 2 else 2.0) for t in range(1, 31)}  # Drama liked, Comedy not
    pool = list(range(31, 41))
    cfg = settings(taste_cluster_k_range=(2, 3))
    base = {r.tmdb_id: r for r in out_of_fold(rated, movies, emb, pool, None, cfg)}
    assert set(base) == set(rated)
    assert all(r.collab is None and 0 <= r.profile <= 1 and 0 <= r.embedding <= 1 for r in base.values())
    # Genre drives ①: held-out Drama films score above held-out Comedies.
    drama = np.mean([r.profile for t, r in base.items() if t % 2])
    comedy = np.mean([r.profile for t, r in base.items() if not t % 2])
    assert drama > comedy

    # A film's own rating must not influence its own out-of-fold scores.
    flipped = dict(rated)
    flipped[7] = 0.5
    again = {r.tmdb_id: r for r in out_of_fold(flipped, movies, emb, pool, None, cfg)}
    assert again[7].profile == pytest.approx(base[7].profile)
    assert again[7].embedding == pytest.approx(base[7].embedding)
    assert again[7].rating == 0.5


def test_out_of_fold_needs_enough_films() -> None:
    movies, emb = library(4)
    assert out_of_fold({1: 4.0, 2: 3.0}, movies, emb, [3, 4], None, settings()) == []
