from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.collab import CollabScorer, MFModel, TrainParams, load_scorer, train_als
from app.movielens import load_dataset
from tests.fake_movielens import THIN_MOVIE, write_fake_movielens

TMDB = list(range(100, 140))
EVEN = TMDB[0::2]  # "A" films
ODD = TMDB[1::2]  # "B" films
PARAMS = TrainParams(factors=4, iterations=8, reg=0.1, val_fraction=0.1, seed=0)


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> MFModel:
    d = write_fake_movielens(tmp_path_factory.mktemp("ml") / "fake", TMDB)
    return train_als(load_dataset(d), PARAMS, "fake")


def test_training_beats_global_mean(model: MFModel) -> None:
    assert model.meta["val_rmse"] < 0.8 * model.meta["val_rmse_global_mean"]
    assert model.meta["fingerprint"] == PARAMS.fingerprint("fake")
    assert model.item_factors.shape == (len(model.movie_ids), 4)


def test_fingerprint_changes_with_params() -> None:
    assert PARAMS.fingerprint("a") != PARAMS.fingerprint("b")
    assert PARAMS.fingerprint("a") != TrainParams(factors=8).fingerprint("a")


def test_fold_in_learns_the_users_taste(model: MFModel) -> None:
    scorer = CollabScorer(model, min_item_ratings=5)
    likes_even = {t: 4.5 for t in EVEN[:6]} | {t: 1.5 for t in ODD[:6]}
    user = scorer.fold_in(likes_even)
    assert user is not None and user.n_mapped == 12
    pred = scorer.predict(user, EVEN[6:] + ODD[6:])
    assert min(pred[t] for t in EVEN[6:]) > max(pred[t] for t in ODD[6:])
    assert all(0.5 <= p <= 5.0 for p in pred.values())

    flipped = scorer.fold_in({t: 6.0 - r for t, r in likes_even.items()})
    assert flipped is not None
    assert scorer.predict(flipped, [EVEN[7]])[EVEN[7]] < scorer.predict(flipped, [ODD[7]])[ODD[7]]


def test_fold_in_needs_enough_mapped_ratings(model: MFModel) -> None:
    scorer = CollabScorer(model)
    assert scorer.fold_in({t: 4.0 for t in EVEN[:4]}, min_ratings=5) is None
    assert scorer.fold_in({1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: 4.0}, min_ratings=5) is None  # unknown ids
    assert scorer.fold_in({t: 4.0 for t in EVEN[:5]}, min_ratings=5) is not None


def test_item_mapping(model: MFModel) -> None:
    scorer = CollabScorer(model, min_item_ratings=5)
    # tmdb 100 has two movieIds: the well-rated one wins over the 3-rating duplicate.
    idx = scorer.item_for_tmdb[100]
    assert model.movie_ids[idx] == 1
    assert 999999 not in scorer.item_for_tmdb  # THIN_MOVIE: one rating < min_item_ratings
    assert 999999 in CollabScorer(model, min_item_ratings=1).item_for_tmdb
    assert THIN_MOVIE in model.movie_ids.tolist()


def test_score_is_percentile_over_known_films(model: MFModel) -> None:
    scorer = CollabScorer(model)
    user = scorer.fold_in({t: 4.5 for t in EVEN[:6]} | {t: 1.5 for t in ODD[:6]})
    assert user is not None
    scores = scorer.score(user, [EVEN[10], ODD[10], 424242])
    assert set(scores) == {EVEN[10], ODD[10]}  # 424242 isn't in MovieLens
    assert scores[EVEN[10]].normalized == 1.0 and scores[ODD[10]].normalized == 0.0
    assert scores[EVEN[10]].predicted > scores[ODD[10]].predicted
    assert scorer.score(user, [424242]) == {}


def test_top_unseen_skips_seen_and_thin_films(model: MFModel) -> None:
    scorer = CollabScorer(model, min_item_ratings=1)
    user = scorer.fold_in({t: 4.5 for t in EVEN[:6]} | {t: 1.5 for t in ODD[:6]})
    assert user is not None
    top = scorer.top_unseen(user, set(EVEN[:6]), 5)
    assert len(top) == 5 and not set(EVEN[:6]) & {t for t, _ in top}
    assert [s for _, s in top] == sorted((s for _, s in top), reverse=True)
    everything = {t for t, _ in scorer.top_unseen(user, set(), 1000)}
    assert 999999 in everything  # THIN_MOVIE has a single rating...
    assert 999999 not in {t for t, _ in scorer.top_unseen(user, set(), 1000, min_ratings=2)}  # ...so it's dropped


def test_save_load_roundtrip(model: MFModel, tmp_path: Path) -> None:
    path = tmp_path / "sub" / "model.npz"
    model.save(path)
    loaded = MFModel.load(path)
    assert loaded is not None
    assert loaded.meta == model.meta and loaded.mu == pytest.approx(model.mu)
    np.testing.assert_array_equal(loaded.item_factors, model.item_factors)
    np.testing.assert_array_equal(loaded.tmdb_ids, model.tmdb_ids)
    assert MFModel.load(tmp_path / "nope.npz") is None
    (tmp_path / "bad.npz").write_bytes(b"not a zip")
    assert MFModel.load(tmp_path / "bad.npz") is None


def test_load_scorer_caches_until_file_changes(model: MFModel, tmp_path: Path) -> None:
    path = tmp_path / "model.npz"
    assert load_scorer(path, 5) is None
    model.save(path)
    first = load_scorer(path, 5)
    assert first is not None and load_scorer(path, 5) is first
    assert load_scorer(path, 1) is not first  # different threshold
    model.save(path)
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + 10))
    assert load_scorer(path, 5) is not first
