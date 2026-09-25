from __future__ import annotations

import numpy as np
import pytest

from app.embeddings import l2_normalize
from app.taste import (
    TASTE_SOURCE,
    TasteModel,
    build_clusters,
    build_taste_model,
    build_taste_vector,
    percentile_rank,
    score_embeddings,
)

RNG = np.random.default_rng(0)
DIM = 16


def axis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def near(i: int, noise: float = 0.05) -> np.ndarray:
    return l2_normalize(axis(i) + RNG.normal(0, noise, DIM).astype(np.float32))


def test_taste_vector_points_toward_liked_and_away_from_disliked() -> None:
    ratings = np.array([5.0, 4.5, 1.0, 0.5], dtype=np.float32)
    embs = np.stack([near(0), near(0), near(1), near(1)])
    taste, mu = build_taste_vector(ratings, embs)
    assert mu == pytest.approx(2.75)
    assert np.linalg.norm(taste) == pytest.approx(1.0, abs=1e-5)
    assert taste @ axis(0) > 0.5
    assert taste @ axis(1) < -0.5


def test_taste_vector_with_identical_ratings_falls_back_to_mean() -> None:
    embs = np.stack([axis(0), axis(1)])
    taste, _ = build_taste_vector(np.array([4.0, 4.0], dtype=np.float32), embs)
    assert taste @ axis(0) == pytest.approx(taste @ axis(1))
    assert taste @ axis(0) > 0


def three_blobs() -> tuple[list[int], np.ndarray, np.ndarray, dict[int, list[str]]]:
    ids, vecs, ratings, genres = [], [], [], {}
    blob_genres = {0: ["Science Fiction", "Drama"], 1: ["Romance", "Drama"], 2: ["Horror"]}
    for b in range(3):
        for j in range(6):
            tid = 100 * (b + 1) + j
            ids.append(tid)
            vecs.append(near(b * 4))
            ratings.append(5.0 - 0.1 * j)
            genres[tid] = blob_genres[b]
    return ids, np.stack(vecs), np.array(ratings, dtype=np.float32), genres


def test_clusters_pick_k_by_silhouette_and_label_by_genre() -> None:
    ids, embs, ratings, genres = three_blobs()
    clusters, sil = build_clusters(ids, embs, ratings, genres, (3, 6))
    assert len(clusters) == 3
    assert sil is not None and sil > 0.8
    labels = sorted(c.label for c in clusters)
    assert labels == ["Horror", "Romance · Drama", "Science Fiction · Drama"]
    for c in clusters:
        assert len(c.member_ids) == 6
        # members sorted by rating, highest first
        assert c.member_ids[0] % 100 == 0
        assert np.linalg.norm(c.centroid) == pytest.approx(1.0, abs=1e-5)


def test_too_few_liked_films_means_no_clusters() -> None:
    ids, embs, ratings, genres = three_blobs()
    clusters, sil = build_clusters(ids[:5], embs[:5], ratings[:5], genres, (3, 6))
    assert clusters == [] and sil is None


def test_build_model_skips_films_without_embeddings() -> None:
    ids, embs, _, genres = three_blobs()
    ratings = {i: 4.5 for i in ids} | {999: 1.0}
    model = build_taste_model(ratings, dict(zip(ids, embs)), genres)
    assert model is not None and model.n_rated == len(ids)
    assert build_taste_model({1: 4.0}, {}, {}) is None


def test_cluster_match_is_recorded() -> None:
    ids, embs, ratings, genres = three_blobs()
    model = build_taste_model(dict(zip(ids, ratings.tolist())), dict(zip(ids, embs)), genres)
    assert model is not None
    horror = next(c for c in model.clusters if c.label == "Horror")
    cands = {1: near(8), 2: near(0), 3: near(4), 4: near(15)}
    scores = score_embeddings(model, cands, mode="max")
    assert scores[1].source == horror.source
    assert model.label_for(scores[1].source) == "Horror"
    assert scores[4].raw < scores[1].raw  # unrelated direction scores lower
    assert sorted(s.normalized for s in scores.values()) == pytest.approx([0, 1 / 3, 2 / 3, 1])


def test_zmax_lets_taste_vector_compete() -> None:
    # One cluster centroid with high raw cosines everywhere, and a taste vector
    # with small raw cosines that still clearly prefers candidate 1.
    model = TasteModel(
        mean_rating=3.0,
        n_rated=2,
        taste_vector=axis(1),
    )
    from app.taste import TasteCluster

    centroid = l2_normalize(axis(0) * 3 + axis(1) * 0.0)
    model.clusters = [TasteCluster(id=0, centroid=centroid, member_ids=[], label="C")]
    cands = {
        1: l2_normalize(axis(0) * 3 + axis(1) * 1.0),  # the taste vector's standout
        2: l2_normalize(axis(0) * 3 + axis(2) * 0.2),
        3: l2_normalize(axis(0) * 3 + axis(3) * 0.4),
    }
    raw = score_embeddings(model, cands, mode="max")
    assert all(s.source == "cluster:0" for s in raw.values())
    z = score_embeddings(model, cands, mode="zmax")
    assert z[1].source == TASTE_SOURCE
    assert z[1].normalized == 1.0


def test_percentile_rank() -> None:
    np.testing.assert_allclose(percentile_rank(np.array([3.0, 1.0, 2.0])), [1.0, 0.0, 0.5])
    np.testing.assert_allclose(percentile_rank(np.array([1.0, 1.0, 2.0])), [0.25, 0.25, 1.0])
    np.testing.assert_allclose(percentile_rank(np.array([7.0])), [1.0])
    assert len(percentile_rank(np.array([]))) == 0


def test_min_cluster_size_rejects_singleton_clusters() -> None:
    ids, embs, ratings, genres = three_blobs()
    # One stray favourite far from every blob.
    ids = ids + [999]
    embs = np.vstack([embs, near(12)[None, :]])
    ratings = np.append(ratings, 5.0).astype(np.float32)
    clusters, _ = build_clusters(ids, embs, ratings, genres, (3, 6), min_size=3)
    assert all(len(c.member_ids) >= 3 for c in clusters)
    loose, _ = build_clusters(ids, embs, ratings, genres, (3, 6), min_size=1)
    assert any(c.member_ids == [999] for c in loose)
