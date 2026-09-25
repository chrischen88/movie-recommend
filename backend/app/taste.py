"""Taste vector + taste clusters over film embeddings, and score ② (embedding similarity)."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from app.embeddings import l2_normalize

log = logging.getLogger(__name__)

TASTE_SOURCE = "taste"


@dataclass
class TasteCluster:
    id: int
    centroid: np.ndarray
    member_ids: list[int]  # tmdb ids, highest-rated first
    label: str

    @property
    def source(self) -> str:
        return f"cluster:{self.id}"


@dataclass
class TasteModel:
    mean_rating: float
    n_rated: int
    taste_vector: np.ndarray
    clusters: list[TasteCluster] = field(default_factory=list)
    silhouette: float | None = None
    embedding_model: str = ""

    def vectors(self) -> list[tuple[str, np.ndarray]]:
        return [(TASTE_SOURCE, self.taste_vector)] + [(c.source, c.centroid) for c in self.clusters]

    def label_for(self, source: str) -> str:
        for c in self.clusters:
            if c.source == source:
                return c.label
        return "Your overall taste"

# ---------------------------------------------------------------- building


def build_taste_vector(ratings: np.ndarray, embeddings: np.ndarray) -> tuple[np.ndarray, float]:
    """Σ (rating − μ) · embedding, L2-normalized. Disliked films push it away."""
    mu = float(ratings.mean())
    vec = ((ratings - mu)[:, None] * embeddings).sum(axis=0)
    if np.linalg.norm(vec) < 1e-9:
        # Every rating identical: no signal in the deviations, fall back to the mean.
        vec = embeddings.mean(axis=0)
    return l2_normalize(vec.astype(np.float32)), mu


def _cluster_label(member_ids: list[int], genres: dict[int, list[str]], fallback: str) -> str:
    counts = Counter(g for i in member_ids for g in genres.get(i, []))
    top = [g for g, _ in counts.most_common(2)]
    return " · ".join(top) if top else fallback


def build_clusters(
    ids: list[int],
    embeddings: np.ndarray,
    ratings: np.ndarray,
    genres: dict[int, list[str]],
    k_range: tuple[int, int],
    seed: int = 0,
    min_size: int = 3,
) -> tuple[list[TasteCluster], float | None]:
    """k-means over liked films, k chosen by cosine silhouette score among the
    ks whose clusters all have ≥ `min_size` members."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(ids)
    k_min, k_max = k_range
    ks = [k for k in range(k_min, k_max + 1) if k <= n - 1 and k * min_size <= n]
    if n < 2 * k_min or not ks:
        log.info("only %d liked films: skipping taste clusters (need ≥ %d)", n, max(2 * k_min, k_min * min_size))
        return [], None

    best: tuple[float, int, np.ndarray] | None = None
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embeddings)
        sizes = np.bincount(labels, minlength=k)
        if sizes.min() < min_size:
            log.info("taste clusters: k=%d rejected (smallest cluster has %d films)", k, sizes.min())
            continue
        score = float(silhouette_score(embeddings, labels, metric="cosine"))
        log.info("taste clusters: k=%d silhouette=%.3f sizes=%s", k, score, sorted(sizes.tolist()))
        if best is None or score > best[0]:
            best = (score, k, labels)
    if best is None:
        return [], None

    score, k, labels = best
    clusters: list[TasteCluster] = []
    for cid in range(k):
        idx = np.flatnonzero(labels == cid)
        if len(idx) == 0:
            continue
        order = idx[np.argsort(-ratings[idx], kind="stable")]
        members = [ids[i] for i in order]
        clusters.append(
            TasteCluster(
                id=len(clusters),
                centroid=l2_normalize(embeddings[idx].mean(axis=0)),
                member_ids=members,
                label=_cluster_label(members, genres, f"Cluster {cid + 1}"),
            )
        )
    # Disambiguate identical labels ("Drama · Romance", "Drama · Romance (2)").
    seen: Counter[str] = Counter()
    for c in clusters:
        seen[c.label] += 1
        if seen[c.label] > 1:
            c.label = f"{c.label} ({seen[c.label]})"
    return clusters, score


def build_taste_model(
    ratings: dict[int, float],
    embeddings: dict[int, np.ndarray],
    genres: dict[int, list[str]],
    *,
    cluster_min_rating: float = 4.0,
    k_range: tuple[int, int] = (3, 6),
    min_cluster_size: int = 3,
    embedding_model: str = "",
    seed: int = 0,
) -> TasteModel | None:
    ids = [i for i in ratings if i in embeddings]
    missing = len(ratings) - len(ids)
    if missing:
        log.warning("%d rated films have no embedding and are left out of the taste model", missing)
    if not ids:
        log.warning("no rated films with embeddings: cannot build a taste model")
        return None

    r = np.array([ratings[i] for i in ids], dtype=np.float32)
    e = np.stack([embeddings[i] for i in ids]).astype(np.float32)
    taste, mu = build_taste_vector(r, e)

    liked = r >= cluster_min_rating
    clusters, sil = build_clusters(
        [i for i, keep in zip(ids, liked) if keep], e[liked], r[liked], genres, k_range, seed,
        min_cluster_size,
    )
    log.info(
        "taste model: %d rated films, μ=%.2f, %d clusters (silhouette %s)",
        len(ids), mu, len(clusters), f"{sil:.3f}" if sil is not None else "n/a",
    )
    return TasteModel(
        mean_rating=mu,
        n_rated=len(ids),
        taste_vector=taste,
        clusters=clusters,
        silhouette=sil,
        embedding_model=embedding_model,
    )


# ---------------------------------------------------------------- scoring


@dataclass(frozen=True)
class EmbeddingScore:
    raw: float  # best cosine similarity across taste vector / centroids
    source: str  # which vector gave the best (standardized) match
    normalized: float = 0.0  # percentile rank within the candidate set, 0–1


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] by rank (ties averaged). A single value maps to 1."""
    from scipy.stats import rankdata

    n = len(values)
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    return (rankdata(values, method="average") - 1) / (n - 1)


def score_embeddings(
    model: TasteModel,
    candidates: dict[int, np.ndarray],
    mode: Literal["zmax", "max"] = "zmax",
) -> dict[int, EmbeddingScore]:
    """Score ②. `zmax` standardizes each vector's similarities across the
    candidate set before taking the max, so the taste vector (a difference
    direction with small raw cosines) can compete with cluster centroids."""
    if not candidates:
        return {}
    ids = list(candidates)
    cand = np.stack([candidates[i] for i in ids]).astype(np.float32)
    sources = model.vectors()
    vecs = np.stack([v for _, v in sources]).astype(np.float32)
    sims = cand @ vecs.T  # (n_candidates, n_vectors)

    if mode == "zmax" and len(ids) > 1:
        std = sims.std(axis=0)
        comparable = (sims - sims.mean(axis=0)) / np.where(std > 1e-9, std, 1.0)
    else:
        comparable = sims
    best = comparable.argmax(axis=1)
    ranked = percentile_rank(comparable.max(axis=1))
    return {
        cid: EmbeddingScore(
            raw=float(sims[row, best[row]]),
            source=sources[best[row]][0],
            normalized=float(ranked[row]),
        )
        for row, cid in enumerate(ids)
    }
