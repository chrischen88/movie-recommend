"""Score ③: collaborative filtering over MovieLens.

Explicit matrix factorization with bias terms, r̂ = μ + b_u + b_i + p_u·q_i,
fitted by alternating least squares with weighted-λ regularization (each
row's penalty scales with its number of ratings). Plain numpy/scipy, since
`implicit` may lack Python 3.13 wheels and is built for implicit feedback anyway.

The base model is trained once on MovieLens and saved. The user is then
"folded in": one ALS user step against the frozen item factors, which is
exactly what adding them as a new row would give for their own vector, minus
their (negligible) pull on the item factors. It takes microseconds, so it runs
per request and always reflects the current ratings (PROGRESS.md, decision 13).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from app.movielens import MovieLensData
from app.taste import percentile_rank

log = logging.getLogger(__name__)

MIN_STARS, MAX_STARS = 0.5, 5.0


@dataclass(frozen=True)
class TrainParams:
    factors: int = 32
    iterations: int = 15
    reg: float = 0.05
    val_fraction: float = 0.05
    seed: int = 0

    def fingerprint(self, dataset: str) -> str:
        raw = json.dumps({"dataset": dataset, **self.__dict__}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class MFModel:
    mu: float
    item_factors: np.ndarray  # (n_items, k) float32
    item_bias: np.ndarray  # (n_items,) float32
    item_counts: np.ndarray  # (n_items,) training ratings per item
    movie_ids: np.ndarray
    tmdb_ids: np.ndarray  # -1 when unknown
    reg: float
    meta: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp,
            mu=np.float64(self.mu),
            item_factors=self.item_factors,
            item_bias=self.item_bias,
            item_counts=self.item_counts,
            movie_ids=self.movie_ids,
            tmdb_ids=self.tmdb_ids,
            reg=np.float64(self.reg),
            meta=np.array(json.dumps(self.meta)),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> MFModel | None:
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                return cls(
                    mu=float(z["mu"]),
                    item_factors=z["item_factors"],
                    item_bias=z["item_bias"],
                    item_counts=z["item_counts"],
                    movie_ids=z["movie_ids"],
                    tmdb_ids=z["tmdb_ids"],
                    reg=float(z["reg"]),
                    meta=json.loads(str(z["meta"])),
                )
        except (OSError, KeyError, ValueError) as exc:
            log.error("could not read collaborative model %s: %s", path, exc)
            return None


# ---------------------------------------------------------------- training


def _solve_rows(
    indptr: np.ndarray,
    indices: np.ndarray,
    targets: np.ndarray,
    fixed_factors: np.ndarray,
    fixed_bias: np.ndarray,
    reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One ALS half-step: for each row, least-squares [factors, bias] against the
    fixed side. `targets` are ratings minus μ, aligned with `indices`."""
    n_rows, k = len(indptr) - 1, fixed_factors.shape[1]
    design = np.hstack([fixed_factors, np.ones((len(fixed_factors), 1), dtype=fixed_factors.dtype)])
    eye = np.eye(k + 1)
    out = np.zeros((n_rows, k + 1))
    for r in range(n_rows):
        lo, hi = indptr[r], indptr[r + 1]
        if lo == hi:
            continue
        cols = indices[lo:hi]
        x = design[cols]
        y = targets[lo:hi] - fixed_bias[cols]
        out[r] = np.linalg.solve(x.T @ x + reg * (hi - lo) * eye, x.T @ y)
    return out[:, :k].astype(np.float32), out[:, k].astype(np.float32)


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2))) if len(truth) else float("nan")


def train_als(
    data: MovieLensData,
    params: TrainParams,
    dataset: str = "",
    on_iteration: Callable[[int, int], None] | None = None,
) -> MFModel:
    """Fit the base model. A random `val_fraction` of ratings is held out to
    report RMSE against a global-mean baseline (logged and kept in `meta`)."""
    started = time.monotonic()
    rng = np.random.default_rng(params.seed)
    val = rng.random(len(data.rating)) < params.val_fraction
    tr = ~val
    u, i, r = data.user_idx[tr], data.item_idx[tr], data.rating[tr]
    mu = float(r.mean())
    shape = (data.n_users, data.n_items)
    by_user = csr_matrix((r - mu, (u, i)), shape=shape, dtype=np.float64)
    by_item = by_user.T.tocsr()
    by_user.sort_indices()
    by_item.sort_indices()

    k = params.factors
    q = rng.normal(0.0, 0.1, (data.n_items, k)).astype(np.float32)
    b_i = np.zeros(data.n_items, dtype=np.float32)
    p = np.zeros((data.n_users, k), dtype=np.float32)
    b_u = np.zeros(data.n_users, dtype=np.float32)
    vu, vi, vr = data.user_idx[val], data.item_idx[val], data.rating[val]
    val_rmse = float("nan")
    for it in range(params.iterations):
        p, b_u = _solve_rows(by_user.indptr, by_user.indices, by_user.data, q, b_i, params.reg)
        q, b_i = _solve_rows(by_item.indptr, by_item.indices, by_item.data, p, b_u, params.reg)
        if len(vr):
            pred = np.clip(mu + b_u[vu] + b_i[vi] + (p[vu] * q[vi]).sum(axis=1), MIN_STARS, MAX_STARS)
            val_rmse = _rmse(pred, vr)
        log.info("ALS iteration %d/%d: validation RMSE %.4f", it + 1, params.iterations, val_rmse)
        if on_iteration:
            on_iteration(it + 1, params.iterations)

    meta = {
        "dataset": dataset,
        "fingerprint": params.fingerprint(dataset),
        "params": params.__dict__,
        "n_users": data.n_users,
        "n_items": data.n_items,
        "n_ratings": int(len(data.rating)),
        "val_rmse": val_rmse,
        "val_rmse_global_mean": _rmse(np.full(len(vr), mu), vr),
        "train_seconds": round(time.monotonic() - started, 1),
    }
    log.info("collaborative model trained: %s", meta)
    return MFModel(
        mu=mu,
        item_factors=q,
        item_bias=b_i,
        item_counts=np.bincount(i, minlength=data.n_items).astype(np.int32),
        movie_ids=data.movie_ids,
        tmdb_ids=data.tmdb_ids,
        reg=params.reg,
        meta=meta,
    )


# ---------------------------------------------------------------- scoring


@dataclass(frozen=True)
class FoldedUser:
    factors: np.ndarray
    bias: float
    n_mapped: int  # the user's rated films found in MovieLens


@dataclass(frozen=True)
class CollabScore:
    predicted: float  # predicted rating in stars, 0.5–5
    normalized: float  # percentile rank among candidates that have a prediction, 0–1


class CollabScorer:
    def __init__(self, model: MFModel, min_item_ratings: int = 5) -> None:
        self.model = model
        # Several movieIds can share a tmdbId (re-releases, duplicates):
        # keep the one with the most ratings.
        best: dict[int, int] = {}
        for idx, tid in enumerate(model.tmdb_ids.tolist()):
            if tid < 0 or model.item_counts[idx] < min_item_ratings:
                continue
            if tid not in best or model.item_counts[idx] > model.item_counts[best[tid]]:
                best[tid] = idx
        self.item_for_tmdb = best

    def fold_in(self, ratings: Mapping[int, float], min_ratings: int = 5) -> FoldedUser | None:
        """Solve the user's factors and bias against the frozen item factors."""
        pairs = [(self.item_for_tmdb[t], r) for t, r in ratings.items() if t in self.item_for_tmdb]
        if len(pairs) < min_ratings:
            log.info("only %d rated films are in MovieLens (need %d): no collaborative score", len(pairs), min_ratings)
            return None
        items = np.array([i for i, _ in pairs])
        y = np.array([r for _, r in pairs], dtype=np.float64) - self.model.mu - self.model.item_bias[items]
        m = self.model
        x = np.hstack([m.item_factors[items], np.ones((len(items), 1), dtype=np.float32)]).astype(np.float64)
        w = np.linalg.solve(x.T @ x + m.reg * len(items) * np.eye(x.shape[1]), x.T @ y)
        return FoldedUser(factors=w[:-1].astype(np.float32), bias=float(w[-1]), n_mapped=len(items))

    def predict(self, user: FoldedUser, tmdb_ids: list[int]) -> dict[int, float]:
        found = [(t, self.item_for_tmdb[t]) for t in tmdb_ids if t in self.item_for_tmdb]
        if not found:
            return {}
        items = np.array([i for _, i in found])
        m = self.model
        pred = m.mu + user.bias + m.item_bias[items] + m.item_factors[items] @ user.factors
        pred = np.clip(pred, MIN_STARS, MAX_STARS)
        return {t: float(p) for (t, _), p in zip(found, pred)}

    def top_unseen(
        self, user: FoldedUser, exclude: set[int], n: int, min_ratings: int = 0
    ) -> list[tuple[int, float]]:
        """The `n` highest predicted films not in `exclude` with at least
        `min_ratings` MovieLens ratings, as (tmdb_id, stars). Thinly rated films
        top this list on noise: their few raters are self-selected fans."""
        counts = self.model.item_counts
        ids = [t for t, i in self.item_for_tmdb.items() if t not in exclude and counts[i] >= min_ratings]
        preds = self.predict(user, ids)
        return sorted(preds.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def score(self, user: FoldedUser, tmdb_ids: list[int]) -> dict[int, CollabScore]:
        """Score ③. Films not in MovieLens (or too thinly rated there) are absent."""
        preds = self.predict(user, tmdb_ids)
        ranked = percentile_rank(np.array(list(preds.values())))
        return {t: CollabScore(predicted=p, normalized=float(pct)) for (t, p), pct in zip(preds.items(), ranked)}


_scorer_cache: dict[Path, tuple[float, int, CollabScorer]] = {}


def load_scorer(path: Path, min_item_ratings: int) -> CollabScorer | None:
    """The saved model, cached in memory until the file changes."""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    hit = _scorer_cache.get(path)
    if hit and hit[0] == mtime and hit[1] == min_item_ratings:
        return hit[2]
    model = MFModel.load(path)
    if model is None:
        return None
    scorer = CollabScorer(model, min_item_ratings)
    _scorer_cache[path] = (mtime, min_item_ratings, scorer)
    return scorer
