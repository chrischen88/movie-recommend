"""The learned blend of scores ①②③, and the holdout metrics behind it.

Cross-fitting: the user's rated films are split into 5 folds. For each fold,
① (taste profile), ② (taste vector + clusters) and ③ (MovieLens fold-in) are
rebuilt from the *other* 80% of ratings only, and the fold's films are scored
alongside the current candidate pool, so each score is a percentile on the same
scale as in production. That yields honest out-of-fold (OOF) scores for every
rated film.

A Ridge regression with non-negative weights then predicts the rating from those
scores, plus standardized log TMDB vote count. Two models are fitted: one with
③ for films in MovieLens, one without for the rest. Its accuracy is measured by
a second 5-fold CV over the OOF rows, next to each score alone and the fixed-
weight fallback, which is used with fewer than `blend_min_ratings_for_learning`
ratings.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.collab import CollabScorer
from app.config import Settings
from app.db import Movie
from app.profile import build_profile, score_profile
from app.taste import build_taste_model, percentile_rank, score_embeddings

log = logging.getLogger(__name__)

SCORES = ("profile", "embedding", "collab")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
MIN_ROWS_FOR_METRICS = 10


@dataclass(frozen=True)
class OOFRow:
    tmdb_id: int
    rating: float
    profile: float
    embedding: float
    collab: float | None
    log_votes: float


@dataclass
class LinearBlend:
    features: list[str]
    coef: list[float]
    intercept: float
    alpha: float

    def predict(self, x: Mapping[str, float]) -> float:
        return self.intercept + sum(c * x[f] for f, c in zip(self.features, self.coef))


@dataclass
class BlendModel:
    mode: str  # "learned" | "fixed"
    fixed_weights: dict[str, float]
    full: LinearBlend | None = None  # with ③
    partial: LinearBlend | None = None  # without ③
    votes_mean: float = 0.0
    votes_std: float = 1.0
    use_votes: bool = True
    metrics: dict = field(default_factory=dict)
    n_ratings: int = 0
    n_with_collab: int = 0
    trained_at: str = ""

    # -- scoring

    def vote_feature(self, vote_count: int | None) -> float:
        return (math.log1p(vote_count or 0) - self.votes_mean) / (self.votes_std or 1.0)

    def predict(
        self, profile: float | None, embedding: float, collab: float | None, vote_count: int | None
    ) -> float | None:
        """Predicted rating in stars (learned mode), or None in fixed mode."""
        if self.mode != "learned":
            return None
        model = self.full if collab is not None and self.full is not None else self.partial
        if model is None:
            return None
        x = {"profile": profile if profile is not None else 0.5, "embedding": embedding,
             "collab": collab if collab is not None else 0.5, "votes": self.vote_feature(vote_count)}
        return model.predict(x)

    def fixed_score(self, profile: float | None, embedding: float, collab: float | None) -> float:
        """Weighted mean of the available scores; missing ones are re-weighted away."""
        parts = [(self.fixed_weights["profile"], profile), (self.fixed_weights["embedding"], embedding),
                 (self.fixed_weights["collab"], collab)]
        present = [(w, s) for w, s in parts if s is not None and w > 0]
        total = sum(w for w, _ in present)
        return sum(w * s for w, s in present) / total if total else embedding



def taste_fit(profile: float | None, embedding: float) -> float:
    """How much a film looks like your kind of film: the mean of ① and ②."""
    return embedding if profile is None else (profile + embedding) / 2


def rank_with_fit(predicted: np.ndarray, fit: np.ndarray, weight: float) -> np.ndarray:
    """The ranking score, 0–1: percentile of (1−w)·pct(predicted) + w·pct(fit).

    The learned blend predicts how you'd *rate* a film if you watched it, and on
    real data puts nearly all its weight on ③, so ① and ② stop mattering and the
    top fills with acclaimed films that aren't your kind of thing. Mixing taste
    fit back in trades a little predicted rating for recommendations that match
    what you choose to watch (PROGRESS.md, decision 23)."""
    if weight <= 0 or len(predicted) == 0:
        return percentile_rank(predicted)
    return percentile_rank((1 - weight) * percentile_rank(predicted) + weight * percentile_rank(fit))


def fixed_model(settings: Settings) -> BlendModel:
    w = settings.blend_fallback_weights
    return BlendModel(mode="fixed", fixed_weights=dict(zip(SCORES, w)))


# ---------------------------------------------------------------- out-of-fold scores


def _folds(ids: Sequence[int], k: int, seed: int) -> list[list[int]]:
    order = np.random.default_rng(seed).permutation(len(ids))
    return [[ids[i] for i in order[f::k]] for f in range(k)]


def out_of_fold(
    ratings: Mapping[int, float],
    movies: Mapping[int, Movie],
    embeddings: Mapping[int, np.ndarray],
    pool: Sequence[int],
    scorer: CollabScorer | None,
    settings: Settings,
) -> list[OOFRow]:
    """Honest ①②③ for every rated film that has metadata and an embedding."""
    rated = sorted(t for t in ratings if t in movies and t in embeddings)
    k = settings.blend_cv_folds
    if len(rated) < k:
        return []
    genres = {t: m.genres for t, m in movies.items()}
    pool = [t for t in pool if t in movies and t in embeddings]
    rows: list[OOFRow] = []
    for fold in _folds(rated, k, settings.collab_seed):
        held = set(fold)
        train = {t: r for t, r in ratings.items() if t not in held}
        ids = pool + fold
        profile = build_profile(train, movies, settings.shrinkage_k)
        taste = build_taste_model(
            train, dict(embeddings), genres,
            cluster_min_rating=settings.taste_cluster_min_rating,
            k_range=settings.taste_cluster_k_range,
            min_cluster_size=settings.taste_cluster_min_size,
        )
        if profile is None or taste is None:
            continue
        p = score_profile(profile, [movies[t] for t in ids], settings.feature_weights, 0)
        e = score_embeddings(taste, {t: embeddings[t] for t in ids}, settings.embedding_score_mode)
        c = {}
        if scorer is not None:
            user = scorer.fold_in(train, settings.collab_min_user_ratings)
            if user is not None:
                c = scorer.score(user, ids)
        for t in fold:
            rows.append(OOFRow(
                tmdb_id=t, rating=ratings[t], profile=p[t].normalized, embedding=e[t].normalized,
                collab=c[t].normalized if t in c else None,
                log_votes=math.log1p(movies[t].vote_count or 0),
            ))
    return rows


# ---------------------------------------------------------------- fitting + metrics


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic)


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def _cv_predict(model_factory, x: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    from sklearn.model_selection import KFold, cross_val_predict

    return cross_val_predict(model_factory(), x, y, cv=KFold(min(k, len(y)), shuffle=True, random_state=seed))


def _fit_ridge(x: np.ndarray, y: np.ndarray, features: list[str], k: int, seed: int) -> tuple[LinearBlend, np.ndarray]:
    """Pick alpha by k-fold CV, refit on everything. Returns the model and its
    CV predictions (for metrics)."""
    from sklearn.linear_model import Ridge

    best: tuple[float, float, np.ndarray] | None = None
    for alpha in ALPHAS:
        pred = _cv_predict(lambda: Ridge(alpha=alpha, positive=True), x, y, k, seed)
        err = _rmse(pred, y)
        if best is None or err < best[0]:
            best = (err, alpha, pred)
    assert best is not None
    _, alpha, pred = best
    model = Ridge(alpha=alpha, positive=True).fit(x, y)
    return LinearBlend(features, [float(c) for c in model.coef_], float(model.intercept_), alpha), pred


def _single_score_metrics(scores: np.ndarray, y: np.ndarray, k: int, seed: int) -> dict[str, float | None]:
    from sklearn.linear_model import LinearRegression

    pred = _cv_predict(LinearRegression, scores.reshape(-1, 1), y, k, seed)
    return {"rmse": _rmse(pred, y), "spearman": _spearman(scores, y), "n": int(len(y))}


def fit_blend(rows: Sequence[OOFRow], settings: Settings) -> BlendModel:
    k, seed = settings.blend_cv_folds, settings.collab_seed
    model = fixed_model(settings)
    model.n_ratings = len(rows)
    model.use_votes = settings.blend_use_vote_count
    model.trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if len(rows) < MIN_ROWS_FOR_METRICS:
        log.info("only %d scored ratings: fixed blend weights, no metrics", len(rows))
        return model

    y = np.array([r.rating for r in rows])
    votes = np.array([r.log_votes for r in rows])
    model.votes_mean, model.votes_std = float(votes.mean()), float(votes.std() or 1.0)
    z_votes = (votes - model.votes_mean) / model.votes_std
    has_c = np.array([r.collab is not None for r in rows])
    model.n_with_collab = int(has_c.sum())
    cols = {
        "profile": np.array([r.profile for r in rows]),
        "embedding": np.array([r.embedding for r in rows]),
        "collab": np.array([r.collab if r.collab is not None else np.nan for r in rows]),
        "votes": z_votes,
    }

    # Baseline: always predict the mean of the training folds.
    from sklearn.dummy import DummyRegressor

    metrics: dict = {"baseline_mean": {"rmse": _rmse(_cv_predict(DummyRegressor, z_votes.reshape(-1, 1), y, k, seed), y),
                                       "spearman": None, "n": len(rows)}}
    for name in SCORES:
        mask = ~np.isnan(cols[name])
        if mask.sum() >= MIN_ROWS_FOR_METRICS:
            metrics[name] = _single_score_metrics(cols[name][mask], y[mask], k, seed)
    fixed = np.array([model.fixed_score(r.profile, r.embedding, r.collab) for r in rows])
    metrics["fixed_blend"] = _single_score_metrics(fixed, y, k, seed)

    extra = ["votes"] if model.use_votes else []
    partial_feats = ["profile", "embedding", *extra]
    x_partial = np.column_stack([cols[f] for f in partial_feats])
    model.partial, pred = _fit_ridge(x_partial, y, partial_feats, k, seed)
    blended = pred.copy()
    if model.n_with_collab >= max(MIN_ROWS_FOR_METRICS, 2 * k):
        full_feats = ["profile", "embedding", "collab", *extra]
        x_full = np.column_stack([cols[f][has_c] for f in full_feats])
        model.full, pred_full = _fit_ridge(x_full, y[has_c], full_feats, k, seed)
        blended[has_c] = pred_full
    metrics["learned_blend"] = {"rmse": _rmse(blended, y), "spearman": _spearman(blended, y), "n": len(rows)}
    if settings.rank_fit_weight > 0:
        fit = np.array([taste_fit(r.profile, r.embedding) for r in rows])
        ranked = rank_with_fit(blended, fit, settings.rank_fit_weight)
        metrics["ranking"] = {"rmse": None, "spearman": _spearman(ranked, y), "n": len(rows)}
    model.metrics = metrics

    if len(rows) >= settings.blend_min_ratings_for_learning:
        model.mode = "learned"
    log.info("blend: mode=%s, %d ratings (%d with ③), learned RMSE %.3f vs baseline %.3f",
             model.mode, len(rows), model.n_with_collab, metrics["learned_blend"]["rmse"],
             metrics["baseline_mean"]["rmse"])
    return model

