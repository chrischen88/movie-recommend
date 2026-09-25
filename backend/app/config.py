"""Central configuration.

Secrets (API keys) come from `.env` / the environment only. Everything else here
is a tunable default that can also be overridden via environment variables
(e.g. `BLEND_FALLBACK_WEIGHTS='[0.2,0.5,0.3]'`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Secrets (never hard-code; all optional so the app degrades gracefully) ---
    tmdb_api_key: str | None = None
    omdb_api_key: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # --- Access (hosted deployments) ---
    # Setting a password puts every route except /api/health behind HTTP Basic auth.
    auth_username: str = "letterboxd"
    auth_password: str | None = None

    # --- Storage ---
    data_dir: Path = BACKEND_DIR / "data"
    db_filename: str = "app.sqlite3"
    chroma_dirname: str = "chroma"
    # Built UI (`npm run build`). When present, the API also serves it, so one
    # process on one port hosts the whole app; in dev, Vite serves the UI instead.
    frontend_dist: Path = PROJECT_DIR / "frontend" / "dist"

    # --- HTTP cache / rate limiting ---
    # TTL in seconds per namespace; None = never expires.
    cache_ttl_default: int | None = 60 * 60 * 24 * 30
    cache_ttl_tmdb: int | None = 60 * 60 * 24 * 30
    cache_ttl_omdb: int | None = 60 * 60 * 24 * 90
    tmdb_requests_per_second: float = 20.0
    omdb_requests_per_second: float = 5.0
    omdb_daily_limit: int = 1000
    omdb_shortlist_size: int = 150  # top-ranked candidates looked up on OMDb each run
    http_max_retries: int = 5
    http_backoff_base_seconds: float = 0.5
    http_backoff_max_seconds: float = 30.0
    http_timeout_seconds: float = 20.0

    # --- Upload limits ---
    max_upload_bytes: int = 50 * 1024 * 1024
    max_uncompressed_csv_bytes: int = 100 * 1024 * 1024
    max_export_films: int = 5000

    # --- User sessions (in memory only; nothing about a user is written to disk) ---
    session_ttl_seconds: int = 60 * 60  # dropped after this long without a request
    max_sessions: int = 20
    max_queued_runs: int = 5  # pipeline runs waiting or running; one runs at a time
    uploads_per_ip_per_hour: int = 10  # 0 disables the limit
    # Sample-profile sessions (app/demo.py), counted separately: once its films are
    # cached a demo run costs no API calls or embedding, and visitors from one
    # office share an IP. The session and queue caps above still apply.
    demo_sessions_per_ip_per_hour: int = 60  # 0 disables the limit

    # --- TMDB matching ---
    match_low_confidence_threshold: float = 0.75

    # --- MovieLens / collaborative filtering (score ③) ---
    movielens_dataset: str = "ml-latest-small"  # or "ml-32m" (≈240 MB download, minutes to train)
    movielens_auto_download: bool = True  # fetch the dataset during a pipeline run if missing
    collab_enabled: bool = True
    collab_factors: int = 32
    collab_iterations: int = 15
    collab_reg: float = 0.1  # weighted-λ; best of 0.02–0.2 on ml-latest-small (val RMSE 0.845)
    collab_val_fraction: float = 0.05  # MovieLens ratings held out to report RMSE
    collab_seed: int = 0
    collab_min_item_ratings: int = 5  # films rated by fewer MovieLens users get no score ③
    collab_min_user_ratings: int = 5  # your rated films that must be in MovieLens for score ③

    # --- Candidate generation ---
    candidate_seed_count: int = 60  # top-rated films used as recommendation seeds
    candidate_seed_min_rating: float = 4.0
    candidate_min_votes: int = 50  # skip obscure TMDB entries with too few votes
    # /discover/movie: acclaimed films in the genres and languages your profile likes most.
    candidate_discover_genres: int = 4
    candidate_discover_languages: int = 2  # non-English languages you rate above average
    candidate_discover_pages: int = 2
    candidate_discover_min_votes: int = 300
    # Score ③'s top predictions among films you haven't seen (needs a MovieLens model).
    candidate_collab_count: int = 150
    # ...among films with this many MovieLens ratings (③ itself scores films with
    # `collab_min_item_ratings`; this only stops niche films entering on noise).
    candidate_collab_min_ratings: int = 100

    # --- Embeddings / vector DB ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64  # films embedded and written to the index per step
    # Texts per model call. Peak memory grows with batch × text length²: 64 long
    # film documents need ~3 GB, 8 need ~0.9 GB, at the same speed on one CPU.
    embedding_model_batch_size: int = 8
    vector_collection: str = "movies"
    taste_cluster_min_rating: float = 4.0
    taste_cluster_k_range: tuple[int, int] = (3, 6)
    # A k is only eligible if every cluster has at least this many films, so a
    # single stray favourite can't become its own "taste" and flood the results.
    taste_cluster_min_size: int = 3
    doc_max_reviews: int = 2
    # "zmax": standardize each taste/cluster vector's similarities before taking
    # the max (see taste.score_embeddings); "max": raw max cosine.
    embedding_score_mode: Literal["zmax", "max"] = "zmax"

    # --- Taste profile (score ①) ---
    shrinkage_k: float = 3.0  # pseudo-count for Bayesian shrinkage toward 0
    profile_reasons_per_film: int = 5  # top contributions kept for explanations
    profile_min_reason_stars: float = 0.05  # hide explanations weaker than this (in ★)
    feature_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "director": 1.0,
            "genre": 0.6,
            "actor": 0.5,
            "keyword": 0.4,
            "decade": 0.3,
            "language": 0.3,
            "country": 0.2,
        }
    )

    # --- Blending ---
    blend_min_ratings_for_learning: int = 50
    blend_fallback_weights: tuple[float, float, float] = (0.3, 0.4, 0.3)
    blend_holdout_fraction: float = 0.2
    blend_cv_folds: int = 5
    blend_use_vote_count: bool = True  # log TMDB vote count as a 4th blend feature
    # Learned mode ranks by (1−w)·predicted rating + w·taste fit (mean of ① and ②),
    # both as percentiles. 0 ranks by predicted rating alone.
    rank_fit_weight: float = 0.5
    watchlist_boost: float = 0.05
    quality_floor_tomatometer: int = 60
    quality_floor_imdb: float = 6.5
    mmr_lambda: float = 0.7
    recommend_min_runtime: int = 40  # minutes; hides shorts (unless watchlisted). 0 shows everything
    top_n: int = 20

    # --- LLM ---
    llm_rerank_pool: int = 30
    llm_max_position_shift: int = 10

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / self.chroma_dirname

    @property
    def movielens_root(self) -> Path:
        return self.data_dir / "movielens"

    @property
    def movielens_dir(self) -> Path:
        return self.movielens_root / self.movielens_dataset

    @property
    def collab_model_path(self) -> Path:
        return self.movielens_root / f"model-{self.movielens_dataset}.npz"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
