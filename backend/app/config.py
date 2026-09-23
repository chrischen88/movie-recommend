"""Central configuration.

Secrets (API keys) come from `.env` / the environment only. Everything else here
is a tunable default that can also be overridden via environment variables
(e.g. `BLEND_FALLBACK_WEIGHTS='[0.2,0.5,0.3]'`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # --- Storage ---
    data_dir: Path = BACKEND_DIR / "data"
    db_filename: str = "app.sqlite3"
    chroma_dirname: str = "chroma"

    # --- HTTP cache / rate limiting ---
    # TTL in seconds per namespace; None = never expires.
    cache_ttl_default: int | None = 60 * 60 * 24 * 30
    cache_ttl_tmdb: int | None = 60 * 60 * 24 * 30
    cache_ttl_omdb: int | None = 60 * 60 * 24 * 90
    tmdb_requests_per_second: float = 20.0
    omdb_requests_per_second: float = 5.0
    omdb_daily_limit: int = 1000
    http_max_retries: int = 5
    http_backoff_base_seconds: float = 0.5
    http_backoff_max_seconds: float = 30.0
    http_timeout_seconds: float = 20.0

    # --- Upload limits ---
    max_upload_bytes: int = 50 * 1024 * 1024
    max_uncompressed_csv_bytes: int = 100 * 1024 * 1024

    # --- TMDB matching ---
    match_low_confidence_threshold: float = 0.75

    # --- MovieLens ---
    movielens_dataset: str = "ml-latest-small"  # or "ml-32m"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    taste_cluster_min_rating: float = 4.0
    taste_cluster_k_range: tuple[int, int] = (3, 6)

    # --- Taste profile (score ①) ---
    shrinkage_k: float = 3.0  # pseudo-count for Bayesian shrinkage toward 0
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
    watchlist_boost: float = 0.05
    quality_floor_tomatometer: int = 60
    quality_floor_imdb: float = 6.5
    mmr_lambda: float = 0.7
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


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
