"""OMDb client: IMDb rating, Rotten Tomatoes Tomatometer and Metascore by IMDb id.

The free tier allows 1,000 requests a day, so this is only called for the
shortlist (the top-ranked candidates), every response is cached for
`cache_ttl_omdb`, and `CachedHttpClient` enforces `omdb_daily_limit`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.cache import CachedHttpClient, ResponseCache
from app.config import Settings

log = logging.getLogger(__name__)

OMDB_BASE_URL = "https://www.omdbapi.com"


@dataclass(frozen=True)
class OmdbRatings:
    imdb_rating: float | None = None  # 0–10
    rt_score: int | None = None  # 0–100
    metacritic: int | None = None  # 0–100


class OmdbClient:
    def __init__(
        self,
        api_key: str,
        cache: ResponseCache,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.http = CachedHttpClient(
            namespace="omdb",
            base_url=OMDB_BASE_URL,
            cache=cache,
            rate_per_second=settings.omdb_requests_per_second,
            ttl_seconds=settings.cache_ttl_omdb,
            secret_params={"apikey": api_key},
            max_retries=settings.http_max_retries,
            backoff_base=settings.http_backoff_base_seconds,
            backoff_max=settings.http_backoff_max_seconds,
            timeout=settings.http_timeout_seconds,
            daily_limit=settings.omdb_daily_limit,
            transport=transport,
        )

    def close(self) -> None:
        self.http.close()

    def ratings(self, imdb_id: str) -> OmdbRatings:
        """Missing fields (common) come back as None. An unknown id is cached and
        returns empty ratings; auth or quota errors raise ApiError."""
        body = self.http.get_json("/", {"i": imdb_id})
        if not isinstance(body, dict) or body.get("Response") == "False":
            if isinstance(body, dict):
                log.info("OMDb has no entry for %s: %s", imdb_id, body.get("Error"))
            return OmdbRatings()
        return parse_ratings(body)


def _number(text: Any, suffix: str = "") -> float | None:
    if not isinstance(text, str):
        return None
    text = text.strip().removesuffix(suffix).replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None  # "N/A"


def parse_ratings(body: dict[str, Any]) -> OmdbRatings:
    """Prefer the `Ratings` array; fall back to the top-level fields."""
    by_source = {r.get("Source"): r.get("Value") for r in body.get("Ratings") or [] if isinstance(r, dict)}
    imdb = _number(by_source.get("Internet Movie Database"), "/10")
    if imdb is None:
        imdb = _number(body.get("imdbRating"))
    rt = _number(by_source.get("Rotten Tomatoes"), "%")
    meta = _number(by_source.get("Metacritic"), "/100")
    if meta is None:
        meta = _number(body.get("Metascore"))
    return OmdbRatings(
        imdb_rating=imdb,
        rt_score=int(rt) if rt is not None else None,
        metacritic=int(meta) if meta is not None else None,
    )
