"""TMDB API client (cached + throttled) and metadata parsing."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.cache import CachedHttpClient, ResponseCache
from app.config import Settings
from app.db import Movie, utcnow

log = logging.getLogger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
REVIEW_MAX_CHARS = 500
MAX_REVIEWS = 5
TOP_CAST = 5


class TmdbClient:
    def __init__(
        self,
        api_key: str,
        cache: ResponseCache,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        # v4 "API Read Access Tokens" are JWTs and go in a Bearer header;
        # v3 keys go in the api_key query param.
        is_bearer = api_key.startswith("eyJ")
        self.http = CachedHttpClient(
            namespace="tmdb",
            base_url=TMDB_BASE_URL,
            cache=cache,
            rate_per_second=settings.tmdb_requests_per_second,
            ttl_seconds=settings.cache_ttl_tmdb,
            secret_params=None if is_bearer else {"api_key": api_key},
            secret_headers={"Authorization": f"Bearer {api_key}"} if is_bearer else None,
            max_retries=settings.http_max_retries,
            backoff_base=settings.http_backoff_base_seconds,
            backoff_max=settings.http_backoff_max_seconds,
            timeout=settings.http_timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self.http.close()

    def search_movie(self, query: str, year: int | None = None) -> list[dict[str, Any]]:
        body = self.http.get_json(
            "/search/movie", {"query": query, "year": year, "include_adult": "false"}
        )
        return list(body.get("results", [])) if isinstance(body, dict) else []

    def movie_details(self, tmdb_id: int) -> dict[str, Any] | None:
        body = self.http.get_json(
            f"/movie/{tmdb_id}", {"append_to_response": "credits,keywords,external_ids"}
        )
        return body if isinstance(body, dict) else None

    def movie_reviews(self, tmdb_id: int) -> list[dict[str, Any]]:
        body = self.http.get_json(f"/movie/{tmdb_id}/reviews", {"page": 1})
        return list(body.get("results", [])) if isinstance(body, dict) else []

    def recommendations(self, tmdb_id: int, page: int = 1) -> list[dict[str, Any]]:
        body = self.http.get_json(f"/movie/{tmdb_id}/recommendations", {"page": page})
        return list(body.get("results", [])) if isinstance(body, dict) else []

    def similar(self, tmdb_id: int, page: int = 1) -> list[dict[str, Any]]:
        body = self.http.get_json(f"/movie/{tmdb_id}/similar", {"page": page})
        return list(body.get("results", [])) if isinstance(body, dict) else []

    def discover(self, **params: Any) -> list[dict[str, Any]]:
        body = self.http.get_json("/discover/movie", params)
        return list(body.get("results", [])) if isinstance(body, dict) else []


def year_of(release_date: str | None) -> int | None:
    if release_date and len(release_date) >= 4 and release_date[:4].isdigit():
        return int(release_date[:4])
    return None


def _truncate(text: str, limit: int = REVIEW_MAX_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def parse_movie(details: dict[str, Any], reviews: list[dict[str, Any]]) -> Movie:
    """Build a Movie row from a `/movie/{id}?append_to_response=…` body + reviews."""
    tmdb_id = int(details["id"])
    credits = details.get("credits") or {}
    crew = credits.get("crew") or []
    cast = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 999))
    keywords = (details.get("keywords") or {}).get("keywords") or []
    external = details.get("external_ids") or {}

    missing = [
        f for f in ("overview", "runtime", "release_date", "genres") if not details.get(f)
    ]
    if missing:
        log.info("tmdb %s (%s): missing fields %s", tmdb_id, details.get("title"), missing)

    directors: list[str] = []
    for member in crew:
        if member.get("job") == "Director" and member.get("name") not in directors:
            directors.append(member["name"])

    return Movie(
        tmdb_id=tmdb_id,
        title=details.get("title") or details.get("original_title") or f"TMDB {tmdb_id}",
        original_title=details.get("original_title"),
        year=year_of(details.get("release_date")),
        release_date=details.get("release_date") or None,
        overview=details.get("overview") or None,
        runtime=details.get("runtime") or None,
        original_language=details.get("original_language") or None,
        genres=[g["name"] for g in details.get("genres") or [] if g.get("name")],
        directors=directors,
        cast=[c["name"] for c in cast[:TOP_CAST] if c.get("name")],
        keywords=[k["name"] for k in keywords if k.get("name")],
        countries=[
            c.get("iso_3166_1") or c.get("name")
            for c in details.get("production_countries") or []
        ],
        poster_path=details.get("poster_path") or None,
        vote_average=details.get("vote_average"),
        vote_count=details.get("vote_count"),
        popularity=details.get("popularity"),
        imdb_id=external.get("imdb_id") or details.get("imdb_id") or None,
        reviews=[
            _truncate(r["content"]) for r in reviews[:MAX_REVIEWS] if r.get("content")
        ],
        enriched_at=utcnow(),
    )


def fetch_movie(client: TmdbClient, tmdb_id: int) -> Movie | None:
    """Fetch + parse one movie; None if TMDB has no such id."""
    details = client.movie_details(tmdb_id)
    if details is None:
        log.warning("tmdb id %s not found", tmdb_id)
        return None
    return parse_movie(details, client.movie_reviews(tmdb_id))
