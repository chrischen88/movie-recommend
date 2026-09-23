"""A tiny in-memory TMDB for tests, served through httpx.MockTransport."""

from __future__ import annotations

from typing import Any

import httpx

from app.matching import normalize_for_match


def movie(
    tmdb_id: int, title: str, year: int | None, votes: int = 1000, **extra: Any
) -> dict[str, Any]:
    return {
        "id": tmdb_id,
        "title": title,
        "original_title": extra.pop("original_title", title),
        "release_date": f"{year}-06-01" if year else "",
        "vote_count": votes,
        "vote_average": 7.5,
        "popularity": votes / 100,
        "poster_path": f"/p{tmdb_id}.jpg",
        **extra,
    }


class FakeTmdb:
    def __init__(self, movies: list[dict[str, Any]]) -> None:
        self.movies = {m["id"]: m for m in movies}
        self.requests: list[httpx.Request] = []
        self.fail_ids: set[int] = set()
        self.status_override: int | None = None

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _search(self, params: httpx.QueryParams) -> dict[str, Any]:
        q = normalize_for_match(params["query"])
        year = params.get("year")
        results = [
            m
            for m in self.movies.values()
            if q and q in normalize_for_match(m["title"])
            and (year is None or m["release_date"].startswith(year))
        ]
        return {"page": 1, "results": results, "total_results": len(results)}

    def _details(self, m: dict[str, Any]) -> dict[str, Any]:
        tid = m["id"]
        return {
            **m,
            "overview": f"Overview of {m['title']}.",
            "runtime": 100 + tid % 60,
            "original_language": "en",
            "genres": [{"id": 18, "name": "Drama"}, {"id": 878, "name": "Science Fiction"}],
            "production_countries": [{"iso_3166_1": "US", "name": "United States of America"}],
            "credits": {
                "cast": [{"name": f"Actor {i}", "order": 6 - i} for i in range(7)],
                "crew": [
                    {"job": "Director", "name": f"Director {tid}"},
                    {"job": "Writer", "name": "Someone"},
                ],
            },
            "keywords": {"keywords": [{"id": 1, "name": "alien"}, {"id": 2, "name": "language"}]},
            "external_ids": {"imdb_id": f"tt{tid:07d}"},
        }

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status_override:
            return httpx.Response(self.status_override, json={"status_message": "nope"})
        path = request.url.path.removeprefix("/3")
        if path == "/search/movie":
            return httpx.Response(200, json=self._search(request.url.params))
        parts = path.strip("/").split("/")
        if parts[0] == "movie" and parts[1].isdigit():
            tid = int(parts[1])
            if tid in self.fail_ids:
                return httpx.Response(500)
            m = self.movies.get(tid)
            if m is None:
                return httpx.Response(404, json={"status_code": 34})
            if len(parts) == 2:
                return httpx.Response(200, json=self._details(m))
            if parts[2] == "reviews":
                return httpx.Response(
                    200,
                    json={"results": [{"author": "a", "content": "Great. " * 200}, {"content": "ok"}]},
                )
        return httpx.Response(404)
