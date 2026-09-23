"""A fake OMDb served through httpx.MockTransport. Ratings are derived from the
IMDb id (fake TMDB uses tt + tmdb id), with the usual gaps: some films have no
Rotten Tomatoes score or Metascore."""

from __future__ import annotations

from typing import Any

import httpx


def omdb_body(tmdb_id: int) -> dict[str, Any]:
    imdb = 5.0 + tmdb_id % 5  # 5.0–9.0
    ratings = [{"Source": "Internet Movie Database", "Value": f"{imdb:.1f}/10"}]
    if tmdb_id % 3:
        ratings.append({"Source": "Rotten Tomatoes", "Value": f"{30 + (tmdb_id % 7) * 10}%"})
    meta = "N/A" if tmdb_id % 4 == 0 else str(40 + tmdb_id % 50)
    return {"Title": f"Film {tmdb_id}", "imdbRating": f"{imdb:.1f}", "Metascore": meta,
            "Ratings": ratings, "Response": "True"}


class FakeOmdb:
    def __init__(self, status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.status = status

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"Response": "False", "Error": "Request limit reached!"})
        imdb_id = request.url.params.get("i", "")
        if not imdb_id.startswith("tt") or not imdb_id[2:].isdigit():
            return httpx.Response(200, json={"Response": "False", "Error": "Incorrect IMDb ID."})
        return httpx.Response(200, json=omdb_body(int(imdb_id[2:])))
