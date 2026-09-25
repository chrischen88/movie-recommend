"""One user's films and everything derived from them, held in memory only.

A `Library` is built from an uploaded export, lives in its `UserSession`, and is
dropped with it: nothing here is ever written to disk. What *is* shared and
persisted is film data (`Movie`, the API cache, embeddings), keyed by TMDB id.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from app import matching
from app.letterboxd import LetterboxdExport, ParsedFilm

if TYPE_CHECKING:
    from app.blend import BlendModel
    from app.taste import TasteModel


@dataclass
class LibraryFilm(ParsedFilm):
    """A film from the export plus its TMDB match."""

    tmdb_id: int | None = None
    match_confidence: float | None = None
    match_status: str | None = None  # matched | low_confidence | unmatched | manual | error | ignored
    match_note: str | None = None  # human-readable reason, shown in match review


class InvalidFixes(ValueError):
    """The match fixes sent with an upload aren't in the expected shape."""


@dataclass
class Library:
    films: dict[str, LibraryFilm]
    # tmdb_id → where the candidate came from, e.g. ["recommendations:329865"]
    candidates: dict[int, list[str]] = field(default_factory=dict)
    taste: TasteModel | None = None
    blend: BlendModel | None = None

    # ------------------------------------------------------------ views

    def ratings(self) -> dict[int, float]:
        """tmdb_id → rating. Two Letterboxd entries can map to one TMDB film: average them."""
        per_film: dict[int, list[float]] = {}
        for f in self.films.values():
            if f.tmdb_id is not None and f.rating is not None:
                per_film.setdefault(f.tmdb_id, []).append(f.rating)
        return {tid: sum(rs) / len(rs) for tid, rs in per_film.items()}

    def own_ids(self) -> set[int]:
        return {f.tmdb_id for f in self.films.values() if f.tmdb_id is not None}

    def seen_ids(self) -> set[int]:
        return {f.tmdb_id for f in self.films.values() if f.watched and f.tmdb_id is not None}

    def watchlist_ids(self) -> set[int]:
        return {f.tmdb_id for f in self.films.values() if f.in_watchlist and f.tmdb_id is not None}

    def current_ids(self) -> set[int]:
        """The user's own films plus the current candidates."""
        return self.own_ids() | set(self.candidates)

    def eligible_ids(self) -> set[int]:
        """Films that can be recommended: candidates and watchlist films not yet seen."""
        return self.current_ids() - self.seen_ids()

    # ------------------------------------------------------------ match fixes

    def film(self, film_key: str) -> LibraryFilm:
        try:
            return self.films[film_key]
        except KeyError:
            raise KeyError(film_key) from None

    def set_manual(self, film_key: str, tmdb_id: int, note: str) -> LibraryFilm:
        film = self.film(film_key)
        film.tmdb_id, film.match_status = tmdb_id, matching.MANUAL
        film.match_confidence, film.match_note = 1.0, note
        return film

    def set_status(self, film_key: str, status: str, note: str) -> LibraryFilm:
        film = self.film(film_key)
        film.match_status, film.match_note = status, note
        if status == matching.MANUAL:
            film.match_confidence = 1.0
        elif status == matching.IGNORED:
            film.tmdb_id, film.match_confidence = None, None
        return film


def library_from_export(export: LetterboxdExport, fixes: Mapping[str, Any] | None = None) -> Library:
    """A fresh library for `export`, with the user's saved match fixes applied.

    `fixes` maps film keys to `{"tmdb_id": n}` or `{"ignored": true}`; the browser
    keeps them and sends them with every upload. Fixed films skip matching.
    Fixes for films that aren't in this export are ignored.
    """
    films = {key: LibraryFilm(**asdict(pf)) for key, pf in export.films.items()}
    lib = Library(films)
    for key, fix in (fixes or {}).items():
        if not isinstance(fix, Mapping):
            raise InvalidFixes(f"fix for {key!r} must be an object")
        tmdb_id = fix.get("tmdb_id")
        ignored = fix.get("ignored") is True
        if not ignored and not (isinstance(tmdb_id, int) and not isinstance(tmdb_id, bool) and tmdb_id > 0):
            raise InvalidFixes(f"fix for {key!r} needs a positive tmdb_id or ignored=true")
        if key not in films:
            continue
        if ignored:
            lib.set_status(key, matching.IGNORED, "ignored by you")
        else:
            lib.set_manual(key, tmdb_id, f"set by you (tmdb {tmdb_id})")
    return lib


def applied_fixes(lib: Library) -> int:
    return sum(1 for f in lib.films.values() if f.match_status in (matching.MANUAL, matching.IGNORED))
