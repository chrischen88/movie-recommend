"""Incremental sync of a parsed Letterboxd export into the local database."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlmodel import Session, select

from app.db import UserFilm, utcnow
from app.letterboxd import LetterboxdExport, ParsedFilm

log = logging.getLogger(__name__)


@dataclass
class SyncResult:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def needs_processing(self) -> list[str]:
        """Film keys that downstream stages (matching, embedding, …) must (re)process."""
        return self.added + self.changed

    def summary(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "changed": len(self.changed),
            "unchanged": len(self.unchanged),
            "removed": len(self.removed),
        }


def _apply(row: UserFilm, film: ParsedFilm, content_hash: str) -> None:
    row.name = film.name
    row.year = film.year
    row.letterboxd_uri = film.letterboxd_uri
    row.rating = film.rating
    row.watched = film.watched
    row.watched_date = film.watched_date
    row.diary_entries = film.diary_entries
    row.rewatch_count = film.rewatch_count
    row.in_watchlist = film.in_watchlist
    row.tags = list(film.tags)
    row.review_text = film.review_text
    row.content_hash = content_hash
    row.updated_at = utcnow()


def sync_export(session: Session, export: LetterboxdExport) -> SyncResult:
    """Upsert films from `export`; delete films no longer present.

    Existing TMDB match fields are preserved (the film key encodes title+year,
    so an unchanged key means the match is still valid).
    """
    result = SyncResult()
    existing = {row.film_key: row for row in session.exec(select(UserFilm)).all()}

    for key, film in export.films.items():
        content_hash = film.content_hash()
        row = existing.pop(key, None)
        if row is None:
            row = UserFilm(film_key=key, name=film.name, content_hash=content_hash)
            _apply(row, film, content_hash)
            session.add(row)
            result.added.append(key)
        elif row.content_hash != content_hash:
            _apply(row, film, content_hash)
            session.add(row)
            result.changed.append(key)
        else:
            result.unchanged.append(key)

    for key, row in existing.items():
        session.delete(row)
        result.removed.append(key)

    session.commit()
    log.info("sync complete: %s", result.summary())
    return result
