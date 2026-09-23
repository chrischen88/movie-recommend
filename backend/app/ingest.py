"""Incremental sync of a parsed Letterboxd export into the local database."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlmodel import Session, delete, select

from app.db import AppState, Candidate, UserFilm, utcnow
from app.letterboxd import LetterboxdExport, ParsedFilm

log = logging.getLogger(__name__)

ACCOUNT_KEY = "letterboxd_username"


@dataclass
class SyncResult:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    # True when the export is from a different account (or the DB predates
    # account tracking): everything was replaced rather than merged.
    reset: bool = False
    account: str | None = None

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
            "reset": self.reset,
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


def current_account(session: Session) -> str | None:
    state = session.get(AppState, ACCOUNT_KEY)
    return state.value if state else None


def sync_export(session: Session, export: LetterboxdExport) -> SyncResult:
    """Make the DB hold exactly `export`'s films.

    Same account (a newer export): incremental. Upsert changed films, delete
    missing ones, and keep existing TMDB match fields, including manual fixes
    (the film key encodes title+year, so an unchanged key means the match is
    still valid).

    Different account: nothing carries over. All user films and candidates are
    deleted first, so matching, candidates and the taste model are rebuilt from
    scratch (cheaply: TMDB responses and film metadata are cached). A DB with
    films but no recorded account predates account tracking and is reset too.
    An export without profile.csv can't be identified and is treated as the
    same account.
    """
    result = SyncResult(account=export.username)
    existing = {row.film_key: row for row in session.exec(select(UserFilm)).all()}
    previous = current_account(session)
    if export.username is not None and existing and previous != export.username:
        log.warning(
            "export is from %r but the loaded data is %s: replacing everything",
            export.username, repr(previous) if previous else "from an unknown account",
        )
        result.reset = True
        result.removed = sorted(existing)
        session.exec(delete(UserFilm))  # type: ignore[call-overload]
        session.exec(delete(Candidate))  # type: ignore[call-overload]
        existing = {}
    elif export.username is None and previous is not None:
        log.warning("export has no profile.csv: assuming it's %r's and merging", previous)

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

    if export.username is not None:
        state = session.get(AppState, ACCOUNT_KEY) or AppState(key=ACCOUNT_KEY, value=export.username)
        state.value, state.updated_at = export.username, utcnow()
        session.add(state)
    session.commit()
    log.info("sync complete: %s", result.summary())
    return result
