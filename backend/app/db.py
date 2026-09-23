"""SQLite engine + table definitions (SQLModel)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Engine, event, inspect, text
from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine

from app.config import get_settings

log = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ApiCache(SQLModel, table=True):
    """Cached external API response, keyed by a hash of namespace + endpoint + params."""

    key: str = Field(primary_key=True)
    namespace: str = Field(index=True)
    endpoint: str
    params_json: str
    status_code: int
    body_json: str
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = Field(default=None, index=True)


class UserFilm(SQLModel, table=True):
    """One film from the user's Letterboxd export (merged across CSVs)."""

    film_key: str = Field(primary_key=True)  # normalized "title|year"
    name: str
    year: int | None = None
    letterboxd_uri: str | None = None
    rating: float | None = None
    watched: bool = False
    watched_date: str | None = None
    diary_entries: int = 0
    rewatch_count: int = 0
    in_watchlist: bool = False
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    review_text: str | None = None
    content_hash: str
    updated_at: datetime = Field(default_factory=utcnow)

    # Filled in by TMDB matching (milestone 2)
    tmdb_id: int | None = Field(default=None, index=True)
    match_confidence: float | None = None
    match_status: str | None = None  # matched | low_confidence | unmatched | manual | error | ignored
    match_note: str | None = None  # human-readable reason, shown in match review


class Movie(SQLModel, table=True):
    """TMDB metadata for any film we know about (user films and candidates)."""

    tmdb_id: int = Field(primary_key=True)
    title: str
    original_title: str | None = None
    year: int | None = Field(default=None, index=True)
    release_date: str | None = None
    overview: str | None = None
    runtime: int | None = None
    original_language: str | None = None
    genres: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    directors: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    cast: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    keywords: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    countries: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    poster_path: str | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    popularity: float | None = None
    imdb_id: str | None = Field(default=None, index=True)
    reviews: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    enriched_at: datetime = Field(default_factory=utcnow)


class IngestRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    status: str = "running"  # running | done | error
    stage: str = "parsing"
    progress_done: int = 0
    progress_total: int = 0
    message: str | None = None
    stats: dict = Field(default_factory=dict, sa_column=Column(JSON))


_engine: Engine | None = None


def make_engine(db_path: Path | str) -> Engine:
    url = "sqlite://" if str(db_path) == ":memory:" else f"sqlite:///{db_path}"
    # timeout: worker threads may briefly contend for SQLite's write lock.
    engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    SQLModel.metadata.create_all(engine)
    _add_missing_columns(engine)
    return engine


def _add_missing_columns(engine: Engine) -> None:
    """Minimal auto-migration: add columns that exist in the models but not the DB.

    Only handles nullable/defaulted additions, which is all the milestones need.
    """
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                col_type = col.type.compile(dialect=engine.dialect)
                default = ""
                if col.default is not None and getattr(col.default, "is_scalar", False):
                    value = col.default.arg
                    default = f" DEFAULT {int(value) if isinstance(value, bool) else repr(value)}"
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col_type}{default}'))
                log.info("migrated: added column %s.%s", table.name, col.name)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine(get_settings().db_path)
    return _engine


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
