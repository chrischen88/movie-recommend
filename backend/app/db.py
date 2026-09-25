"""SQLite engine + table definitions (SQLModel).

Only shared film data lives here: the API response cache and TMDB/OMDb metadata.
Nothing about a user is stored (their library lives in memory: app/library.py).
"""

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


class Movie(SQLModel, table=True):
    """TMDB metadata for every film looked up so far, shared by all users."""

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

    # From OMDb (milestone 6), fetched for the shortlist only; any may be missing.
    imdb_rating: float | None = None  # 0–10
    rt_score: int | None = None  # Rotten Tomatoes Tomatometer, 0–100
    metacritic: int | None = None  # Metascore, 0–100
    omdb_fetched_at: datetime | None = None


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


# Tables and files from before user data moved into memory. They held one
# person's library, so they're removed at startup (see purge_user_data).
LEGACY_USER_TABLES = ("userfilm", "candidate", "appstate", "ingestrun")
LEGACY_USER_FILES = ("taste_model.json", "blend_model.json")


def purge_user_data(engine: Engine, data_dir: Path) -> list[str]:
    """Drop the legacy per-user tables and model files; returns what was removed.

    Idempotent. VACUUMs after dropping, because SQLite keeps a dropped table's
    pages (and so its rows) in the file until then.
    """
    removed = [t for t in LEGACY_USER_TABLES if inspect(engine).has_table(t)]
    if removed:
        with engine.begin() as conn:
            for table in removed:
                conn.execute(text(f'DROP TABLE "{table}"'))
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("VACUUM"))
    for name in LEGACY_USER_FILES:
        path = data_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    if removed:
        log.warning("removed stored user data: %s", ", ".join(removed))
    return removed


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine(get_settings().db_path)
    return _engine


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
