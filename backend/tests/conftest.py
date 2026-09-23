from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session

from app.db import make_engine
from app.ingest import sync_export
from app.letterboxd import parse_export
from tests.fixtures.sample_export import build_files, build_zip


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    return make_engine(tmp_path / "test.sqlite3")


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with Session(engine) as s:
        yield s


@pytest.fixture
def sample_zip() -> bytes:
    return build_zip()


@pytest.fixture
def sample_files() -> dict[str, bytes]:
    return build_files()


@pytest.fixture
def ingested(engine: Engine, sample_zip: bytes) -> Engine:
    """A DB with the sample export synced, before any pipeline run."""
    with Session(engine) as s:
        sync_export(s, parse_export(sample_zip))
    return engine
