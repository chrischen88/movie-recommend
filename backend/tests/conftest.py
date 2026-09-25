from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlmodel import Session

from app.db import make_engine
from app.letterboxd import parse_export
from app.library import Library, library_from_export
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
def library(sample_zip: bytes) -> Library:
    """The sample export as a fresh in-memory library, before any pipeline run."""
    return library_from_export(parse_export(sample_zip))
