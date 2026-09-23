from __future__ import annotations

import csv
import io

from sqlmodel import Session, select

from app.db import UserFilm
from app.ingest import sync_export
from app.letterboxd import make_film_key, parse_export
from tests.fixtures.sample_export import ROOT, build_files, build_zip


def _drop_rows(data: bytes, name: str) -> bytes:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(r for r in rows if r[1] != name)
    return buf.getvalue().encode()


def test_first_sync_adds_everything(session: Session, sample_zip: bytes) -> None:
    result = sync_export(session, parse_export(sample_zip))
    assert result.summary() == {"added": 35, "changed": 0, "unchanged": 0, "removed": 0}
    assert len(session.exec(select(UserFilm)).all()) == 35


def test_resync_same_export_is_noop(session: Session, sample_zip: bytes) -> None:
    sync_export(session, parse_export(sample_zip))
    result = sync_export(session, parse_export(sample_zip))
    assert result.summary() == {"added": 0, "changed": 0, "unchanged": 35, "removed": 0}
    assert result.needs_processing == []


def test_incremental_changes_preserve_match(session: Session, sample_zip: bytes) -> None:
    sync_export(session, parse_export(sample_zip))
    tenet_key = make_film_key("Tenet", 2020)
    tenet = session.get(UserFilm, tenet_key)
    assert tenet is not None
    tenet.tmdb_id, tenet.match_status = 577922, "matched"
    session.add(tenet)
    session.commit()

    files = build_files()
    for name in ("ratings.csv", "watched.csv", "diary.csv", "reviews.csv"):
        path = f"{ROOT}/{name}"
        files[path] = _drop_rows(files[path], "Cats")
    ratings = files[f"{ROOT}/ratings.csv"].decode()
    ratings = ratings.replace("Tenet,2020,https://boxd.it/a23,2.5", "Tenet,2020,https://boxd.it/a23,3.5")
    ratings += "2026-09-01,Anora,2024,https://boxd.it/n01,4.5\n"
    files[f"{ROOT}/ratings.csv"] = ratings.encode()

    result = sync_export(session, parse_export(build_zip(files)))
    assert result.added == [make_film_key("Anora", 2024)]
    assert result.changed == [tenet_key]
    assert result.removed == [make_film_key("Cats", 2019)]
    assert len(result.unchanged) == 33

    session.expire_all()
    tenet = session.get(UserFilm, tenet_key)
    assert tenet is not None
    assert tenet.rating == 3.5
    assert tenet.tmdb_id == 577922 and tenet.match_status == "matched"
