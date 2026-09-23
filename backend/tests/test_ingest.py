from __future__ import annotations

import csv
import io

from sqlmodel import Session, select

from app.db import Candidate, UserFilm
from app.ingest import current_account, sync_export
from app.letterboxd import make_film_key, parse_export
from tests.fixtures.sample_export import ROOT, WATCHED, build_files, build_zip


def _drop_rows(data: bytes, name: str) -> bytes:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(r for r in rows if r[1] != name)
    return buf.getvalue().encode()


def test_first_sync_adds_everything(session: Session, sample_zip: bytes) -> None:
    result = sync_export(session, parse_export(sample_zip))
    assert result.summary() == {"added": 35, "changed": 0, "unchanged": 0, "removed": 0, "reset": False}
    assert len(session.exec(select(UserFilm)).all()) == 35


def test_resync_same_export_is_noop(session: Session, sample_zip: bytes) -> None:
    sync_export(session, parse_export(sample_zip))
    result = sync_export(session, parse_export(sample_zip))
    assert result.summary() == {"added": 0, "changed": 0, "unchanged": 35, "removed": 0, "reset": False}
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


# ---------------------------------------------------------------- accounts

OTHER_WATCHED = WATCHED[:15]


def other_account_zip() -> bytes:
    return build_zip(build_files(username="someoneelse", watched=OTHER_WATCHED))


def _set_manual(session: Session, key: str, tmdb_id: int) -> None:
    film = session.get(UserFilm, key)
    assert film is not None
    film.tmdb_id, film.match_status, film.match_confidence = tmdb_id, "manual", 1.0
    session.add(film)
    session.commit()


def test_same_account_keeps_manual_fixes(session: Session, sample_zip: bytes) -> None:
    sync_export(session, parse_export(sample_zip))
    heat = make_film_key("Heat", 1995)
    _set_manual(session, heat, 949)
    files = build_files()
    for name in ("watched.csv", "ratings.csv", "diary.csv", "reviews.csv"):
        files[f"{ROOT}/{name}"] = _drop_rows(files[f"{ROOT}/{name}"], "Cats")
    result = sync_export(session, parse_export(build_zip(files)))
    assert not result.reset and result.removed == [make_film_key("Cats", 2019)]
    kept = session.get(UserFilm, heat)
    assert kept is not None and kept.match_status == "manual" and kept.tmdb_id == 949
    assert current_account(session) == "sampleuser"


def test_other_account_replaces_everything(session: Session, sample_zip: bytes) -> None:
    sync_export(session, parse_export(sample_zip))
    arrival = make_film_key("Arrival", 2016)  # in both exports
    _set_manual(session, arrival, 12345)
    session.add(Candidate(tmdb_id=777, sources=["similar:12345"]))
    session.commit()

    other = parse_export(other_account_zip())
    result = sync_export(session, other)
    assert result.reset and result.account == "someoneelse"
    assert result.summary()["removed"] == 35 and result.summary()["added"] == len(other.films)
    assert {f.film_key for f in session.exec(select(UserFilm))} == set(other.films)
    fresh = session.get(UserFilm, arrival)
    assert fresh is not None and fresh.tmdb_id is None and fresh.match_status is None  # fix not carried over
    assert fresh.rating == 5.0
    assert session.exec(select(Candidate)).all() == []
    assert current_account(session) == "someoneelse"
    assert sorted(result.needs_processing) == sorted(other.films)


def test_db_from_before_account_tracking_is_reset(session: Session) -> None:
    files = build_files()
    del files[f"{ROOT}/profile.csv"]
    sync_export(session, parse_export(build_zip(files)))  # no account recorded
    assert current_account(session) is None
    result = sync_export(session, parse_export(build_zip(build_files())))
    assert result.reset and current_account(session) == "sampleuser"


def test_export_without_profile_merges(session: Session, sample_zip: bytes) -> None:
    sync_export(session, parse_export(sample_zip))
    heat = make_film_key("Heat", 1995)
    _set_manual(session, heat, 949)
    files = build_files()
    del files[f"{ROOT}/profile.csv"]
    result = sync_export(session, parse_export(build_zip(files)))
    assert not result.reset
    kept = session.get(UserFilm, heat)
    assert kept is not None and kept.match_status == "manual"
    assert current_account(session) == "sampleuser"
