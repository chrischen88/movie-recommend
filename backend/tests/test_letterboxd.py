from __future__ import annotations

import pytest

from app.letterboxd import (
    ExportError,
    make_film_key,
    parse_export,
    parse_rating,
    parse_year,
)
from tests.fixtures.sample_export import ROOT, build_files, build_zip


@pytest.fixture
def export(sample_zip: bytes):
    return parse_export(sample_zip)


def test_counts(export) -> None:
    # 30 watched films + 5 watchlist-only films (Stalker is on both).
    assert len(export.films) == 35
    assert len(export.watched) == 30
    assert len(export.watchlist) == 6
    # 27 from ratings.csv + Uncut Gems via diary fallback.
    assert len(export.rated) == 28
    assert export.files_found == [
        "diary.csv", "ratings.csv", "reviews.csv", "watched.csv", "watchlist.csv"
    ]


def test_decoy_folders_ignored(export) -> None:
    assert make_film_key("Deleted Film", 1999) not in export.films
    assert make_film_key("Liked Film", 1999) not in export.films


def test_bom_and_ratings_source(export) -> None:
    arrival = export.films[make_film_key("Arrival", 2016)]
    assert arrival.rating == 5.0
    assert arrival.rating_source == "ratings"
    # Film URI comes from ratings/watched, not the diary entry URI.
    assert arrival.letterboxd_uri == "https://boxd.it/a01"


def test_diary_merge(export) -> None:
    arrival = export.films[make_film_key("Arrival", 2016)]
    assert arrival.diary_entries == 2
    assert arrival.rewatch_count == 1
    assert arrival.tags == ["sci-fi", "cinema"]
    assert arrival.watched_date == "2024-01-04"
    assert arrival.review_text is not None and "Hannah" in arrival.review_text


def test_diary_rating_fallback(export) -> None:
    gems = export.films[make_film_key("Uncut Gems", 2019)]
    assert gems.rating == 3.0
    assert gems.rating_source == "diary"


def test_invalid_diary_rating_warns_but_keeps_ratings_value(export) -> None:
    heat = export.films[make_film_key("Heat", 1995)]
    assert heat.rating == 4.0
    assert any("Heat" in w and "invalid rating" in w for w in export.warnings)


def test_titles_with_commas_accents_and_no_year(export) -> None:
    assert export.films[make_film_key("Crouching Tiger, Hidden Dragon", 2000)].rating == 4.0
    assert export.films[make_film_key("The Good, the Bad and the Ugly", 1966)].rating == 4.0
    assert export.films[make_film_key("amélie", 2001)].name == "Amélie"
    home = export.films[make_film_key("Home Movie Night", None)]
    assert home.year is None and home.watched and home.rating is None


def test_watchlist_flags(export) -> None:
    stalker = export.films[make_film_key("Stalker", 1979)]
    assert stalker.watched and stalker.in_watchlist
    past_lives = export.films[make_film_key("Past Lives", 2023)]
    assert past_lives.in_watchlist and not past_lives.watched and past_lives.rating is None


def test_multiline_review(export) -> None:
    cats = export.films[make_film_key("Cats", 2019)]
    assert cats.review_text == "Why.\nJust why."


def test_unrated_watched(export) -> None:
    paddington = export.films[make_film_key("Paddington 2", 2017)]
    assert paddington.watched and paddington.rating is None
    assert paddington.review_text and "kind and polite" in paddington.review_text


def test_missing_files_ok(sample_files: dict[str, bytes]) -> None:
    only_watched = {k: v for k, v in sample_files.items() if k.endswith("/watched.csv")}
    export = parse_export(build_zip(only_watched))
    assert len(export.films) == 30
    assert export.rated == []
    assert export.files_found == ["watched.csv"]


def test_files_at_zip_root() -> None:
    files = {"ratings.csv": b"Date,Name,Year,Letterboxd URI,Rating\n2024-01-01,Heat,1995,u,4\n"}
    export = parse_export(build_zip(files))
    assert export.films[make_film_key("Heat", 1995)].rating == 4.0


def test_missing_name_column_warns() -> None:
    files = {
        "ratings.csv": b"Date,Title,Year,Rating\n2024-01-01,Heat,1995,4\n",
        "watched.csv": b"Date,Name,Year,Letterboxd URI\n2024-01-01,Heat,1995,u\n",
    }
    export = parse_export(build_zip(files))
    assert len(export.films) == 1
    assert any("ratings.csv" in w and "Name" in w for w in export.warnings)


def test_malformed_line_warns() -> None:
    files = {"watched.csv": b"Date,Name,Year,Letterboxd URI\n2024-01-01,Heat,1995,u\n1,2,3,4,5,6\n"}
    export = parse_export(build_zip(files))
    assert len(export.films) == 1
    assert any("malformed" in w for w in export.warnings)


def test_empty_file_warns() -> None:
    files = {"watched.csv": b"", "ratings.csv": b"Date,Name,Year,Letterboxd URI,Rating\n"}
    export = parse_export(build_zip(files))
    assert export.films == {}
    assert any("watched.csv" in w and "empty" in w for w in export.warnings)


def test_not_a_zip() -> None:
    with pytest.raises(ExportError, match="not a valid ZIP"):
        parse_export(b"definitely not a zip")


def test_zip_without_csvs() -> None:
    with pytest.raises(ExportError, match="no Letterboxd CSVs"):
        parse_export(build_zip({f"{ROOT}/readme.txt": b"hi"}))


@pytest.mark.parametrize(
    ("raw", "expected"), [("3.5", 3.5), ("5", 5.0), ("0.5", 0.5), ("", None), (" 4.0 ", 4.0)]
)
def test_parse_rating_valid(raw: str, expected: float | None) -> None:
    assert parse_rating(raw) == expected


@pytest.mark.parametrize("raw", ["6", "0", "3.3", "abc", "-1"])
def test_parse_rating_invalid(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_rating(raw)


@pytest.mark.parametrize(
    ("raw", "expected"), [("2016", 2016), ("", None), ("abcd", None), ("1066", None), ("2016.0", 2016)]
)
def test_parse_year(raw: str, expected: int | None) -> None:
    assert parse_year(raw) == expected


def test_film_key_normalization() -> None:
    assert make_film_key("  The  Room ", 2003) == make_film_key("the room", 2003)
    assert make_film_key("Heat", None) == "heat|"


def test_content_hash_tracks_rating(export) -> None:
    film = export.films[make_film_key("Tenet", 2020)]
    before = film.content_hash()
    film.rating = 3.0
    assert film.content_hash() != before


def test_username_from_profile(sample_zip: bytes) -> None:
    export = parse_export(sample_zip)
    assert export.username == "sampleuser"
    assert "profile.csv" not in export.files_found  # identifies the account; not a film source


def test_username_is_normalized() -> None:
    assert parse_export(build_zip(build_files(username="  SampleUser "))).username == "sampleuser"


def test_username_missing() -> None:
    files = build_files()
    del files[f"{ROOT}/profile.csv"]
    assert parse_export(build_zip(files)).username is None
    files[f"{ROOT}/profile.csv"] = b"Date Joined,Given Name\n2020-01-01,Sam\n"
    assert parse_export(build_zip(files)).username is None
