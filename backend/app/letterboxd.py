"""Parse a Letterboxd data-export ZIP into one merged record per film.

Notes on the real export format:
  * Files usually live under a top-level folder (`letterboxd-<user>-<date>-utc/`)
    alongside `deleted/`, `orphaned/`, `likes/` and `lists/` sub-folders that
    reuse the same file names; we only read the shallowest non-excluded copy.
  * The `Letterboxd URI` in diary.csv / reviews.csv points at the *log entry*,
    not the film, so films are merged on a normalized (title, year) key.
  * ratings.csv holds the current rating; diary ratings are per-viewing and
    are only used as a fallback.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath

import pandas as pd

log = logging.getLogger(__name__)

EXPECTED_FILES = ("ratings.csv", "watched.csv", "diary.csv", "watchlist.csv", "reviews.csv")
# Identifies the account, so a different person's export replaces the data
# instead of being merged into it. Only the Username column is read.
PROFILE_FILE = "profile.csv"
EXCLUDED_DIRS = {"deleted", "orphaned", "likes", "lists", "__macosx"}


class ExportError(ValueError):
    """The upload is not a usable Letterboxd export."""


@dataclass
class ParsedFilm:
    film_key: str
    name: str
    year: int | None
    letterboxd_uri: str | None = None
    rating: float | None = None
    rating_source: str | None = None  # ratings | diary | reviews
    watched: bool = False
    watched_date: str | None = None
    diary_entries: int = 0
    rewatch_count: int = 0
    in_watchlist: bool = False
    tags: list[str] = field(default_factory=list)
    review_text: str | None = None

    def content_hash(self) -> str:
        payload = {k: v for k, v in asdict(self).items() if k != "rating_source"}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class LetterboxdExport:
    films: dict[str, ParsedFilm]
    files_found: list[str]
    warnings: list[str]
    username: str | None = None  # from profile.csv; None if the export has none

    @property
    def rated(self) -> list[ParsedFilm]:
        return [f for f in self.films.values() if f.rating is not None]

    @property
    def watched(self) -> list[ParsedFilm]:
        return [f for f in self.films.values() if f.watched]

    @property
    def watchlist(self) -> list[ParsedFilm]:
        return [f for f in self.films.values() if f.in_watchlist]


# ---------------------------------------------------------------- helpers


def normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKC", title).casefold().strip()
    return re.sub(r"\s+", " ", t)


def make_film_key(name: str, year: int | None) -> str:
    return f"{normalize_title(name)}|{year if year is not None else ''}"


def parse_year(raw: str) -> int | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        year = int(float(raw))
    except ValueError:
        return None
    return year if 1870 <= year <= 2100 else None


def parse_rating(raw: str) -> float | None:
    """Parse a 0.5–5.0 half-star rating. Raises ValueError on garbage."""
    raw = raw.strip()
    if not raw:
        return None
    value = float(raw)
    if not 0.5 <= value <= 5.0:
        raise ValueError(f"rating {value} outside 0.5–5.0")
    rounded = round(value * 2) / 2
    if abs(rounded - value) > 1e-9:
        raise ValueError(f"rating {value} is not a half-star step")
    return rounded


def parse_date(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    return date.fromisoformat(raw[:10]).isoformat()


def _cell(row: pd.Series, col: str) -> str:
    value = row.get(col, "")
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


# ---------------------------------------------------------------- zip handling


def _locate_files(
    zf: zipfile.ZipFile, names: tuple[str, ...] = EXPECTED_FILES
) -> dict[str, zipfile.ZipInfo]:
    found: dict[str, zipfile.ZipInfo] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        name = path.name.lower()
        if name not in names:
            continue
        if any(part.lower() in EXCLUDED_DIRS for part in path.parts[:-1]):
            continue
        current = found.get(name)
        if current is None or len(path.parts) < len(PurePosixPath(current.filename).parts):
            found[name] = info
    return found


def _read_csv(raw: bytes, filename: str, warnings: list[str], required: str = "Name") -> pd.DataFrame:
    def warn(msg: str) -> None:
        log.warning(msg)
        warnings.append(msg)

    def bad_line(fields: list[str]) -> None:
        warn(f"{filename}: malformed line skipped: {fields!r:.120}")
        return None

    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            engine="python",
            on_bad_lines=bad_line,
        )
    except pd.errors.EmptyDataError:
        warn(f"{filename}: file is empty")
        return pd.DataFrame()
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        warn(f"{filename}: could not parse ({exc})")
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    if required not in df.columns:
        warn(f"{filename}: missing required '{required}' column; skipped")
        return pd.DataFrame()
    return df


# ---------------------------------------------------------------- merging


class _Merger:
    def __init__(self) -> None:
        self.films: dict[str, ParsedFilm] = {}
        self.warnings: list[str] = []
        # (watched_date_or_date, rating) of the latest diary rating per film
        self._diary_rating: dict[str, tuple[str, float]] = {}
        self._reviews: dict[str, list[str]] = {}

    def warn(self, msg: str) -> None:
        log.warning(msg)
        self.warnings.append(msg)

    def film_for(self, row: pd.Series, filename: str, line: int) -> ParsedFilm | None:
        name = _cell(row, "Name").strip()
        if not name:
            self.warn(f"{filename} row {line}: row has no film name; skipped")
            return None
        raw_year = _cell(row, "Year")
        year = parse_year(raw_year)
        if year is None and raw_year.strip():
            self.warn(f"{filename} row {line}: '{name}' has invalid year {raw_year!r}; ignoring year")
        key = make_film_key(name, year)
        film = self.films.get(key)
        if film is None:
            film = self.films[key] = ParsedFilm(film_key=key, name=name, year=year)
        return film

    def rating_or_warn(self, row: pd.Series, filename: str, line: int, name: str) -> float | None:
        raw = _cell(row, "Rating")
        try:
            return parse_rating(raw)
        except ValueError as exc:
            self.warn(f"{filename} row {line}: '{name}' has invalid rating {raw!r} ({exc}); ignored")
            return None

    def date_or_warn(self, row: pd.Series, col: str, filename: str, line: int) -> str | None:
        raw = _cell(row, col)
        try:
            return parse_date(raw)
        except ValueError:
            self.warn(f"{filename} row {line}: invalid {col} {raw!r}; ignored")
            return None

    # -- per-file handlers (row numbers are 1-based data rows, excluding header) --

    def add_watched(self, df: pd.DataFrame) -> None:
        for i, row in df.iterrows():
            film = self.film_for(row, "watched.csv", int(i) + 1)
            if film:
                film.watched = True
                film.letterboxd_uri = film.letterboxd_uri or _cell(row, "Letterboxd URI") or None

    def add_ratings(self, df: pd.DataFrame) -> None:
        for i, row in df.iterrows():
            line = int(i) + 1
            film = self.film_for(row, "ratings.csv", line)
            if not film:
                continue
            film.watched = True
            film.letterboxd_uri = _cell(row, "Letterboxd URI") or film.letterboxd_uri
            rating = self.rating_or_warn(row, "ratings.csv", line, film.name)
            if rating is not None:
                film.rating, film.rating_source = rating, "ratings"

    def add_diary(self, df: pd.DataFrame) -> None:
        for i, row in df.iterrows():
            line = int(i) + 1
            film = self.film_for(row, "diary.csv", line)
            if not film:
                continue
            film.watched = True
            film.diary_entries += 1
            if _cell(row, "Rewatch").strip().lower() in {"yes", "true", "1"}:
                film.rewatch_count += 1
            when = self.date_or_warn(row, "Watched Date", "diary.csv", line) or self.date_or_warn(
                row, "Date", "diary.csv", line
            )
            if when and (film.watched_date is None or when > film.watched_date):
                film.watched_date = when
            for tag in _cell(row, "Tags").split(","):
                tag = tag.strip()
                if tag and tag not in film.tags:
                    film.tags.append(tag)
            rating = self.rating_or_warn(row, "diary.csv", line, film.name)
            if rating is not None:
                prev = self._diary_rating.get(film.film_key)
                sort_key = when or ""
                if prev is None or sort_key >= prev[0]:
                    self._diary_rating[film.film_key] = (sort_key, rating)

    def add_watchlist(self, df: pd.DataFrame) -> None:
        for i, row in df.iterrows():
            film = self.film_for(row, "watchlist.csv", int(i) + 1)
            if film:
                film.in_watchlist = True
                film.letterboxd_uri = film.letterboxd_uri or _cell(row, "Letterboxd URI") or None

    def add_reviews(self, df: pd.DataFrame) -> None:
        if "Review" not in df.columns:
            self.warn("reviews.csv: no 'Review' column; skipped")
            return
        for i, row in df.iterrows():
            line = int(i) + 1
            film = self.film_for(row, "reviews.csv", line)
            if not film:
                continue
            film.watched = True
            text = _cell(row, "Review").strip()
            if text:
                self._reviews.setdefault(film.film_key, []).append(text)
            if film.rating is None and film.film_key not in self._diary_rating:
                rating = self.rating_or_warn(row, "reviews.csv", line, film.name)
                if rating is not None:
                    film.rating, film.rating_source = rating, "reviews"

    def finalize(self) -> None:
        for key, (_, rating) in self._diary_rating.items():
            film = self.films[key]
            if film.rating is None or film.rating_source == "reviews":
                film.rating, film.rating_source = rating, "diary"
        for key, texts in self._reviews.items():
            self.films[key].review_text = "\n\n".join(texts)


def _read_username(zf: zipfile.ZipFile, max_csv_bytes: int, warnings: list[str]) -> str | None:
    info = _locate_files(zf, (PROFILE_FILE,)).get(PROFILE_FILE)
    if info is None or info.file_size > max_csv_bytes:
        log.info("export has no usable %s: account can't be identified", PROFILE_FILE)
        return None
    df = _read_csv(zf.read(info), PROFILE_FILE, warnings, required="Username")
    if df.empty:
        return None  # _read_csv already warned
    name = _cell(df.iloc[0], "Username").strip().lower()
    return name or None


def parse_export(source: bytes | str | Path, *, max_csv_bytes: int = 100 * 1024 * 1024) -> LetterboxdExport:
    """Parse a Letterboxd export ZIP (bytes or path)."""
    buf = io.BytesIO(source) if isinstance(source, bytes) else open(source, "rb")
    try:
        try:
            zf = zipfile.ZipFile(buf)
        except zipfile.BadZipFile as exc:
            raise ExportError("upload is not a valid ZIP file") from exc
        with zf:
            located = _locate_files(zf)
            if not located:
                raise ExportError(
                    "no Letterboxd CSVs found (expected e.g. ratings.csv, watched.csv, diary.csv)"
                )
            merger = _Merger()
            for missing in sorted(set(EXPECTED_FILES) - located.keys()):
                log.info("export has no %s; continuing without it", missing)

            frames: dict[str, pd.DataFrame] = {}
            for name, info in located.items():
                if info.file_size > max_csv_bytes:
                    raise ExportError(f"{name} is too large ({info.file_size} bytes)")
                frames[name] = _read_csv(zf.read(info), name, merger.warnings)
            username = _read_username(zf, max_csv_bytes, merger.warnings)
    finally:
        buf.close()

    # Order matters: watched/ratings establish film URIs before diary/reviews.
    handlers = [
        ("watched.csv", merger.add_watched),
        ("ratings.csv", merger.add_ratings),
        ("diary.csv", merger.add_diary),
        ("watchlist.csv", merger.add_watchlist),
        ("reviews.csv", merger.add_reviews),
    ]
    for name, handler in handlers:
        df = frames.get(name)
        if df is not None and not df.empty:
            handler(df)
    merger.finalize()

    export = LetterboxdExport(
        films=merger.films, files_found=sorted(located), warnings=merger.warnings, username=username
    )
    log.info(
        "parsed export: %d films (%d rated, %d watched, %d watchlist), %d warnings",
        len(export.films), len(export.rated), len(export.watched),
        len(export.watchlist), len(export.warnings),
    )
    return export
