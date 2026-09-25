"""Match Letterboxd films (title + year) to TMDB ids with a confidence score."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol

from app.tmdb import year_of

log = logging.getLogger(__name__)

MATCHED = "matched"
LOW_CONFIDENCE = "low_confidence"
UNMATCHED = "unmatched"
MANUAL = "manual"
ERROR = "error"
IGNORED = "ignored"  # user said "not a real film / skip it"
NEEDS_REVIEW = (LOW_CONFIDENCE, UNMATCHED, ERROR)

# Score multiplier by |candidate year - letterboxd year|.
YEAR_FACTOR = {0: 1.0, 1: 0.9, 2: 0.6}
YEAR_FACTOR_FAR = 0.3
YEAR_FACTOR_UNKNOWN = 0.8
ARTICLE_ONLY_MATCH = 0.95
AMBIGUITY_MARGIN = 0.03
# A near-tie is only "ambiguous" if the runner-up has ≥10% of the winner's votes.
AMBIGUITY_VOTE_RATIO = 0.1
AMBIGUITY_PENALTY = 0.7


class Searcher(Protocol):
    def search_movie(self, query: str, year: int | None = None) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class MatchResult:
    tmdb_id: int | None
    confidence: float
    status: str
    note: str


_ARTICLES = re.compile(r"^(the|a|an|le|la|les|el|il|der|die|das)\s+")


def normalize_for_match(title: str, strip_articles: bool = True) -> str:
    t = unicodedata.normalize("NFKD", title)
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).casefold()
    t = t.replace("&", " and ")
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return _ARTICLES.sub("", t) if strip_articles else t


def title_similarity(a: str, b: str) -> float:
    if normalize_for_match(a, strip_articles=False) == normalize_for_match(b, strip_articles=False):
        return 1.0
    na, nb = normalize_for_match(a), normalize_for_match(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return ARTICLE_ONLY_MATCH  # e.g. "The Room" vs "Room"
    return SequenceMatcher(None, na, nb).ratio()


def year_factor(wanted: int | None, candidate: int | None) -> float:
    if wanted is None or candidate is None:
        return YEAR_FACTOR_UNKNOWN
    return YEAR_FACTOR.get(abs(wanted - candidate), YEAR_FACTOR_FAR)


def score_candidate(name: str, year: int | None, cand: dict[str, Any]) -> float:
    titles = [t for t in (cand.get("title"), cand.get("original_title")) if t]
    sim = max((title_similarity(name, t) for t in titles), default=0.0)
    return sim * year_factor(year, year_of(cand.get("release_date")))


def _search_plan(year: int | None) -> list[int | None]:
    if year is None:
        return [None]
    return [year, year - 1, year + 1, None]


def match_film(
    searcher: Searcher, name: str, year: int | None, low_threshold: float
) -> MatchResult:
    """Search TMDB (year, then ±1, then no year) and pick the best candidate."""
    results: list[dict[str, Any]] = []
    used_year: int | None = None
    for attempt_year in _search_plan(year):
        results = searcher.search_movie(name, attempt_year)
        if results:
            used_year = attempt_year
            break

    if not results:
        log.debug("no TMDB match for %r (%s)", name, year)
        return MatchResult(None, 0.0, UNMATCHED, "no TMDB search results")

    scored = sorted(
        ((score_candidate(name, year, c), c) for c in results),
        key=lambda sc: sc[0],
        reverse=True,
    )
    top_score = scored[0][0]
    # Among near-ties, prefer the most-voted film (the "real" one, not a same-name short).
    near = sorted(
        (c for s, c in scored if top_score - s < AMBIGUITY_MARGIN),
        key=lambda c: c.get("vote_count") or 0,
        reverse=True,
    )
    best = near[0]
    confidence = score_candidate(name, year, best)
    note = f"best: {best.get('title')!r} ({year_of(best.get('release_date')) or '?'})"

    if len(near) > 1:
        runner_up = near[1]
        best_votes = best.get("vote_count") or 0
        if (runner_up.get("vote_count") or 0) >= AMBIGUITY_VOTE_RATIO * best_votes:
            confidence *= AMBIGUITY_PENALTY
            note += f"; ambiguous with {runner_up.get('title')!r} (tmdb {runner_up.get('id')})"

    if used_year != year:
        note += f"; found via search year {used_year if used_year is not None else 'none'}"

    status = MATCHED if confidence >= low_threshold else LOW_CONFIDENCE
    if status == LOW_CONFIDENCE:
        log.debug("low-confidence match for %r (%s): %.2f, %s", name, year, confidence, note)
    return MatchResult(int(best["id"]), round(confidence, 3), status, note)


_TMDB_URL = re.compile(r"themoviedb\.org/movie/(\d+)")


def parse_tmdb_ref(ref: str | int) -> int | None:
    """Accept a bare TMDB id or a themoviedb.org/movie/<id>-slug URL."""
    if isinstance(ref, int):
        return ref if ref > 0 else None
    ref = ref.strip()
    if ref.isdigit():
        return int(ref) or None
    m = _TMDB_URL.search(ref)
    return int(m.group(1)) if m else None
