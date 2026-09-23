"""Background ingestion pipeline: TMDB matching → enrichment (→ later stages).

Incremental by construction:
  * matching only touches new films (`match_status` None) and earlier TMDB
    errors, so manual fixes and settled results are never redone;
  * enrichment only fetches TMDB ids that have no `Movie` row yet;
  * every HTTP response is cached, so even a forced redo is network-free.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import TypeVar

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from app import matching
from app.cache import ApiError, ResponseCache
from app.config import Settings, get_settings
from app.db import IngestRun, Movie, UserFilm, get_engine, utcnow
from app.tmdb import TmdbClient, fetch_movie

log = logging.getLogger(__name__)

STAGES = ["parsing", "matching", "enrichment"]
PROGRESS_FLUSH_SECONDS = 0.5

T = TypeVar("T")
R = TypeVar("R")


class TmdbUnavailable(RuntimeError):
    pass


class Pipeline:
    def __init__(
        self,
        engine: Engine,
        settings: Settings,
        tmdb: TmdbClient | None,
        max_workers: int = 8,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.tmdb = tmdb
        self.max_workers = max_workers
        self._lock = threading.Lock()

    # ------------------------------------------------------------ run control

    def try_acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        self._lock.release()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def run(self, run_id: int) -> None:
        """Run all post-parse stages for `run_id`. Caller must hold the lock."""
        try:
            self._run(run_id)
        except Exception as exc:  # noqa: BLE001 — record any failure on the run
            log.exception("pipeline run %s failed", run_id)
            self._update(run_id, status="error", message=str(exc), finished_at=utcnow())
        finally:
            self.release()

    def _run(self, run_id: int) -> None:
        if self.tmdb is None:
            msg = "TMDB_API_KEY not set: matching and enrichment skipped"
            log.warning(msg)
            self._update(run_id, stage="done", status="done", message=msg, finished_at=utcnow())
            return

        match_stats = self.match_pending(run_id)
        enrich_stats = self.enrich_pending(run_id)

        with Session(self.engine) as s:
            run = s.get(IngestRun, run_id)
            assert run is not None
            run.stats = {**run.stats, "matching": match_stats, "enrichment": enrich_stats}
            run.stage, run.status, run.finished_at = "done", "done", utcnow()
            s.add(run)
            s.commit()

    # ------------------------------------------------------------ helpers

    def _update(self, run_id: int, **fields: object) -> None:
        with Session(self.engine) as s:
            run = s.get(IngestRun, run_id)
            if run is None:
                return
            for k, v in fields.items():
                setattr(run, k, v)
            s.add(run)
            s.commit()

    def _parallel(
        self,
        run_id: int,
        stage: str,
        items: list[T],
        work: Callable[[T], R],
        on_done: Callable[[Session, T, R | None, Exception | None], None],
    ) -> None:
        """Run `work` over items in a thread pool; apply results on this thread."""
        self._update(run_id, stage=stage, progress_done=0, progress_total=len(items))
        if not items:
            return
        last_flush = time.monotonic()
        with ThreadPoolExecutor(self.max_workers) as pool, Session(self.engine) as s:
            futures: dict[Future[R], T] = {pool.submit(work, it): it for it in items}
            for done, fut in enumerate(as_completed(futures), start=1):
                item = futures[fut]
                exc = fut.exception()
                if isinstance(exc, ApiError) and exc.status_code == 401:
                    for f in futures:
                        f.cancel()
                    raise TmdbUnavailable("TMDB rejected the API key (HTTP 401)") from exc
                on_done(s, item, None if exc else fut.result(), exc)
                # Commit per item: an open write transaction here would hold
                # SQLite's lock and block workers writing to the HTTP cache.
                s.commit()
                if done == len(items) or time.monotonic() - last_flush > PROGRESS_FLUSH_SECONDS:
                    self._update(run_id, progress_done=done)
                    last_flush = time.monotonic()

    # ------------------------------------------------------------ stages

    def match_pending(self, run_id: int) -> dict[str, int]:
        assert self.tmdb is not None
        tmdb = self.tmdb
        threshold = self.settings.match_low_confidence_threshold
        with Session(self.engine) as s:
            pending = [
                (f.film_key, f.name, f.year)
                for f in s.exec(
                    select(UserFilm).where(
                        col(UserFilm.match_status).is_(None)
                        | (col(UserFilm.match_status) == matching.ERROR)
                    )
                )
            ]
        outcomes = (matching.MATCHED, matching.LOW_CONFIDENCE, matching.UNMATCHED, matching.ERROR)
        stats = dict.fromkeys(outcomes, 0)

        def work(item: tuple[str, str, int | None]) -> matching.MatchResult:
            _, name, year = item
            return matching.match_film(tmdb, name, year, threshold)

        def on_done(
            s: Session,
            item: tuple[str, str, int | None],
            result: matching.MatchResult | None,
            exc: Exception | None,
        ) -> None:
            film = s.get(UserFilm, item[0])
            if film is None:
                return
            if result is None:
                log.error("matching %r failed: %s", film.name, exc)
                film.match_status, film.match_note = matching.ERROR, f"TMDB error: {exc}"
            else:
                film.tmdb_id = result.tmdb_id
                film.match_confidence = result.confidence
                film.match_status = result.status
                film.match_note = result.note
            stats[film.match_status] += 1
            s.add(film)

        self._parallel(run_id, "matching", pending, work, on_done)
        log.info("matching done: %s", stats)
        return stats

    def enrich_pending(self, run_id: int) -> dict[str, int]:
        assert self.tmdb is not None
        tmdb = self.tmdb
        with Session(self.engine) as s:
            wanted = set(
                s.exec(select(UserFilm.tmdb_id).where(col(UserFilm.tmdb_id).is_not(None))).all()
            )
            have = set(s.exec(select(Movie.tmdb_id)).all())
        pending = sorted(i for i in wanted - have if i is not None)
        stats = {"enriched": 0, "not_found": 0, "errors": 0, "already_had": len(wanted & have)}

        def on_done(s: Session, tmdb_id: int, movie: Movie | None, exc: Exception | None) -> None:
            if movie is not None:
                s.merge(movie)
                stats["enriched"] += 1
                return
            stats["errors" if exc else "not_found"] += 1
            note = f"enrichment failed: {exc}" if exc else f"TMDB id {tmdb_id} not found"
            log.error("tmdb %s: %s", tmdb_id, note)
            for film in s.exec(select(UserFilm).where(UserFilm.tmdb_id == tmdb_id)):
                if film.match_status != matching.MANUAL:
                    film.match_status, film.match_note = matching.ERROR, note
                    s.add(film)

        self._parallel(run_id, "enrichment", pending, lambda i: fetch_movie(tmdb, i), on_done)
        log.info("enrichment done: %s", stats)
        return stats

    # ------------------------------------------------------------ manual fixes

    def set_manual_match(self, film_key: str, tmdb_id: int) -> tuple[UserFilm, Movie]:
        if self.tmdb is None:
            raise TmdbUnavailable("TMDB_API_KEY not set")
        with Session(self.engine) as s:
            film = s.get(UserFilm, film_key)
            if film is None:
                raise KeyError(film_key)
            movie = s.get(Movie, tmdb_id) or fetch_movie(self.tmdb, tmdb_id)
            if movie is None:
                raise LookupError(f"TMDB id {tmdb_id} not found")
            movie = s.merge(movie)
            film.tmdb_id = tmdb_id
            film.match_status = matching.MANUAL
            film.match_confidence = 1.0
            film.match_note = f"manually set to {movie.title!r} ({movie.year or '?'})"
            s.add(film)
            s.commit()
            s.refresh(film)
            s.refresh(movie)
            log.info("manual match: %s -> %s", film_key, tmdb_id)
            return film, movie

    def set_status(self, film_key: str, status: str, note: str) -> UserFilm:
        with Session(self.engine) as s:
            film = s.get(UserFilm, film_key)
            if film is None:
                raise KeyError(film_key)
            film.match_status, film.match_note = status, note
            if status == matching.MANUAL:
                film.match_confidence = 1.0
            elif status == matching.IGNORED:
                film.tmdb_id, film.match_confidence = None, None
            s.add(film)
            s.commit()
            s.refresh(film)
            return film


def make_tmdb_client(settings: Settings, cache: ResponseCache) -> TmdbClient | None:
    return TmdbClient(settings.tmdb_api_key, cache, settings) if settings.tmdb_api_key else None


@lru_cache
def get_pipeline() -> Pipeline:
    settings = get_settings()
    engine = get_engine()
    return Pipeline(engine, settings, make_tmdb_client(settings, ResponseCache(engine)))


def films_needing_review(session: Session) -> Iterable[UserFilm]:
    return session.exec(
        select(UserFilm)
        .where(col(UserFilm.match_status).in_(matching.NEEDS_REVIEW))
        .order_by(UserFilm.name)
    )
