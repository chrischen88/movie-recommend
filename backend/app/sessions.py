"""In-memory user sessions and the pipeline run queue.

A session holds one user's `Library` for one visit. Nothing in it is ever
written to disk: it's dropped after `session_ttl_seconds` without a request,
when the user deletes it, or when the process restarts. Its id is a random
256-bit token, sent in the `X-Session-Id` header, so it works as a bearer secret.

Pipeline runs go through one worker thread, one at a time (the machine has one
shared CPU and the runs share SQLite), and the queue is capped.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings
from app.db import utcnow
from app.library import Library

UPLOAD_WINDOW_SECONDS = 3600


class Busy(RuntimeError):
    """At the session or queue limit: the user should try again shortly."""


class RateLimited(RuntimeError):
    """Too many uploads from one client in the last hour."""


class AlreadyQueued(RuntimeError):
    """This session already has a run waiting or in progress."""


@dataclass
class RunState:
    """Progress of one pipeline run, returned to the UI as JSON."""

    id: int = 0
    status: str = "queued"  # queued | running | done | error | cancelled
    stage: str = "parsing"
    progress_done: int = 0
    progress_total: int = 0
    message: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None
    # Set when the session is deleted or expires: the pipeline stops at its next check.
    cancelled: bool = False

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")


@dataclass
class UserSession:
    id: str
    library: Library
    run: RunState
    last_seen: float
    upload: dict[str, Any]  # what parsing found (counts, warnings), kept for each run's stats


class InlineExecutor(Executor):
    """Runs each task immediately on the caller's thread (tests and the CLI)."""

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def, override]
        fut: Future[Any] = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 — surfaced through the future
            fut.set_exception(exc)
        return fut


class SessionStore:
    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
        executor: Executor | None = None,
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self._lock = threading.Lock()
        self._sessions: dict[str, UserSession] = {}
        self._queue: list[str] = []  # sessions with a queued or running run, oldest first
        self._uploads: dict[str, deque[float]] = {}  # client → upload times in the last hour

    # ------------------------------------------------------------ sessions

    def create(self, library: Library, upload: dict[str, Any]) -> UserSession:
        with self._lock:
            self._sweep()
            if len(self._sessions) >= self.settings.max_sessions:
                raise Busy("Too many people are using the app right now. Try again in a few minutes.")
            session = UserSession(
                id=secrets.token_urlsafe(32),
                library=library,
                run=RunState(stats=dict(upload)),
                last_seen=self._clock(),
                upload=upload,
            )
            self._sessions[session.id] = session
            return session

    def get(self, session_id: str | None) -> UserSession | None:
        """The live session, marking it used; None if unknown or expired."""
        if not session_id:
            return None
        with self._lock:
            self._sweep()
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_seen = self._clock()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return False
            self._drop(session)
            return True

    def expires_in(self, session: UserSession) -> int:
        return max(0, int(self.settings.session_ttl_seconds - (self._clock() - session.last_seen)))

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------ uploads

    def check_upload_rate(self, client: str) -> None:
        """Count an upload from `client`, or raise if it's over the hourly limit."""
        limit = self.settings.uploads_per_ip_per_hour
        if limit <= 0:
            return
        now = self._clock()
        with self._lock:
            times = self._uploads.setdefault(client, deque())
            while times and now - times[0] >= UPLOAD_WINDOW_SECONDS:
                times.popleft()
            if len(times) >= limit:
                raise RateLimited(f"Upload limit reached ({limit} an hour). Try again later.")
            times.append(now)

    # ------------------------------------------------------------ runs

    def submit(self, session: UserSession, work: Callable[[UserSession], None]) -> None:
        """Queue a pipeline run for `session`. `work` gets the session and must
        record its outcome on `session.run` (Pipeline.run does)."""
        with self._lock:
            if session.id in self._queue:
                raise AlreadyQueued("This export is already being processed.")
            if len(self._queue) >= self.settings.max_queued_runs:
                raise Busy("The server is busy processing other exports. Try again in a few minutes.")
            self._queue.append(session.id)
            session.run = RunState(id=session.run.id + 1, stats=dict(session.upload))
            run = session.run
        self._executor.submit(self._work, session, run, work)

    def queue_position(self, session: UserSession) -> int | None:
        """0 while its run is processing, n while n runs are ahead of it, None when idle."""
        with self._lock:
            return self._queue.index(session.id) if session.id in self._queue else None

    def _work(self, session: UserSession, run: RunState, work: Callable[[UserSession], None]) -> None:
        try:
            if run.cancelled:
                return
            run.status = "running"
            work(session)
        finally:
            with self._lock:
                if session.id in self._queue:
                    self._queue.remove(session.id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------ helpers

    def _drop(self, session: UserSession) -> None:
        """Forget a session (caller holds the lock). A run in progress is told to stop."""
        session.run.cancelled = True
        if session.id in self._queue and session.run.status == "queued":
            self._queue.remove(session.id)

    def _sweep(self) -> None:
        """Drop expired sessions and stale rate-limit entries (caller holds the lock)."""
        now = self._clock()
        ttl = self.settings.session_ttl_seconds
        for sid in [sid for sid, s in self._sessions.items() if now - s.last_seen > ttl]:
            self._drop(self._sessions.pop(sid))
        for client in [c for c, t in self._uploads.items() if not t or now - t[-1] >= UPLOAD_WINDOW_SECONDS]:
            del self._uploads[client]


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore(get_settings())
