from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator

import pytest

from app.config import Settings
from app.library import Library
from app.sessions import AlreadyQueued, Busy, InlineExecutor, RateLimited, SessionStore, UserSession


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[call-arg]


def lib() -> Library:
    return Library({})


def wait_until(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(clock: Clock) -> Iterator[SessionStore]:
    s = SessionStore(settings(session_ttl_seconds=60, max_sessions=3), clock=clock, executor=InlineExecutor())
    yield s
    s.shutdown()


def test_create_get_delete(store: SessionStore) -> None:
    a, b = store.create(lib(), {}), store.create(lib(), {})
    assert a.id != b.id and len(a.id) >= 40
    assert store.get(a.id) is a and store.get(None) is None and store.get("nope") is None
    assert store.delete(a.id) is True and store.delete(a.id) is False
    assert store.get(a.id) is None and store.get(b.id) is b


def test_sessions_expire_when_idle(store: SessionStore, clock: Clock) -> None:
    s = store.create(lib(), {})
    clock.now += 50
    assert store.get(s.id) is s  # use resets the idle timer
    assert store.expires_in(s) == 60
    clock.now += 59
    assert store.get(s.id) is s
    clock.now += 61
    assert store.get(s.id) is None and len(store) == 0


def test_session_cap(store: SessionStore, clock: Clock) -> None:
    for _ in range(3):
        store.create(lib(), {})
    with pytest.raises(Busy):
        store.create(lib(), {})
    clock.now += 61  # expired sessions free their slots
    store.create(lib(), {})


def test_upload_rate_limit(clock: Clock) -> None:
    store = SessionStore(settings(uploads_per_ip_per_hour=2), clock=clock, executor=InlineExecutor())
    store.check_upload_rate("1.2.3.4")
    store.check_upload_rate("1.2.3.4")
    with pytest.raises(RateLimited):
        store.check_upload_rate("1.2.3.4")
    store.check_upload_rate("5.6.7.8")  # per client
    clock.now += 3600
    store.check_upload_rate("1.2.3.4")
    SessionStore(settings(uploads_per_ip_per_hour=0), executor=InlineExecutor()).check_upload_rate("x")


def test_runs_queue_one_at_a_time(clock: Clock) -> None:
    store = SessionStore(settings(max_queued_runs=2, max_sessions=10), clock=clock)
    release = threading.Event()
    started = threading.Event()
    order: list[str] = []

    def work(session: UserSession) -> None:
        order.append(session.id)
        started.set()
        release.wait(5)
        session.run.status = "done"

    a, b, c = (store.create(lib(), {"films": 1}) for _ in range(3))
    store.submit(a, work)
    assert started.wait(5)
    store.submit(b, work)
    assert store.queue_position(a) == 0 and store.queue_position(b) == 1
    assert a.run.status == "running" and b.run.status == "queued" and b.run.stats == {"films": 1}
    with pytest.raises(AlreadyQueued):
        store.submit(b, work)
    with pytest.raises(Busy):
        store.submit(c, work)  # queue full

    release.set()
    wait_until(lambda: store.queue_position(b) is None)
    assert order == [a.id, b.id]
    assert store.queue_position(a) is None and b.run.status == "done"
    store.shutdown()


def test_deleting_a_queued_session_cancels_its_run(clock: Clock) -> None:
    store = SessionStore(settings(max_sessions=10), clock=clock)
    release = threading.Event()
    ran: list[str] = []

    def work(session: UserSession) -> None:
        ran.append(session.id)
        release.wait(5)
        session.run.status = "done"

    a, b = store.create(lib(), {}), store.create(lib(), {})
    store.submit(a, work)
    store.submit(b, work)
    assert store.delete(b.id)
    assert b.run.cancelled and store.queue_position(b) is None
    assert store.delete(a.id)
    assert a.run.cancelled  # the running pipeline stops at its next check
    release.set()
    wait_until(lambda: store.queue_position(a) is None)
    assert ran == [a.id]  # b's run never started
    store.shutdown()


def test_resubmit_starts_a_new_run(store: SessionStore) -> None:
    s = store.create(lib(), {"films": 3})

    def work(session: UserSession) -> None:
        session.run.status = "done"

    store.submit(s, work)
    first = s.run
    store.submit(s, work)
    assert s.run is not first and s.run.id == first.id + 1 and s.run.status == "done"
    assert s.run.stats == {"films": 3}
