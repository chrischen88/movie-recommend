from __future__ import annotations

import json

import httpx
import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select

from app.cache import (
    ApiError,
    CachedHttpClient,
    DailyBudgetExceeded,
    RateLimiter,
    ResponseCache,
)
from app.db import ApiCache


class Recorder:
    """httpx MockTransport handler that replays a scripted list of responses."""

    def __init__(self, responses: list[httpx.Response | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def make_client(
    engine: Engine, responses: list[httpx.Response | Exception], **kw
) -> tuple[CachedHttpClient, Recorder, list[float]]:
    rec = Recorder(responses)
    sleeps: list[float] = []
    client = CachedHttpClient(
        namespace=kw.pop("namespace", "tmdb"),
        base_url="https://api.example.test/3",
        cache=ResponseCache(engine),
        rate_per_second=kw.pop("rate_per_second", 1000.0),
        ttl_seconds=kw.pop("ttl_seconds", 3600),
        secret_params={"api_key": "SECRET"},
        max_retries=kw.pop("max_retries", 3),
        transport=httpx.MockTransport(rec),
        sleep=sleeps.append,
        **kw,
    )
    return client, rec, sleeps


# ------------------------------------------------------------ ResponseCache


def test_cache_roundtrip_and_ttl(engine: Engine) -> None:
    cache = ResponseCache(engine)
    cache.set("ns", "/a", {"q": 1}, 200, {"x": 1}, ttl_seconds=60)
    hit = cache.get("ns", "/a", {"q": 1})
    assert hit is not None and hit.body == {"x": 1}
    assert cache.get("ns", "/a", {"q": 2}) is None
    assert cache.get("other", "/a", {"q": 1}) is None

    cache.set("ns", "/expired", None, 200, {"x": 2}, ttl_seconds=-1)
    assert cache.get("ns", "/expired", None) is None

    cache.set("ns", "/forever", None, 200, [1, 2], ttl_seconds=None)
    forever = cache.get("ns", "/forever", None)
    assert forever is not None and forever.body == [1, 2]


def test_cache_key_is_param_order_independent() -> None:
    k1 = ResponseCache.make_key("ns", "/a", {"a": 1, "b": 2})
    k2 = ResponseCache.make_key("ns", "/a", {"b": 2, "a": 1})
    assert k1 == k2


# ------------------------------------------------------------ CachedHttpClient


def test_second_call_served_from_cache(engine: Engine) -> None:
    client, rec, _ = make_client(engine, [httpx.Response(200, json={"id": 1})])
    assert client.get_json("/movie/1") == {"id": 1}
    assert client.get_json("/movie/1") == {"id": 1}
    assert len(rec.requests) == 1


def test_secret_sent_but_not_cached(engine: Engine) -> None:
    client, rec, _ = make_client(engine, [httpx.Response(200, json={})])
    client.get_json("/search/movie", {"query": "Heat", "year": 1995})
    assert rec.requests[0].url.params["api_key"] == "SECRET"
    with Session(engine) as s:
        row = s.exec(select(ApiCache)).one()
    assert "SECRET" not in row.params_json
    assert json.loads(row.params_json) == {"query": "Heat", "year": 1995}


def test_none_params_dropped(engine: Engine) -> None:
    client, rec, _ = make_client(engine, [httpx.Response(200, json={})])
    client.get_json("/search/movie", {"query": "Heat", "year": None})
    assert "year" not in rec.requests[0].url.params


def test_404_cached_as_missing(engine: Engine) -> None:
    client, rec, _ = make_client(engine, [httpx.Response(404, json={"status": "nope"})])
    assert client.get_json("/movie/999") is None
    assert client.get_json("/movie/999") is None
    assert len(rec.requests) == 1


def test_retry_on_429_honours_retry_after(engine: Engine) -> None:
    client, rec, sleeps = make_client(
        engine,
        [httpx.Response(429, headers={"Retry-After": "2"}), httpx.Response(200, json={"ok": 1})],
    )
    assert client.get_json("/x") == {"ok": 1}
    assert len(rec.requests) == 2
    assert 2.0 in sleeps


def test_retry_on_transport_error_with_backoff(engine: Engine) -> None:
    client, rec, sleeps = make_client(
        engine,
        [
            httpx.ConnectError("boom"),
            httpx.Response(503),
            httpx.Response(200, json={"ok": 1}),
        ],
    )
    assert client.get_json("/x") == {"ok": 1}
    backoffs = [s for s in sleeps if s > 0.01]  # ignore limiter micro-sleeps
    assert len(backoffs) == 2
    assert backoffs[1] > backoffs[0] * 0.5  # grows (jittered)


def test_retries_exhausted_raises(engine: Engine) -> None:
    client, rec, _ = make_client(engine, [httpx.Response(500)] * 4, max_retries=3)
    with pytest.raises(ApiError, match="after 4 attempts"):
        client.get_json("/x")
    assert len(rec.requests) == 4


def test_auth_error_not_retried_or_cached(engine: Engine) -> None:
    client, rec, _ = make_client(
        engine, [httpx.Response(401, json={"status_message": "bad key"}), httpx.Response(200, json={})]
    )
    with pytest.raises(ApiError) as info:
        client.get_json("/x")
    assert info.value.status_code == 401
    assert len(rec.requests) == 1
    assert client.get_json("/x") == {}  # not cached, so it refetches


def test_daily_budget(engine: Engine) -> None:
    client, rec, _ = make_client(
        engine,
        [httpx.Response(200, json={"n": i}) for i in range(3)],
        namespace="omdb",
        daily_limit=2,
    )
    client.get_json("/", {"i": "tt1"})
    client.get_json("/", {"i": "tt2"})
    client.get_json("/", {"i": "tt1"})  # cached: doesn't count
    with pytest.raises(DailyBudgetExceeded):
        client.get_json("/", {"i": "tt3"})
    assert len(rec.requests) == 2


def test_daily_budget_holds_under_concurrency(engine: Engine) -> None:
    """Parallel workers must not all pass the budget check before any response
    is cached (they did: 5 requests went out on a budget of 2)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    calls: list[str] = []
    gate = threading.Barrier(8, timeout=5)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["i"])
        return httpx.Response(200, json={"ok": True})

    cache = ResponseCache(engine)
    client = CachedHttpClient(
        namespace="omdb", base_url="https://x.test", cache=cache, rate_per_second=1000,
        ttl_seconds=None, daily_limit=3, transport=httpx.MockTransport(handler), max_retries=0,
    )

    def one(i: int) -> str:
        gate.wait()  # all 8 hit the budget check at the same moment
        try:
            client.get_json("/", {"i": f"tt{i}"})
            return "ok"
        except DailyBudgetExceeded:
            return "over"

    with ThreadPoolExecutor(8) as pool:
        outcomes = list(pool.map(one, range(8)))
    assert len(calls) == 3 and outcomes.count("ok") == 3 and outcomes.count("over") == 5


# ------------------------------------------------------------ RateLimiter


def test_rate_limiter_spaces_requests() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(d: float) -> None:
        sleeps.append(d)
        now[0] += d

    limiter = RateLimiter(4.0, clock=lambda: now[0], sleep=sleep)
    for _ in range(4):
        limiter.acquire()
    assert sleeps == pytest.approx([0.25, 0.25, 0.25])
