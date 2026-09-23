"""SQLite-backed response cache, request throttling and retrying HTTP client.

Every external call goes through `CachedHttpClient.get_json`, which:
  1. returns the cached body if an unexpired entry exists (no network),
  2. otherwise waits on a per-client rate limiter,
  3. retries 429 / 5xx / transport errors with exponential backoff
     (honouring `Retry-After`),
  4. caches successful responses *and* 404s (so a known-missing resource is
     not refetched), but never auth errors or other 4xx.

Secret params/headers (API keys) are sent on the wire but excluded from the cache key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import Engine, func
from sqlmodel import Session, select

from app.db import ApiCache, utcnow

log = logging.getLogger(__name__)

JsonBody = dict[str, Any] | list[Any]


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class DailyBudgetExceeded(ApiError):
    pass


@dataclass(frozen=True)
class CachedResponse:
    status_code: int
    body: JsonBody | None
    created_at: datetime


def _as_utc(dt: datetime) -> datetime:
    # SQLite drops tzinfo on round-trip; values are always stored as UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class ResponseCache:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(namespace: str, endpoint: str, params: Mapping[str, Any] | None) -> str:
        payload = json.dumps(
            {"ns": namespace, "ep": endpoint, "p": dict(params or {})},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(
        self, namespace: str, endpoint: str, params: Mapping[str, Any] | None
    ) -> CachedResponse | None:
        key = self.make_key(namespace, endpoint, params)
        with Session(self.engine) as s:
            row = s.get(ApiCache, key)
            if row is None or (
                row.expires_at is not None and _as_utc(row.expires_at) <= utcnow()
            ):
                self.misses += 1
                return None
            self.hits += 1
            return CachedResponse(
                status_code=row.status_code,
                body=json.loads(row.body_json),
                created_at=_as_utc(row.created_at),
            )

    def set(
        self,
        namespace: str,
        endpoint: str,
        params: Mapping[str, Any] | None,
        status_code: int,
        body: JsonBody | None,
        ttl_seconds: int | None,
    ) -> None:
        key = self.make_key(namespace, endpoint, params)
        now = utcnow()
        row = ApiCache(
            key=key,
            namespace=namespace,
            endpoint=endpoint,
            params_json=json.dumps(dict(params or {}), sort_keys=True, default=str),
            status_code=status_code,
            body_json=json.dumps(body),
            created_at=now,
            expires_at=None if ttl_seconds is None else now + timedelta(seconds=ttl_seconds),
        )
        with Session(self.engine) as s:
            s.merge(row)
            s.commit()

    def count_created_since(self, namespace: str, since: datetime) -> int:
        with Session(self.engine) as s:
            stmt = (
                select(func.count())
                .select_from(ApiCache)
                .where(ApiCache.namespace == namespace, ApiCache.created_at >= since)
            )
            return int(s.exec(stmt).one())


class RateLimiter:
    """Thread-safe minimum-interval throttle (`rate` requests per second)."""

    def __init__(
        self,
        rate_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self.interval = 1.0 / rate_per_second
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self.interval
        if wait > 0:
            self._sleep(wait)


RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class _UseClientTTL:
    """Sentinel: use the client's default TTL (distinct from None = never expire)."""


USE_CLIENT_TTL = _UseClientTTL()


class CachedHttpClient:
    def __init__(
        self,
        *,
        namespace: str,
        base_url: str,
        cache: ResponseCache,
        rate_per_second: float,
        ttl_seconds: int | None,
        secret_params: Mapping[str, str] | None = None,
        secret_headers: Mapping[str, str] | None = None,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        backoff_max: float = 30.0,
        timeout: float = 20.0,
        daily_limit: int | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.namespace = namespace
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.secret_params = dict(secret_params or {})
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.daily_limit = daily_limit
        # Requests that passed the budget check but haven't been cached yet.
        # Without this, parallel workers could all pass the check at once.
        self._budget_lock = threading.Lock()
        self._in_flight = 0
        self._sleep = sleep
        self.limiter = RateLimiter(rate_per_second, sleep=sleep)
        self.network_calls = 0
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers=dict(secret_headers or {}),
        )

    def close(self) -> None:
        self._http.close()

    def _reserve_budget(self) -> bool:
        """Claim one request from today's budget; True if a claim was made (the
        caller must `_release_budget` once the response is cached or failed)."""
        if self.daily_limit is None:
            return False
        start_of_day = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        with self._budget_lock:
            used = self.cache.count_created_since(self.namespace, start_of_day) + self._in_flight
            if used >= self.daily_limit:
                raise DailyBudgetExceeded(
                    f"{self.namespace}: daily request budget of {self.daily_limit} reached"
                )
            self._in_flight += 1
        return True

    def _release_budget(self) -> None:
        with self._budget_lock:
            self._in_flight -= 1

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), self.backoff_max)
            except ValueError:
                pass
        delay = self.backoff_base * (2**attempt)
        return min(delay, self.backoff_max) * (0.5 + random.random() / 2)

    def get_json(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        *,
        ttl_seconds: int | None | _UseClientTTL = USE_CLIENT_TTL,
        use_cache: bool = True,
    ) -> JsonBody | None:
        """GET `endpoint` and return the JSON body, or None for a 404."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        ttl = self.ttl_seconds if isinstance(ttl_seconds, _UseClientTTL) else ttl_seconds

        if use_cache:
            cached = self.cache.get(self.namespace, endpoint, params)
            if cached is not None:
                return cached.body if cached.status_code != 404 else None

        claim = [self._reserve_budget()]  # [still holding a budget reservation?]
        try:
            return self._fetch(endpoint, params, ttl, claim)
        finally:
            if claim[0]:
                self._release_budget()

    def _store(
        self, endpoint: str, params: dict[str, Any], status: int, body: JsonBody | None,
        ttl: int | None, claim: list[bool],
    ) -> None:
        """Cache a response. With a budget, the cached row replaces the in-flight
        reservation atomically, so the request is never counted twice or not at all."""
        if not claim[0]:
            self.cache.set(self.namespace, endpoint, params, status, body, ttl)
            return
        with self._budget_lock:
            self.cache.set(self.namespace, endpoint, params, status, body, ttl)
            self._in_flight -= 1
            claim[0] = False

    def _fetch(
        self, endpoint: str, params: dict[str, Any], ttl: int | None, claim: list[bool]
    ) -> JsonBody | None:
        last_error: str = "unknown error"
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                self.network_calls += 1
                resp = self._http.get(endpoint, params={**params, **self.secret_params})
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc!r}"
                retry_after = None
            else:
                if resp.status_code == 200:
                    body = resp.json()
                    self._store(endpoint, params, 200, body, ttl, claim)
                    return body
                if resp.status_code == 404:
                    log.info("%s %s -> 404 (cached as missing)", self.namespace, endpoint)
                    self._store(endpoint, params, 404, None, ttl, claim)
                    return None
                if resp.status_code not in RETRYABLE_STATUS:
                    raise ApiError(
                        f"{self.namespace} {endpoint} -> HTTP {resp.status_code}: "
                        f"{resp.text[:200]}",
                        status_code=resp.status_code,
                    )
                last_error = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")

            if attempt < self.max_retries:
                delay = self._backoff(attempt, retry_after)
                log.warning(
                    "%s %s failed (%s); retry %d/%d in %.2fs",
                    self.namespace, endpoint, last_error, attempt + 1, self.max_retries, delay,
                )
                self._sleep(delay)

        raise ApiError(
            f"{self.namespace} {endpoint} failed after {self.max_retries + 1} attempts: "
            f"{last_error}"
        )
