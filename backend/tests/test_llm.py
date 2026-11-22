from __future__ import annotations

import random

import pytest
from sqlalchemy import Engine
from sqlmodel import Session, select

from app.db import LlmUsage
from app.llm import (
    LLMService,
    LLMSessionState,
    LLMUnavailable,
    RatedExample,
    constrain_order,
    parse_request,
    rerank,
    template_why,
)
from app.profile import Contribution
from app.recommend import Recommendation
from tests.fake_llm import FakeLLM
from tests.test_pipeline import test_settings


def rec(tid: int, **kw: object) -> Recommendation:
    base: dict = dict(
        tmdb_id=tid, title=f"Film {tid}", year=2000, directors=[], genres=["Drama"], poster_path=None,
        overview="", runtime=100, vote_average=7.0, vote_count=100, original_language="en",
        in_watchlist=False, score=0.5, embedding_score=0.5, similarity=0.5, source="taste",
        source_label="Your overall taste",
    )
    base.update(kw)
    return Recommendation(**base)


def service(engine: Engine, client: FakeLLM | None, **settings: object) -> LLMService:
    s = test_settings()
    for k, v in settings.items():
        setattr(s, k, v)
    return LLMService(client, engine, s)


# ---------------------------------------------------------------- constrain_order


def test_constrain_order_follows_the_model_within_the_shift() -> None:
    original = list(range(10))
    assert constrain_order(original, original, 3) == original
    assert constrain_order(original, [2, 0, 1], 3) == [2, 0, 1, 3, 4, 5, 6, 7, 8, 9]
    # Film 9 can rise only 3 places; films pushed down can fall only 3.
    out = constrain_order(original, list(reversed(original)), 3)
    assert out.index(9) == 6
    assert all(abs(out.index(t) - t) <= 3 for t in original)


def test_constrain_order_ignores_unknown_and_repeated_ids() -> None:
    assert constrain_order([1, 2, 3], [3, 99, 3, 1], 5) == [3, 1, 2]


@pytest.mark.parametrize("seed", range(20))
def test_constrain_order_never_moves_a_film_too_far(seed: int) -> None:
    rng = random.Random(seed)
    original = list(range(30))
    proposed = rng.sample(original, rng.randint(0, 30))
    shift = rng.randint(0, 12)
    out = constrain_order(original, proposed, shift)
    assert sorted(out) == original
    assert all(abs(out.index(t) - t) <= shift for t in original)


# ---------------------------------------------------------------- re-rank


def test_rerank_reorders_the_pool_and_explains(engine: Engine) -> None:
    llm = FakeLLM()
    svc = service(engine, llm, llm_rerank_pool=4, llm_max_position_shift=10)
    recs = [rec(t) for t in (1, 2, 3, 4, 5)]
    items, whys = rerank(svc, LLMSessionState(), recs, [RatedExample("Arrival", 2016, 5.0)], [], svc.settings)
    assert [r.tmdb_id for r in items] == [4, 3, 2, 1, 5]  # pool reversed, the rest unchanged
    assert whys[4] == "Because of film 4."
    name, prompt = llm.calls[0]
    assert name == "rerank" and "Arrival (2016): 5★" in prompt and '"tmdb_id": 5' not in prompt


def test_rerank_drops_films_the_model_made_up(engine: Engine) -> None:
    llm = FakeLLM(lambda *_: {"picks": [{"tmdb_id": 999, "why": "Invented."}, {"tmdb_id": 2, "why": "Real."}]})
    svc = service(engine, llm)
    items, whys = rerank(svc, LLMSessionState(), [rec(1), rec(2)], [], [], svc.settings)
    assert [r.tmdb_id for r in items] == [2, 1] and whys == {2: "Real."}
    nothing = FakeLLM(lambda *_: {"picks": [{"tmdb_id": 999, "why": "Invented."}]})
    with pytest.raises(LLMUnavailable):
        rerank(service(engine, nothing), LLMSessionState(), [rec(1), rec(2)], [], [], svc.settings)


# ---------------------------------------------------------------- service


def test_service_caches_per_session_and_records_usage(engine: Engine) -> None:
    llm = FakeLLM()
    svc = service(engine, llm)
    state, other = LLMSessionState(), LLMSessionState()
    for _ in range(3):
        svc.call(state, "sys", "same prompt", "film_request", {})
    assert len(llm.calls) == 1 and state.calls == 1
    svc.call(other, "sys", "same prompt", "film_request", {})  # caches aren't shared between users
    assert len(llm.calls) == 2
    with Session(engine) as s:
        row = s.exec(select(LlmUsage)).one()
    assert (row.model, row.calls, row.prompt_tokens, row.completion_tokens) == ("fake-llm", 2, 2000, 400)
    usage = svc.usage(state)
    # gpt-4o-mini prices: 2000 × $0.15/M + 400 × $0.60/M
    assert usage["all_time"]["cost_usd"] == pytest.approx(0.00054)
    assert usage["session"]["calls"] == 1 and usage["today"]["calls"] == 2


def test_service_limits(engine: Engine) -> None:
    svc = service(engine, FakeLLM(), llm_calls_per_session=2, llm_daily_call_limit=3)
    a, b = LLMSessionState(), LLMSessionState()
    svc.call(a, "s", "1", "x", {})
    svc.call(a, "s", "2", "x", {})
    with pytest.raises(LLMUnavailable, match="session"):
        svc.call(a, "s", "3", "x", {})
    svc.call(a, "s", "1", "x", {})  # cached answers are free
    svc.call(b, "s", "1", "x", {})
    with pytest.raises(LLMUnavailable, match="today"):
        svc.call(b, "s", "2", "x", {})


def test_failed_calls_dont_use_up_the_session(engine: Engine) -> None:
    llm = FakeLLM(fail=True)
    svc = service(engine, llm, llm_calls_per_session=1)
    state = LLMSessionState()
    for _ in range(3):
        with pytest.raises(LLMUnavailable, match="failed"):
            svc.call(state, "s", "u", "x", {})
    assert state.calls == 0 and len(llm.calls) == 3


def test_service_without_a_key(engine: Engine) -> None:
    svc = service(engine, None)
    assert not svc.enabled
    with pytest.raises(LLMUnavailable, match="OPENAI_API_KEY"):
        svc.call(LLMSessionState(), "s", "u", "x", {})
    assert svc.usage()["enabled"] is False


# ---------------------------------------------------------------- requests


def test_parse_request_validates_the_answer(engine: Engine) -> None:
    llm = FakeLLM(request_answer={
        "genres": ["Thriller", "Made-up Genre"], "decade": 1995, "max_runtime": 95, "language": "Korean",
        "mood_keywords": ["tense"], "query": "a twisty heist",
    })
    parsed, note = parse_request(service(engine, llm), LLMSessionState(), "  a tense 90s heist  ", ["Thriller", "Drama"])
    assert note is None and parsed.by_llm
    assert parsed.genres == ("Thriller",) and parsed.decade == 1990 and parsed.max_runtime == 95
    assert parsed.language is None  # not an ISO 639-1 code
    assert parsed.query == "a twisty heist tense"
    assert "Request: a tense 90s heist" in llm.calls[0][1]


def test_parse_request_falls_back_to_the_raw_text(engine: Engine) -> None:
    parsed, note = parse_request(service(engine, FakeLLM(fail=True)), LLMSessionState(), "cosy mystery", ["Mystery"])
    assert parsed.query == "cosy mystery" and not parsed.by_llm and parsed.genres == ()
    assert note and "failed" in note


# ---------------------------------------------------------------- templates


def test_template_why() -> None:
    villeneuve = Contribution("director", "Denis Villeneuve", "Director Denis Villeneuve", 0.6, 3, 0.4)
    r = rec(1, profile_reasons=[villeneuve], source="cluster:0", source_label="Science Fiction · Drama",
            collab_predicted=4.2)
    assert template_why(r) == (
        "You rate Denis Villeneuve's films highly, it fits your Science Fiction · Drama streak"
        " and viewers with similar ratings give it ~4.2★."
    )
    assert template_why(rec(2)) == "It's close to your overall taste."
    japanese = Contribution("language", "ja", "Language ja", 0.3, 4, 0.2)
    assert template_why(rec(3, profile_reasons=[japanese], source="cluster:1", source_label="Drama (2)")) == (
        "You like films in Japanese and it fits your Drama streak."
    )
