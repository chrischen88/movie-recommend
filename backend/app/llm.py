"""The LLM layer (SPEC "LLM layer"): re-rank + explain the top picks, and turn
natural-language requests into filters plus a query.

Every call goes through `LLMService`, which adds:
- a cache per session (`LLMSessionState`, in memory). The prompts carry the
  user's ratings, so unlike API responses for film data they're never written
  to disk;
- cost guards: a daily call limit across all users and a per-session limit;
- usage accounting: calls and tokens per day and model in `LlmUsage` (counts
  only), shown with their cost on the metrics page.

Nothing here is required: with no `OPENAI_API_KEY`, or when a call fails,
callers fall back to the blended order and `template_why` explanations, and a
request's raw text becomes its query.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import Engine, func
from sqlmodel import Session, select

from app.config import Settings
from app.db import LlmUsage, utcnow

if TYPE_CHECKING:
    from app.recommend import Recommendation

log = logging.getLogger(__name__)

SESSION_CACHE_SIZE = 64


class LLMUnavailable(RuntimeError):
    """No key, over a limit, or the call failed: use the non-LLM fallback."""


@dataclass(frozen=True)
class LLMResult:
    data: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    """One structured-output completion. Swap providers by implementing this."""

    model: str

    def complete_json(self, system: str, user: str, name: str, schema: dict[str, Any]) -> LLMResult:
        """Return the model's JSON reply, which must match `schema`. Raise
        LLMUnavailable on any failure."""
        ...


class OpenAIClient:
    """OpenAI chat completions with Structured Outputs (strict JSON schema)."""

    def __init__(self, api_key: str, model: str, timeout: float = 30.0) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=2)

    def complete_json(self, system: str, user: str, name: str, schema: dict[str, Any]) -> LLMResult:
        import openai

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}},
                temperature=0.3,
            )
        except openai.OpenAIError as exc:
            raise LLMUnavailable(f"OpenAI call failed: {type(exc).__name__}") from exc
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None) or not msg.content:
            raise LLMUnavailable("the model returned no answer")
        try:
            data = json.loads(msg.content)
        except ValueError as exc:
            raise LLMUnavailable("the model's answer wasn't valid JSON") from exc
        usage = resp.usage
        return LLMResult(
            data=data,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )


@dataclass
class LLMSessionState:
    """One session's LLM cache and usage. Lives on its Library, in memory."""

    cache: OrderedDict[str, dict[str, Any]] = field(default_factory=OrderedDict)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMService:
    def __init__(self, client: LLMClient | None, engine: Engine, settings: Settings) -> None:
        self.client = client
        self.engine = engine
        self.settings = settings
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @property
    def model(self) -> str | None:
        return self.client.model if self.client else None

    def call(self, state: LLMSessionState, system: str, user: str, name: str, schema: dict[str, Any]) -> dict[str, Any]:
        """The model's JSON answer, from this session's cache when the same
        prompt was sent before. Raises LLMUnavailable."""
        if self.client is None:
            raise LLMUnavailable("OPENAI_API_KEY isn't set")
        key = hashlib.sha256(
            json.dumps([self.client.model, name, schema, system, user], sort_keys=True).encode()
        ).hexdigest()
        with self._lock:
            if key in state.cache:
                state.cache.move_to_end(key)
                return state.cache[key]
            cfg = self.settings
            if cfg.llm_calls_per_session and state.calls >= cfg.llm_calls_per_session:
                raise LLMUnavailable("this session has used its AI requests")
            if cfg.llm_daily_call_limit and self._calls_today() >= cfg.llm_daily_call_limit:
                raise LLMUnavailable("the app has used today's AI requests")
            state.calls += 1  # counted before the call, so concurrent requests can't overshoot
        try:
            result = self.client.complete_json(system, user, name, schema)
        except Exception:
            with self._lock:
                state.calls -= 1  # a failed call doesn't use up the session's allowance
            raise
        with self._lock:
            state.prompt_tokens += result.prompt_tokens
            state.completion_tokens += result.completion_tokens
            self._record(result)
            state.cache[key] = result.data
            while len(state.cache) > SESSION_CACHE_SIZE:
                state.cache.popitem(last=False)
        log.info("llm %s: %d prompt + %d completion tokens", name, result.prompt_tokens, result.completion_tokens)
        return result.data

    # -------------------------------------------------------- usage + cost

    @staticmethod
    def _today() -> str:
        return utcnow().date().isoformat()

    def _calls_today(self) -> int:
        with Session(self.engine) as s:
            return int(s.exec(select(func.coalesce(func.sum(LlmUsage.calls), 0)).where(LlmUsage.day == self._today())).one())

    def _record(self, result: LLMResult) -> None:
        assert self.client is not None
        with Session(self.engine) as s:
            row = s.get(LlmUsage, (self._today(), self.client.model)) or LlmUsage(day=self._today(), model=self.client.model)
            row.calls += 1
            row.prompt_tokens += result.prompt_tokens
            row.completion_tokens += result.completion_tokens
            s.add(row)
            s.commit()

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cfg = self.settings
        return (prompt_tokens * cfg.llm_price_input_per_mtok + completion_tokens * cfg.llm_price_output_per_mtok) / 1e6

    def usage(self, state: LLMSessionState | None = None) -> dict[str, Any]:
        """For the metrics page: today's and all-time usage across all users,
        plus this session's."""
        with Session(self.engine) as s:
            rows = s.exec(select(LlmUsage)).all()
        today = [r for r in rows if r.day == self._today()]

        def total(rs: Sequence[LlmUsage]) -> dict[str, Any]:
            p, c = sum(r.prompt_tokens for r in rs), sum(r.completion_tokens for r in rs)
            return {"calls": sum(r.calls for r in rs), "prompt_tokens": p, "completion_tokens": c,
                    "cost_usd": round(self.cost(p, c), 6)}

        out: dict[str, Any] = {
            "enabled": self.enabled,
            "model": self.model,
            "daily_call_limit": self.settings.llm_daily_call_limit,
            "today": total(today),
            "all_time": total(rows),
        }
        if state is not None:
            out["session"] = {
                "calls": state.calls, "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "cost_usd": round(self.cost(state.prompt_tokens, state.completion_tokens), 6),
                "limit": self.settings.llm_calls_per_session,
            }
        return out


# ---------------------------------------------------------------- explanations


def template_why(r: Recommendation) -> str:
    """A one-sentence explanation from the top contributing features, used when
    there's no LLM explanation."""
    parts: list[str] = []
    liked = [c for c in r.profile_reasons if c.stars > 0]
    if liked:
        parts.append(_feature_phrase(liked[0].type, liked[0].value))
    if r.source.startswith("cluster:"):
        label = re.sub(r" \(\d+\)$", "", r.source_label)  # "Drama · Romance (2)": the 2 means nothing to a reader
        parts.append(f"it fits your {label} streak")
    elif not parts:
        parts.append("it's close to your overall taste")
    if r.collab_predicted is not None and r.collab_predicted >= 3.5:
        parts.append(f"viewers with similar ratings give it ~{r.collab_predicted:.1f}★")
    text = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return text[0].upper() + text[1:] + "."


LANGUAGES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "cn": "Cantonese", "hi": "Hindi", "ru": "Russian", "pt": "Portuguese",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish", "pl": "Polish", "fa": "Persian",
    "tr": "Turkish", "ar": "Arabic", "nl": "Dutch", "el": "Greek", "hu": "Hungarian", "cs": "Czech",
    "ro": "Romanian", "th": "Thai", "id": "Indonesian", "he": "Hebrew", "uk": "Ukrainian", "ta": "Tamil",
}


def _feature_phrase(kind: str, value: str) -> str:
    if kind == "language":
        value = LANGUAGES.get(value, value)
    return {
        "director": f"you rate {value}'s films highly",
        "actor": f"you like films with {value}",
        "genre": f"you rate {value} highly",
        "keyword": f"you like films about {value}",
        "decade": f"you like films from the {value}",
        "language": f"you like films in {value}",
        "country": f"you like films from {value}",
    }.get(kind, f"you like {value}")


# ---------------------------------------------------------------- re-rank


RERANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"tmdb_id": {"type": "integer"}, "why": {"type": "string"}},
                "required": ["tmdb_id", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["picks"],
    "additionalProperties": False,
}

RERANK_SYSTEM = """You help a film recommender pick and explain films for one viewer.
You get the viewer's highest- and lowest-rated films and a list of candidate films the
recommender already ranked, with its scores. Choose the {keep} candidates this viewer is
most likely to love and order them best first. For each, write one sentence (at most 25
words, second person) on why they'll like it, pointing to films they rated when that helps.
Rules: use only tmdb_ids from the candidate list; never add or invent films; don't state
facts that aren't in the data you were given."""


@dataclass(frozen=True)
class RatedExample:
    title: str
    year: int | None
    rating: float


def rerank_prompt(
    recs: Sequence[Recommendation], liked: Sequence[RatedExample], disliked: Sequence[RatedExample], keep: int
) -> tuple[str, str]:
    def film(e: RatedExample) -> str:
        return f"- {e.title} ({e.year or '?'}): {e.rating:g}★"

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{round(100 * v)}"

    lines = ["Viewer's highest-rated films:", *map(film, liked), "", "Viewer's lowest-rated films:", *map(film, disliked),
             "", "Candidates (recommender's order; scores are percentiles, 100 = best match):"]
    for r in recs:
        ext = [f"IMDb {r.imdb_rating:g}" if r.imdb_rating is not None else "",
               f"RT {r.rt_score}%" if r.rt_score is not None else ""]
        reasons = "; ".join(c.label for c in r.profile_reasons[:3] if c.stars > 0)
        lines.append(json.dumps({
            "tmdb_id": r.tmdb_id,
            "title": f"{r.title} ({r.year or '?'})",
            "genres": r.genres,
            "directors": r.directors[:2],
            "overview": (r.overview or "")[:300],
            "taste_profile": pct(r.profile_score),
            "embedding_similarity": pct(r.embedding_score),
            "similar_viewers": pct(r.collab_score),
            "predicted_stars": None if r.predicted_rating is None else round(r.predicted_rating, 1),
            "matches_cluster": r.source_label,
            "profile_reasons": reasons,
            "critics": ", ".join(x for x in ext if x) or None,
        }, ensure_ascii=False))
    return RERANK_SYSTEM.format(keep=keep), "\n".join(lines)


def constrain_order(original: Sequence[int], proposed: Sequence[int], max_shift: int) -> list[int]:
    """`original` reordered toward `proposed`, moving no film more than
    `max_shift` places either way. Ids not in `proposed` keep their relative
    order after the proposed ones; unknown and repeated ids in `proposed` are
    ignored. Greedy: each slot takes a film that would otherwise fall more than
    `max_shift` places, else the most-preferred film allowed to move up to it."""
    known = set(original)
    preferred: list[int] = []
    for t in proposed:
        if t in known and t not in preferred:
            preferred.append(t)
    preferred += [t for t in original if t not in preferred]
    old = {t: i for i, t in enumerate(original)}
    out: list[int] = []
    left = list(preferred)
    for pos in range(len(original)):
        forced = [t for t in left if old[t] + max_shift <= pos]
        pick = min(forced, key=old.__getitem__) if forced else next(t for t in left if old[t] - max_shift <= pos)
        out.append(pick)
        left.remove(pick)
    return out


def rerank(
    service: LLMService,
    state: LLMSessionState,
    recs: Sequence[Recommendation],
    liked: Sequence[RatedExample],
    disliked: Sequence[RatedExample],
    settings: Settings,
) -> tuple[list[Recommendation], dict[int, str]]:
    """The first `llm_rerank_pool` films in the model's order (within
    `llm_max_position_shift` of their blended rank), then the rest unchanged, and
    the model's "why" per film. Raises LLMUnavailable, or when the model's answer
    has no usable picks."""
    pool = list(recs[: settings.llm_rerank_pool])
    if len(pool) < 2:
        return list(recs), {}
    keep = min(settings.llm_rerank_keep, len(pool))
    system, user = rerank_prompt(pool, liked, disliked, keep)
    data = service.call(state, system, user, "rerank", RERANK_SCHEMA)
    ids = {r.tmdb_id for r in pool}
    whys: dict[int, str] = {}
    proposed: list[int] = []
    for p in data.get("picks", []):
        t, why = p.get("tmdb_id"), str(p.get("why", "")).strip()
        if t in ids and t not in whys:
            proposed.append(t)
            whys[t] = why[:300]
    dropped = len(data.get("picks", [])) - len(proposed)
    if dropped:
        log.warning("llm rerank: ignored %d picks with unknown or repeated ids", dropped)
    if not proposed:
        raise LLMUnavailable("the model returned no valid picks")
    by_id = {r.tmdb_id: r for r in pool}
    order = constrain_order([r.tmdb_id for r in pool], proposed, settings.llm_max_position_shift)
    return [by_id[t] for t in order] + list(recs[len(pool):]), {t: w for t, w in whys.items() if w}


# ---------------------------------------------------------------- requests


# TMDB's movie genres (GET /genre/movie/list), as stored on Movie.genres.
TMDB_GENRES = (
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy",
    "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War",
    "Western",
)


@dataclass(frozen=True)
class ParsedRequest:
    query: str  # what gets embedded
    genres: tuple[str, ...] = ()
    decade: int | None = None
    max_runtime: int | None = None
    language: str | None = None
    keywords: tuple[str, ...] = ()
    by_llm: bool = False


REQUEST_SYSTEM = """You turn a viewer's film request into search filters for a recommender.
Only set a filter the request clearly asks for; otherwise leave it null or empty.
- genres: from the allowed list only (any of them may match).
- decade: e.g. 1990 for "90s films".
- max_runtime: minutes, e.g. 100 for "under 100 minutes", ~110 for "a quick watch tonight".
- language: ISO 639-1 code if a language or country's cinema is requested, e.g. "ko".
- mood_keywords: up to 5 words for tone or themes ("lighter", "tense", "melancholy").
- query: 10-25 words describing the films wanted (themes, tone, setting, style), for
  semantic search over plot summaries. When the request names a film, describe what that
  film is like instead of naming it, then apply the requested changes. Example: "like
  Parasite but less violent" -> "sharp, darkly funny social satire about class and two
  families, suspenseful but with little violence"."""


def request_schema(genres: Sequence[str]) -> dict[str, Any]:
    genre_item: dict[str, Any] = {"type": "string", "enum": list(genres)} if genres else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "genres": {"type": "array", "items": genre_item},
            "decade": {"type": ["integer", "null"]},
            "max_runtime": {"type": ["integer", "null"]},
            "language": {"type": ["string", "null"]},
            "mood_keywords": {"type": "array", "items": {"type": "string"}},
            "query": {"type": "string"},
        },
        "required": ["genres", "decade", "max_runtime", "language", "mood_keywords", "query"],
        "additionalProperties": False,
    }


def parse_request(
    service: LLMService, state: LLMSessionState, text: str, genres: Sequence[str]
) -> tuple[ParsedRequest, str | None]:
    """The request as filters + query, and why the LLM wasn't used (None if it
    was). Without the LLM the raw text is the query."""
    text = " ".join(text.split())[:300]
    try:
        data = service.call(state, REQUEST_SYSTEM, f"Allowed genres: {', '.join(genres)}\nRequest: {text}",
                            "film_request", request_schema(sorted(genres)))
    except LLMUnavailable as exc:
        return ParsedRequest(query=text), str(exc)
    allowed = set(genres)
    decade = data.get("decade")
    runtime = data.get("max_runtime")
    lang = data.get("language")
    keywords = tuple(str(k)[:30] for k in (data.get("mood_keywords") or [])[:5] if str(k).strip())
    query = str(data.get("query") or "").strip()[:300] or text
    return ParsedRequest(
        query=" ".join([query, *keywords]),
        genres=tuple(g for g in data.get("genres") or [] if g in allowed),
        decade=decade // 10 * 10 if isinstance(decade, int) and 1870 <= decade <= 2100 else None,
        max_runtime=runtime if isinstance(runtime, int) and 30 <= runtime <= 400 else None,
        language=lang.lower() if isinstance(lang, str) and re.fullmatch(r"[a-zA-Z]{2}", lang) else None,
        keywords=keywords,
        by_llm=True,
    ), None
