"""A scripted LLMClient: tests never call OpenAI."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from app.llm import LLMResult, LLMUnavailable


class FakeLLM:
    """Answers with `respond(name, system, user)`. By default, re-rank reverses
    the candidates it was given and request parsing returns `request_answer`."""

    model = "fake-llm"

    def __init__(
        self,
        respond: Callable[[str, str, str], dict[str, Any]] | None = None,
        request_answer: dict[str, Any] | None = None,
        fail: bool = False,
    ) -> None:
        self.respond = respond or self._default
        self.request_answer = request_answer or {
            "genres": [], "decade": None, "max_runtime": None, "language": None, "mood_keywords": [], "query": "",
        }
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def _default(self, name: str, system: str, user: str) -> dict[str, Any]:
        if name == "rerank":
            cands = [json.loads(line) for line in user.splitlines() if line.startswith('{"n": ')]
            return {"picks": [{"n": c["n"], "title": c["title"], "why": f"Because of {c['title']}."}
                              for c in reversed(cands)]}
        return dict(self.request_answer)

    def complete_json(self, system: str, user: str, name: str, schema: dict[str, Any]) -> LLMResult:
        self.calls.append((name, user))
        if self.fail:
            raise LLMUnavailable("OpenAI call failed: APITimeoutError")
        data = self.respond(name, system, user)
        json.dumps(data)  # must be JSON, like a real answer
        return LLMResult(data=data, prompt_tokens=1000, completion_tokens=200)
