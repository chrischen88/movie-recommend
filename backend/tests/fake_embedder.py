"""Deterministic bag-of-words hashing embedder: fast, offline, and documents
sharing words get similar vectors, which is all the tests need."""

from __future__ import annotations

import hashlib
import re

import numpy as np


class HashEmbedder:
    model_name = "hash-test"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0
        self.texts_embedded = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.texts_embedded += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)
