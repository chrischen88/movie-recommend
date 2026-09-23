"""Film documents + text embedding (sentence-transformers behind a small protocol)."""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Protocol

import numpy as np

from app.db import Movie

log = logging.getLogger(__name__)

# BGE models expect this prefix on short *queries* (not on documents).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalized embeddings."""
        ...


class SentenceTransformerEmbedder:
    """Lazily loads the model on first use (the download can take a while)."""

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):  # type: ignore[no-untyped-def]
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                log.info("loading embedding model %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vecs = self._load().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        prefix = BGE_QUERY_INSTRUCTION if "bge" in self.model_name.lower() else ""
        return self.embed([prefix + text])[0]


def build_document(movie: Movie, max_reviews: int = 2) -> str:
    """title + year + genres + director + keywords + overview + review snippets."""
    parts = [f"{movie.title} ({movie.year})" if movie.year else movie.title]
    if movie.genres:
        parts.append("Genres: " + ", ".join(movie.genres))
    if movie.directors:
        parts.append("Directed by " + ", ".join(movie.directors))
    if movie.keywords:
        parts.append("Keywords: " + ", ".join(movie.keywords[:20]))
    if movie.overview:
        parts.append(movie.overview)
    for review in movie.reviews[:max_reviews]:
        parts.append("Review: " + review)
    return "\n".join(parts)


def doc_hash(document: str, model_name: str) -> str:
    """Changes when the text *or* the model changes → triggers re-embedding."""
    return hashlib.sha256(f"{model_name}\n{document}".encode()).hexdigest()[:16]


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)
