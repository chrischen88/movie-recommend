"""Vector store interface with Chroma and in-memory implementations.

Everything outside this module talks to `VectorStore`, so Qdrant / pgvector can
be added as another implementation without touching the scoring code.
Ids are TMDB ids; similarity is cosine (embeddings are L2-normalized).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

Metadata = dict[str, str | int | float | bool]
Where = dict[str, str | int | float | bool]


class VectorStore(Protocol):
    def upsert(
        self,
        ids: Sequence[int],
        embeddings: np.ndarray,
        metadatas: Sequence[Metadata],
        documents: Sequence[str],
    ) -> None: ...

    def update_metadata(self, ids: Sequence[int], metadatas: Sequence[Metadata]) -> None: ...

    def get_embeddings(self, ids: Iterable[int]) -> dict[int, np.ndarray]: ...

    def get_metadata(self, ids: Iterable[int] | None = None) -> dict[int, Metadata]: ...

    def query(self, vector: np.ndarray, k: int, where: Where | None = None) -> list[tuple[int, float]]:
        """Top-k (id, cosine similarity), highest first."""
        ...

    def delete(self, ids: Sequence[int]) -> None: ...

    def count(self) -> int: ...


def _clean(meta: Metadata) -> Metadata:
    # Chroma rejects None values; drop them rather than inventing sentinels.
    return {k: v for k, v in meta.items() if v is not None}


class ChromaStore:
    BATCH = 1000

    def __init__(self, path: Path, collection: str = "movies") -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            collection,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # we always supply vectors ourselves
        )

    def upsert(
        self,
        ids: Sequence[int],
        embeddings: np.ndarray,
        metadatas: Sequence[Metadata],
        documents: Sequence[str],
    ) -> None:
        for i in range(0, len(ids), self.BATCH):
            sl = slice(i, i + self.BATCH)
            self._col.upsert(
                ids=[str(x) for x in ids[sl]],
                embeddings=np.asarray(embeddings[sl], dtype=np.float32),
                metadatas=[_clean(m) for m in metadatas[sl]],  # type: ignore[misc]
                documents=list(documents[sl]),
            )

    def update_metadata(self, ids: Sequence[int], metadatas: Sequence[Metadata]) -> None:
        for i in range(0, len(ids), self.BATCH):
            sl = slice(i, i + self.BATCH)
            self._col.update(
                ids=[str(x) for x in ids[sl]],
                metadatas=[_clean(m) for m in metadatas[sl]],  # type: ignore[misc]
            )

    def get_embeddings(self, ids: Iterable[int]) -> dict[int, np.ndarray]:
        id_list = [str(x) for x in ids]
        if not id_list:
            return {}
        res = self._col.get(ids=id_list, include=["embeddings"])  # type: ignore[list-item]
        embs = res.get("embeddings")
        if embs is None:
            return {}
        return {int(i): np.asarray(e, dtype=np.float32) for i, e in zip(res["ids"], embs)}

    def get_metadata(self, ids: Iterable[int] | None = None) -> dict[int, Metadata]:
        kwargs: dict[str, Any] = {"include": ["metadatas"]}
        if ids is not None:
            kwargs["ids"] = [str(x) for x in ids]
            if not kwargs["ids"]:
                return {}
        res = self._col.get(**kwargs)
        metas = res.get("metadatas") or []
        return {int(i): dict(m or {}) for i, m in zip(res["ids"], metas)}

    def query(self, vector: np.ndarray, k: int, where: Where | None = None) -> list[tuple[int, float]]:
        n = self.count()
        if n == 0 or k <= 0:
            return []
        res = self._col.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32)],
            n_results=min(k, n),
            where=where or None,  # type: ignore[arg-type]
            include=["distances"],  # type: ignore[list-item]
        )
        ids = res["ids"][0]
        dists = (res.get("distances") or [[]])[0]
        return [(int(i), 1.0 - float(d)) for i, d in zip(ids, dists)]

    def delete(self, ids: Sequence[int]) -> None:
        if ids:
            self._col.delete(ids=[str(x) for x in ids])

    def count(self) -> int:
        return int(self._col.count())


class InMemoryStore:
    """Brute-force numpy store: used in tests, and a reference implementation."""

    def __init__(self) -> None:
        self._emb: dict[int, np.ndarray] = {}
        self._meta: dict[int, Metadata] = {}
        self._docs: dict[int, str] = {}

    def upsert(
        self,
        ids: Sequence[int],
        embeddings: np.ndarray,
        metadatas: Sequence[Metadata],
        documents: Sequence[str],
    ) -> None:
        for i, e, m, d in zip(ids, embeddings, metadatas, documents):
            self._emb[int(i)] = np.asarray(e, dtype=np.float32)
            self._meta[int(i)] = _clean(m)
            self._docs[int(i)] = d

    def update_metadata(self, ids: Sequence[int], metadatas: Sequence[Metadata]) -> None:
        for i, m in zip(ids, metadatas):
            if int(i) in self._meta:
                self._meta[int(i)] = {**self._meta[int(i)], **_clean(m)}

    def get_embeddings(self, ids: Iterable[int]) -> dict[int, np.ndarray]:
        return {int(i): self._emb[int(i)] for i in ids if int(i) in self._emb}

    def get_metadata(self, ids: Iterable[int] | None = None) -> dict[int, Metadata]:
        keys = self._meta.keys() if ids is None else [int(i) for i in ids]
        return {i: dict(self._meta[i]) for i in keys if i in self._meta}

    def query(self, vector: np.ndarray, k: int, where: Where | None = None) -> list[tuple[int, float]]:
        ids = [
            i for i, m in self._meta.items()
            if not where or all(m.get(key) == val for key, val in where.items())
        ]
        if not ids or k <= 0:
            return []
        mat = np.stack([self._emb[i] for i in ids])
        sims = mat @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-sims)[:k]
        return [(ids[j], float(sims[j])) for j in order]

    def delete(self, ids: Sequence[int]) -> None:
        for i in ids:
            self._emb.pop(int(i), None)
            self._meta.pop(int(i), None)
            self._docs.pop(int(i), None)

    def count(self) -> int:
        return len(self._emb)
