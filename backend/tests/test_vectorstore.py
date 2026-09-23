"""One contract test, run against every VectorStore implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.vectorstore import ChromaStore, InMemoryStore, VectorStore


@pytest.fixture(params=["memory", "chroma"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> VectorStore:
    return InMemoryStore() if request.param == "memory" else ChromaStore(tmp_path / "chroma")


def unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_contract(store: VectorStore) -> None:
    assert store.count() == 0 and store.query(unit(1, 0, 0), 5) == []
    embs = np.stack([unit(1, 0, 0), unit(1, 1, 0), unit(0, 1, 0), unit(0, 0, 1)])
    metas = [
        {"seen": True, "year": 2016, "title": "A"},
        {"seen": False, "title": "B"},
        {"seen": False, "title": "C"},
        {"seen": False, "title": "D", "year": None},  # None values are dropped
    ]
    store.upsert([1, 2, 3, 4], embs, metas, ["a", "b", "c", "d"])
    assert store.count() == 4

    hits = store.query(unit(1, 0, 0), 3)
    assert [i for i, _ in hits] == [1, 2, 3] or [i for i, _ in hits][:2] == [1, 2]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert hits[1][1] == pytest.approx(np.sqrt(0.5), abs=1e-4)

    unseen = store.query(unit(1, 0, 0), 10, where={"seen": False})
    assert {i for i, _ in unseen} == {2, 3, 4}

    assert store.get_metadata([4]) == {4: {"seen": False, "title": "D"}}
    store.update_metadata([2], [{"seen": True}])
    assert store.get_metadata([2])[2]["seen"] is True
    assert store.get_metadata([2])[2]["title"] == "B"  # update merges
    assert {i for i, _ in store.query(unit(1, 0, 0), 10, where={"seen": False})} == {3, 4}

    got = store.get_embeddings([3, 99])
    assert set(got) == {3}
    np.testing.assert_allclose(got[3], embs[2], rtol=1e-5)
    assert store.get_embeddings([]) == {}

    # upsert replaces
    store.upsert([3], unit(1, 0, 0)[None, :], [{"seen": False, "title": "C2"}], ["c2"])
    assert store.count() == 4
    assert store.get_metadata([3])[3]["title"] == "C2"

    store.delete([1, 2])
    assert store.count() == 2 and set(store.get_metadata()) == {3, 4}


def test_chroma_persists(tmp_path: Path) -> None:
    s1 = ChromaStore(tmp_path / "c")
    s1.upsert([7], unit(1, 2, 3)[None, :], [{"seen": False}], ["x"])
    s2 = ChromaStore(tmp_path / "c")
    assert s2.count() == 1 and 7 in s2.get_embeddings([7])
