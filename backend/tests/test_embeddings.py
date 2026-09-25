from __future__ import annotations

import numpy as np
import pytest

from app.db import Movie
from app.embeddings import OnnxEmbedder, build_document, doc_hash


def test_build_document() -> None:
    m = Movie(
        tmdb_id=1, title="Arrival", year=2016, genres=["Drama", "Science Fiction"],
        directors=["Denis Villeneuve"], keywords=["alien", "linguistics"],
        overview="A linguist is recruited.", reviews=["Stunning.", "Slow but great.", "Third."],
    )
    doc = build_document(m, max_reviews=2)
    assert doc.splitlines()[0] == "Arrival (2016)"
    for part in ("Genres: Drama, Science Fiction", "Directed by Denis Villeneuve",
                 "Keywords: alien, linguistics", "A linguist is recruited.",
                 "Review: Stunning.", "Review: Slow but great."):
        assert part in doc
    assert "Third." not in doc


def test_build_document_sparse_movie() -> None:
    assert build_document(Movie(tmdb_id=2, title="Untitled")) == "Untitled"


def test_doc_hash_depends_on_text_and_model() -> None:
    assert doc_hash("a", "m1") == doc_hash("a", "m1")
    assert doc_hash("a", "m1") != doc_hash("b", "m1")
    assert doc_hash("a", "m1") != doc_hash("a", "m2")


def test_onnx_embedder_keeps_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Texts are batched by length; vectors must still come back in input order.
    Uses the cached model; skipped when it isn't cached (tests stay offline)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    emb = OnnxEmbedder("BAAI/bge-small-en-v1.5", batch_size=2)
    try:
        emb.embed(["warm up"])
    except Exception as exc:  # noqa: BLE001 — model not in the local Hugging Face cache
        pytest.skip(f"embedding model not cached: {exc}")
    texts = ["A much longer text about a linguist and aliens and time.", "x", "Heat (1995)", "", "Arrival"]
    together = emb.embed(texts)
    one_by_one = np.stack([emb.embed([t])[0] for t in texts])
    np.testing.assert_allclose(together, one_by_one, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(together, axis=1), 1.0, atol=1e-5)
    assert emb.embed([]).shape == (0, 0)
