from __future__ import annotations

from app.db import Movie
from app.embeddings import build_document, doc_hash


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
