"""Film documents + text embedding (ONNX Runtime behind a small protocol)."""

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


class OnnxEmbedder:
    """BGE sentence embeddings with onnxruntime instead of PyTorch: the model's
    own ONNX export (`onnx/model.onnx` in the Hugging Face repo), CLS pooling and
    L2 normalization, as sentence-transformers does for BGE. Same vectors, but a
    far smaller install, faster to load, and faster on a CPU.

    Lazily loads on first use (the download can take a while; the Docker image
    has it baked in).
    """

    MODEL_FILE = "onnx/model.onnx"

    def __init__(self, model_name: str, batch_size: int = 8, threads: int | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.threads = threads
        self._session = None
        self._tokenizer = None
        self._inputs: list[str] = []
        self._lock = threading.Lock()

    @classmethod
    def download(cls, model_name: str) -> dict[str, str]:
        """Fetch (or find in the Hugging Face cache) the files the embedder needs."""
        from huggingface_hub import hf_hub_download

        return {
            name: hf_hub_download(model_name, name)
            for name in (cls.MODEL_FILE, "tokenizer.json", "sentence_bert_config.json")
        }

    def _load(self):  # type: ignore[no-untyped-def]
        with self._lock:
            if self._session is None:
                import json

                import onnxruntime as ort
                from tokenizers import Tokenizer

                log.info("loading embedding model %s (onnx)", self.model_name)
                files = self.download(self.model_name)
                with open(files["sentence_bert_config.json"]) as f:
                    max_length = int(json.load(f).get("max_seq_length", 512))
                tokenizer = Tokenizer.from_file(files["tokenizer.json"])
                tokenizer.enable_truncation(max_length=max_length)
                tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]") or 0, pad_token="[PAD]")
                opts = ort.SessionOptions()
                if self.threads:
                    opts.intra_op_num_threads = self.threads
                session = ort.InferenceSession(files[self.MODEL_FILE], opts, providers=["CPUExecutionProvider"])
                self._inputs = [i.name for i in session.get_inputs()]
                self._tokenizer, self._session = tokenizer, session
        return self._session, self._tokenizer

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        session, tokenizer = self._load()
        out = np.zeros((len(texts), 0), dtype=np.float32)
        # Batch texts of similar length together: each batch is padded to its
        # longest text, so this avoids wasting compute on padding.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = tokenizer.encode_batch([texts[i] for i in idx])
            feed = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            hidden = session.run(None, {k: v for k, v in feed.items() if k in self._inputs})[0]
            cls = l2_normalize(np.asarray(hidden[:, 0], dtype=np.float32))  # CLS pooling
            if out.shape[1] == 0:
                out = np.zeros((len(texts), cls.shape[1]), dtype=np.float32)
            out[idx] = cls
        return out

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
