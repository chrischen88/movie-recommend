"""Background ingestion pipeline:
matching → enrichment → candidates → embedding → taste model.

Incremental by construction:
  * matching only touches new films (`match_status` None) and earlier TMDB
    errors, so manual fixes and settled results are never redone;
  * enrichment only fetches TMDB ids that have no `Movie` row yet;
  * candidate enrichment likewise only fetches unknown ids;
  * embedding only re-embeds films whose document hash changed;
  * every HTTP response is cached, so even a forced redo is network-free.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import TypeVar

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from app import matching
from app.cache import ApiError, ResponseCache
from app.config import Settings, get_settings
from app.db import Candidate, IngestRun, Movie, UserFilm, get_engine, utcnow
from app.embeddings import Embedder, SentenceTransformerEmbedder, build_document, doc_hash
from app.profile import user_ratings
from app.taste import TasteModel, build_taste_model
from app.tmdb import TmdbClient, fetch_movie
from app.vectorstore import ChromaStore, Metadata, VectorStore

log = logging.getLogger(__name__)

STAGES = ["parsing", "matching", "enrichment", "candidates", "embedding", "taste"]
PROGRESS_FLUSH_SECONDS = 0.5

T = TypeVar("T")
R = TypeVar("R")


class TmdbUnavailable(RuntimeError):
    pass


class Pipeline:
    def __init__(
        self,
        engine: Engine,
        settings: Settings,
        tmdb: TmdbClient | None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        max_workers: int = 8,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.tmdb = tmdb
        self.embedder = embedder
        self.store = store
        self.max_workers = max_workers
        self._lock = threading.Lock()

    # ------------------------------------------------------------ run control

    def try_acquire(self) -> bool:
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        self._lock.release()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def run(self, run_id: int) -> None:
        """Run all post-parse stages for `run_id`. Caller must hold the lock."""
        try:
            self._run(run_id)
        except Exception as exc:  # noqa: BLE001 — record any failure on the run
            log.exception("pipeline run %s failed", run_id)
            self._update(run_id, status="error", message=str(exc), finished_at=utcnow())
        finally:
            self.release()

    def _run(self, run_id: int) -> None:
        stats: dict[str, object] = {}
        messages: list[str] = []
        if self.tmdb is None:
            messages.append("TMDB_API_KEY not set: matching, enrichment and candidates skipped")
            log.warning(messages[-1])
        else:
            stats["matching"] = self.match_pending(run_id)
            stats["enrichment"] = self.enrich_pending(run_id)
            stats["candidates"] = self.generate_candidates(run_id)

        if self.embedder is None or self.store is None:
            messages.append("no embedder/vector store configured: embedding skipped")
            log.warning(messages[-1])
        else:
            stats["embedding"] = self.embed_pending(run_id)
            stats["taste"] = self.build_taste(run_id)

        with Session(self.engine) as s:
            run = s.get(IngestRun, run_id)
            assert run is not None
            run.stats = {**run.stats, **stats}
            run.stage, run.status, run.finished_at = "done", "done", utcnow()
            run.message = "; ".join(messages) or None
            s.add(run)
            s.commit()

    # ------------------------------------------------------------ helpers

    def _update(self, run_id: int, **fields: object) -> None:
        with Session(self.engine) as s:
            run = s.get(IngestRun, run_id)
            if run is None:
                return
            for k, v in fields.items():
                setattr(run, k, v)
            s.add(run)
            s.commit()

    def _parallel(
        self,
        run_id: int,
        stage: str,
        items: list[T],
        work: Callable[[T], R],
        on_done: Callable[[Session, T, R | None, Exception | None], None],
    ) -> None:
        """Run `work` over items in a thread pool; apply results on this thread."""
        self._update(run_id, stage=stage, progress_done=0, progress_total=len(items))
        if not items:
            return
        last_flush = time.monotonic()
        with ThreadPoolExecutor(self.max_workers) as pool, Session(self.engine) as s:
            futures: dict[Future[R], T] = {pool.submit(work, it): it for it in items}
            for done, fut in enumerate(as_completed(futures), start=1):
                item = futures[fut]
                exc = fut.exception()
                if isinstance(exc, ApiError) and exc.status_code == 401:
                    for f in futures:
                        f.cancel()
                    raise TmdbUnavailable("TMDB rejected the API key (HTTP 401)") from exc
                on_done(s, item, None if exc else fut.result(), exc)
                # Commit per item: an open write transaction here would hold
                # SQLite's lock and block workers writing to the HTTP cache.
                s.commit()
                if done == len(items) or time.monotonic() - last_flush > PROGRESS_FLUSH_SECONDS:
                    self._update(run_id, progress_done=done)
                    last_flush = time.monotonic()

    # ------------------------------------------------------------ stages

    def match_pending(self, run_id: int) -> dict[str, int]:
        assert self.tmdb is not None
        tmdb = self.tmdb
        threshold = self.settings.match_low_confidence_threshold
        with Session(self.engine) as s:
            pending = [
                (f.film_key, f.name, f.year)
                for f in s.exec(
                    select(UserFilm).where(
                        col(UserFilm.match_status).is_(None)
                        | (col(UserFilm.match_status) == matching.ERROR)
                    )
                )
            ]
        outcomes = (matching.MATCHED, matching.LOW_CONFIDENCE, matching.UNMATCHED, matching.ERROR)
        stats = dict.fromkeys(outcomes, 0)

        def work(item: tuple[str, str, int | None]) -> matching.MatchResult:
            _, name, year = item
            return matching.match_film(tmdb, name, year, threshold)

        def on_done(
            s: Session,
            item: tuple[str, str, int | None],
            result: matching.MatchResult | None,
            exc: Exception | None,
        ) -> None:
            film = s.get(UserFilm, item[0])
            if film is None:
                return
            if result is None:
                log.error("matching %r failed: %s", film.name, exc)
                film.match_status, film.match_note = matching.ERROR, f"TMDB error: {exc}"
            else:
                film.tmdb_id = result.tmdb_id
                film.match_confidence = result.confidence
                film.match_status = result.status
                film.match_note = result.note
            stats[film.match_status] += 1
            s.add(film)

        self._parallel(run_id, "matching", pending, work, on_done)
        log.info("matching done: %s", stats)
        return stats

    def enrich_pending(self, run_id: int) -> dict[str, int]:
        """Fetch TMDB metadata for the user's matched films."""
        with Session(self.engine) as s:
            wanted = {
                i
                for i in s.exec(select(UserFilm.tmdb_id).where(col(UserFilm.tmdb_id).is_not(None)))
                if i is not None
            }
        return self._enrich_ids(run_id, "enrichment", wanted, mark_user_films=True)

    def _enrich_ids(
        self, run_id: int, stage: str, wanted: set[int], mark_user_films: bool
    ) -> dict[str, int]:
        assert self.tmdb is not None
        tmdb = self.tmdb
        with Session(self.engine) as s:
            have = set(s.exec(select(Movie.tmdb_id)).all())
        pending = sorted(wanted - have)
        stats = {"enriched": 0, "not_found": 0, "errors": 0, "already_had": len(wanted & have)}

        def on_done(s: Session, tmdb_id: int, movie: Movie | None, exc: Exception | None) -> None:
            if movie is not None:
                s.merge(movie)
                stats["enriched"] += 1
                return
            stats["errors" if exc else "not_found"] += 1
            note = f"enrichment failed: {exc}" if exc else f"TMDB id {tmdb_id} not found"
            log.error("tmdb %s: %s", tmdb_id, note)
            if not mark_user_films:
                return
            for film in s.exec(select(UserFilm).where(UserFilm.tmdb_id == tmdb_id)):
                if film.match_status != matching.MANUAL:
                    film.match_status, film.match_note = matching.ERROR, note
                    s.add(film)

        self._parallel(run_id, stage, pending, lambda i: fetch_movie(tmdb, i), on_done)
        log.info("%s done: %s", stage, stats)
        return stats

    def generate_candidates(self, run_id: int) -> dict[str, int]:
        """TMDB recommendations + similar lists, seeded from the user's top-rated films."""
        assert self.tmdb is not None
        tmdb = self.tmdb
        cfg = self.settings
        with Session(self.engine) as s:
            rows = s.exec(
                select(UserFilm.tmdb_id, UserFilm.rating, Movie.vote_count)
                .join(Movie, col(Movie.tmdb_id) == col(UserFilm.tmdb_id))
                .where(col(UserFilm.rating) >= cfg.candidate_seed_min_rating)
            ).all()
            seen = self._seen_ids(s)
        # Highest-rated first; among equals, better-known films make better seeds.
        ranked = sorted(rows, key=lambda r: (-(r[1] or 0), -(r[2] or 0)))
        seeds = list(dict.fromkeys(r[0] for r in ranked if r[0] is not None))[: cfg.candidate_seed_count]

        found: dict[int, set[str]] = {}
        skipped_low_votes = 0

        def work(seed: int) -> list[tuple[str, dict]]:
            return [("recommendations", r) for r in tmdb.recommendations(seed)] + [
                ("similar", r) for r in tmdb.similar(seed)
            ]

        def on_done(_s: Session, seed: int, results: list[tuple[str, dict]] | None, exc: Exception | None) -> None:
            nonlocal skipped_low_votes
            if results is None:
                log.error("candidate lists for seed %s failed: %s", seed, exc)
                return
            for kind, r in results:
                tid = r.get("id")
                if not isinstance(tid, int) or tid in seen or r.get("adult"):
                    continue
                if (r.get("vote_count") or 0) < cfg.candidate_min_votes:
                    skipped_low_votes += 1
                    continue
                found.setdefault(tid, set()).add(f"{kind}:{seed}")

        self._parallel(run_id, "candidates", seeds, work, on_done)

        with Session(self.engine) as s:
            existing = {c.tmdb_id: c for c in s.exec(select(Candidate))}
            new = 0
            for tid, sources in found.items():
                cand = existing.get(tid)
                if cand is None:
                    s.add(Candidate(tmdb_id=tid, sources=sorted(sources)))
                    new += 1
                elif not sources <= set(cand.sources):
                    cand.sources = sorted(set(cand.sources) | sources)
                    cand.updated_at = utcnow()
                    s.add(cand)
            s.commit()
            all_ids = set(s.exec(select(Candidate.tmdb_id)).all())

        enrich = self._enrich_ids(run_id, "candidates", all_ids, mark_user_films=False)
        stats = {
            "seeds": len(seeds),
            "found": len(found),
            "new": new,
            "total": len(all_ids),
            "skipped_low_votes": skipped_low_votes,
            "enriched": enrich["enriched"],
            "enrich_errors": enrich["errors"] + enrich["not_found"],
        }
        log.info("candidates done: %s", stats)
        return stats

    # ------------------------------------------------------------ embeddings

    @staticmethod
    def _seen_ids(s: Session) -> set[int]:
        return {
            i
            for i in s.exec(
                select(UserFilm.tmdb_id).where(
                    col(UserFilm.watched).is_(True), col(UserFilm.tmdb_id).is_not(None)
                )
            )
            if i is not None
        }

    def embed_pending(self, run_id: int) -> dict[str, int]:
        """Embed films whose document changed; keep the `seen` flag in sync."""
        assert self.embedder is not None and self.store is not None
        embedder, store = self.embedder, self.store
        max_reviews = self.settings.doc_max_reviews
        with Session(self.engine) as s:
            movies = list(s.exec(select(Movie)))
            seen = self._seen_ids(s)

        existing = store.get_metadata()
        docs: dict[int, str] = {}
        metas: dict[int, Metadata] = {}
        for m in movies:
            doc = build_document(m, max_reviews)
            docs[m.tmdb_id] = doc
            meta: Metadata = {
                "tmdb_id": m.tmdb_id,
                "title": m.title,
                "genres": "|".join(m.genres),
                "seen": m.tmdb_id in seen,
                "doc_hash": doc_hash(doc, embedder.model_name),
            }
            if m.year is not None:
                meta["year"] = m.year
            metas[m.tmdb_id] = meta

        to_embed = [i for i, meta in metas.items() if existing.get(i, {}).get("doc_hash") != meta["doc_hash"]]
        seen_changed = [
            i for i, meta in metas.items()
            if i not in to_embed and existing.get(i, {}).get("seen") != meta["seen"]
        ]
        self._update(run_id, stage="embedding", progress_done=0, progress_total=len(to_embed))

        batch = self.settings.embedding_batch_size
        for start in range(0, len(to_embed), batch):
            ids = to_embed[start : start + batch]
            vecs = embedder.embed([docs[i] for i in ids])
            store.upsert(ids, vecs, [metas[i] for i in ids], [docs[i] for i in ids])
            self._update(run_id, progress_done=min(start + batch, len(to_embed)))
        if seen_changed:
            store.update_metadata(seen_changed, [{"seen": metas[i]["seen"]} for i in seen_changed])

        stats = {
            "embedded": len(to_embed),
            "unchanged": len(metas) - len(to_embed),
            "seen_flag_updates": len(seen_changed),
            "index_size": store.count(),
        }
        log.info("embedding done: %s", stats)
        return stats

    def build_taste(self, run_id: int) -> dict[str, object]:
        assert self.embedder is not None and self.store is not None
        self._update(run_id, stage="taste", progress_done=0, progress_total=0)
        with Session(self.engine) as s:
            ratings = user_ratings(s)
            genres = {m.tmdb_id: m.genres for m in s.exec(select(Movie))}

        model = build_taste_model(
            ratings,
            self.store.get_embeddings(ratings),
            genres,
            cluster_min_rating=self.settings.taste_cluster_min_rating,
            k_range=self.settings.taste_cluster_k_range,
            min_cluster_size=self.settings.taste_cluster_min_size,
            embedding_model=self.embedder.model_name,
        )
        path = self.settings.taste_model_path
        if model is None:
            path.unlink(missing_ok=True)
            return {"built": False}
        model.save(path)
        return {
            "built": True,
            "rated": model.n_rated,
            "clusters": len(model.clusters),
            "silhouette": model.silhouette,
        }

    def load_taste(self) -> TasteModel | None:
        return TasteModel.load(self.settings.taste_model_path)

    # ------------------------------------------------------------ manual fixes

    def set_manual_match(self, film_key: str, tmdb_id: int) -> tuple[UserFilm, Movie]:
        if self.tmdb is None:
            raise TmdbUnavailable("TMDB_API_KEY not set")
        with Session(self.engine) as s:
            film = s.get(UserFilm, film_key)
            if film is None:
                raise KeyError(film_key)
            movie = s.get(Movie, tmdb_id) or fetch_movie(self.tmdb, tmdb_id)
            if movie is None:
                raise LookupError(f"TMDB id {tmdb_id} not found")
            movie = s.merge(movie)
            film.tmdb_id = tmdb_id
            film.match_status = matching.MANUAL
            film.match_confidence = 1.0
            film.match_note = f"manually set to {movie.title!r} ({movie.year or '?'})"
            s.add(film)
            s.commit()
            s.refresh(film)
            s.refresh(movie)
            log.info("manual match: %s -> %s", film_key, tmdb_id)
            return film, movie

    def set_status(self, film_key: str, status: str, note: str) -> UserFilm:
        with Session(self.engine) as s:
            film = s.get(UserFilm, film_key)
            if film is None:
                raise KeyError(film_key)
            film.match_status, film.match_note = status, note
            if status == matching.MANUAL:
                film.match_confidence = 1.0
            elif status == matching.IGNORED:
                film.tmdb_id, film.match_confidence = None, None
            s.add(film)
            s.commit()
            s.refresh(film)
            return film


def make_tmdb_client(settings: Settings, cache: ResponseCache) -> TmdbClient | None:
    return TmdbClient(settings.tmdb_api_key, cache, settings) if settings.tmdb_api_key else None


@lru_cache
def get_pipeline() -> Pipeline:
    settings = get_settings()
    engine = get_engine()
    return Pipeline(
        engine,
        settings,
        make_tmdb_client(settings, ResponseCache(engine)),
        embedder=SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size),
        store=ChromaStore(settings.chroma_path, settings.vector_collection),
    )
