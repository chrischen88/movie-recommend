"""Background ingestion pipeline:
matching → enrichment → candidates → embedding → taste model → collaborative model.

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
from datetime import datetime, timedelta, timezone
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import TypeVar

import httpx
from sqlalchemy import Engine
from sqlmodel import Session, col, select

from app import matching
from app.blend import BlendModel, fit_blend, fixed_model, inputs_fingerprint, out_of_fold
from app.cache import ApiError, DailyBudgetExceeded, ResponseCache
from app.collab import MFModel, TrainParams, load_scorer, train_als
from app.config import Settings, get_settings
from app.db import Candidate, IngestRun, Movie, UserFilm, get_engine, utcnow
from app.embeddings import Embedder, SentenceTransformerEmbedder, build_document, doc_hash
from app.movielens import MovieLensError, download_dataset, is_downloaded, load_dataset
from app.profile import TasteProfile, load_profile, user_ratings
from app.omdb import OmdbClient, OmdbRatings
from app.recommend import RecFilters, RecResult, current_film_ids, eligible_ids, recommend
from app.taste import TasteModel, build_taste_model
from app.tmdb import TmdbClient, fetch_movie
from app.vectorstore import ChromaStore, Metadata, VectorStore

log = logging.getLogger(__name__)

STAGES = ["parsing", "matching", "enrichment", "collab", "candidates", "embedding", "taste", "blend", "omdb"]
PROGRESS_FLUSH_SECONDS = 0.5

T = TypeVar("T")
R = TypeVar("R")


class ServiceUnavailable(RuntimeError):
    """An API rejected our key (HTTP 401): stop calling it for this run."""


class TmdbUnavailable(ServiceUnavailable):
    pass


class OmdbUnavailable(ServiceUnavailable):
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
        movielens_transport: httpx.BaseTransport | None = None,
        omdb: OmdbClient | None = None,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.tmdb = tmdb
        self.embedder = embedder
        self.store = store
        self.max_workers = max_workers
        self.movielens_transport = movielens_transport  # tests inject a fake
        self.omdb = omdb
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
        if self.tmdb is not None:
            stats["matching"] = self.match_pending(run_id)
            stats["enrichment"] = self.enrich_pending(run_id)

        # Before candidates, which use score ③'s top predictions as a source.
        try:
            stats["collab"] = self.train_collab(run_id)
        except MovieLensError as exc:
            # Score ③ is optional: recommendations still work from ① and ②.
            messages.append(f"collaborative score unavailable: {exc}")
            log.warning(messages[-1])
            stats["collab"] = {"skipped": str(exc)}

        if self.tmdb is None:
            messages.append("TMDB_API_KEY not set: matching, enrichment and candidates skipped")
            log.warning(messages[-1])
        else:
            stats["candidates"] = self.generate_candidates(run_id)

        if self.embedder is None or self.store is None:
            messages.append("no embedder/vector store configured: embedding skipped")
            log.warning(messages[-1])
        else:
            stats["embedding"] = self.embed_pending(run_id)
            stats["taste"] = self.build_taste(run_id)
            stats["blend"] = self.fit_blend_model(run_id)
            try:
                stats["omdb"] = self.fetch_omdb(run_id)
            except OmdbUnavailable as exc:
                # OMDb answers 401 both for a bad key and for "Request limit reached!".
                stats["omdb"] = {"skipped": str(exc)}
                messages.append(f"OMDb ratings skipped: {exc}")
                log.warning(messages[-1])
            if "budget_exhausted" in stats["omdb"]:
                messages.append("OMDb daily request budget reached: the rest of the shortlist is looked up next run")

        with Session(self.engine) as s:
            run = s.get(IngestRun, run_id)
            assert run is not None
            run.stats = {**run.stats, **stats}
            run.stage, run.status, run.finished_at = "done", "done", utcnow()
            run.message = "; ".join(messages) or None
            s.add(run)
            s.commit()

    # ------------------------------------------------------------ helpers

    def _update(self, run_id: int | None, **fields: object) -> None:
        if run_id is None:  # CLI `train` runs outside an ingest run
            return
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
        unauthorized: type[ServiceUnavailable] = TmdbUnavailable,
    ) -> None:
        """Run `work` over items in a thread pool; apply results on this thread.
        A 401 cancels the rest and raises `unauthorized`."""
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
                    service = "OMDb" if unauthorized is OmdbUnavailable else "TMDB"
                    raise unauthorized(f"{service} rejected the request (HTTP 401): {exc}") from exc
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
        """Build the candidate set from several sources:

        - `recommendations:<seed>` / `similar:<seed>`: TMDB lists for the user's
          top-rated films,
        - `discover:genre:<name>` / `discover:lang:<code>`: acclaimed films in the
          genres and languages the taste profile rates highest,
        - `collab`: score ③'s top predictions among films the user hasn't seen.

        The candidate table is replaced, not appended to: a film stays only while
        a current source still proposes it, so favourites that are gone (or a
        previous account's) stop contributing. Sources whose fetch failed this
        time are kept until the next successful fetch. Films under
        `candidate_min_votes` are dropped (after enrichment, for sources that
        don't report vote counts)."""
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
            profile = load_profile(s, cfg.shrinkage_k)
            ratings = user_ratings(s)
        # Highest-rated first; among equals, better-known films make better seeds.
        ranked = sorted(rows, key=lambda r: (-(r[1] or 0), -(r[2] or 0)))
        seeds = list(dict.fromkeys(r[0] for r in ranked if r[0] is not None))[: cfg.candidate_seed_count]

        # Each job returns (source, tmdb_id, vote_count or None if unknown).
        Hit = tuple[str, int, int | None]
        jobs: dict[str, Callable[[], list[Hit]]] = {}

        def tmdb_hits(source: str, results: list[dict]) -> list[Hit]:
            return [
                (source, r["id"], r.get("vote_count") or 0)
                for r in results
                if isinstance(r.get("id"), int) and not r.get("adult")
            ]

        for seed in seeds:
            jobs[f"seed:{seed}"] = lambda seed=seed: tmdb_hits(
                f"recommendations:{seed}", tmdb.recommendations(seed)
            ) + tmdb_hits(f"similar:{seed}", tmdb.similar(seed))
        jobs.update(self._discover_jobs(profile, tmdb_hits))
        collab_job = self._collab_job(ratings, seen)
        if collab_job is not None:
            jobs["collab"] = collab_job

        found: dict[int, set[str]] = {}
        failed: set[str] = set()
        skipped_low_votes = 0
        per_source: dict[str, int] = {}

        def on_done(_s: Session, key: str, hits: list[Hit] | None, exc: Exception | None) -> None:
            nonlocal skipped_low_votes
            if hits is None:
                log.error("candidate source %s failed: %s", key, exc)
                failed.add(key)
                return
            for source, tid, votes in hits:
                if tid in seen:
                    continue
                if votes is not None and votes < cfg.candidate_min_votes:
                    skipped_low_votes += 1
                    continue
                found.setdefault(tid, set()).add(source)
                kind = source.split(":", 1)[0]
                per_source[kind] = per_source.get(kind, 0) + 1

        self._parallel(run_id, "candidates", list(jobs), lambda key: jobs[key](), on_done)

        with Session(self.engine) as s:
            existing = {c.tmdb_id: c for c in s.exec(select(Candidate))}
            new = removed = 0
            for tid, cand in existing.items():
                kept = {src for src in cand.sources if _source_job(src) in failed}
                sources = found.get(tid, set()) | kept
                if not sources:
                    s.delete(cand)
                    removed += 1
                elif sources != set(cand.sources):
                    cand.sources = sorted(sources)
                    cand.updated_at = utcnow()
                    s.add(cand)
            for tid, sources in found.items():
                if tid not in existing:
                    s.add(Candidate(tmdb_id=tid, sources=sorted(sources)))
                    new += 1
            s.commit()
            all_ids = set(s.exec(select(Candidate.tmdb_id)).all())

        enrich = self._enrich_ids(run_id, "candidates", all_ids, mark_user_films=False)

        # Sources without vote counts (collab) are checked now that details are in.
        with Session(self.engine) as s:
            thin = s.exec(
                select(Candidate)
                .join(Movie, col(Movie.tmdb_id) == col(Candidate.tmdb_id))
                .where(col(Movie.vote_count) < cfg.candidate_min_votes)
            ).all()
            for cand in thin:
                s.delete(cand)
            s.commit()
            total = len(s.exec(select(Candidate.tmdb_id)).all())
        stats = {
            "seeds": len(seeds),
            "sources": len(jobs),
            "found": len(found),
            "by_source": per_source,
            "new": new,
            "removed": removed + len(thin),
            "failed_sources": len(failed),
            "total": total,
            "skipped_low_votes": skipped_low_votes + len(thin),
            "enriched": enrich["enriched"],
            "enrich_errors": enrich["errors"] + enrich["not_found"],
        }
        log.info("candidates done: %s", stats)
        return stats

    def _discover_jobs(
        self, profile: TasteProfile | None, hits: Callable[[str, list[dict]], list[tuple[str, int, int | None]]]
    ) -> dict[str, Callable[[], list[tuple[str, int, int | None]]]]:
        """/discover/movie queries for the best-liked genres and non-English languages."""
        assert self.tmdb is not None
        tmdb, cfg = self.tmdb, self.settings
        if profile is None:
            return {}
        genres = [g for g, st in profile.top("genre", cfg.candidate_discover_genres, min_films=3) if st.value > 0]
        languages = [
            code for code, st in profile.top("language", cfg.candidate_discover_languages + 1, min_films=3)
            if st.value > 0 and code != "en"
        ][: cfg.candidate_discover_languages]
        base = {"sort_by": "vote_average.desc", "vote_count.gte": cfg.candidate_discover_min_votes,
                "include_adult": "false"}
        pages = range(1, cfg.candidate_discover_pages + 1)
        jobs: dict[str, Callable[[], list[tuple[str, int, int | None]]]] = {}

        def genre_job(name: str) -> list[tuple[str, int, int | None]]:
            gid = tmdb.genre_ids().get(name)
            if gid is None:
                log.warning("no TMDB genre id for %r; skipping discover", name)
                return []
            src = f"discover:genre:{name}"
            return [h for p in pages for h in hits(src, tmdb.discover(with_genres=gid, page=p, **base))]

        def lang_job(code: str) -> list[tuple[str, int, int | None]]:
            src = f"discover:lang:{code}"
            return [h for p in pages for h in hits(src, tmdb.discover(with_original_language=code, page=p, **base))]

        for g in genres:
            jobs[f"discover:genre:{g}"] = lambda g=g: genre_job(g)
        for code in languages:
            jobs[f"discover:lang:{code}"] = lambda code=code: lang_job(code)
        return jobs

    def _collab_job(
        self, ratings: dict[int, float], seen: set[int]
    ) -> Callable[[], list[tuple[str, int, int | None]]] | None:
        """Score ③'s top predictions as a candidate source, if a model exists."""
        cfg = self.settings
        if not cfg.collab_enabled or cfg.candidate_collab_count <= 0:
            return None
        scorer = load_scorer(cfg.collab_model_path, cfg.collab_min_item_ratings)
        user = scorer.fold_in(ratings, cfg.collab_min_user_ratings) if scorer else None
        if scorer is None or user is None:
            return None
        exclude = seen | set(ratings)

        def job() -> list[tuple[str, int, int | None]]:
            return [("collab", tid, None) for tid, _ in scorer.top_unseen(user, exclude, cfg.candidate_collab_count, cfg.candidate_collab_min_ratings)]

        return job

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
        """Embed films whose document changed; keep the `seen` flag in sync; drop
        films that no longer belong to this user's library or candidates."""
        assert self.embedder is not None and self.store is not None
        embedder, store = self.embedder, self.store
        max_reviews = self.settings.doc_max_reviews
        with Session(self.engine) as s:
            # Only the user's films and current candidates are indexed; other
            # `movie` rows stay as a metadata cache for if they become relevant again.
            relevant = current_film_ids(s)
            movies = list(s.exec(select(Movie).where(col(Movie.tmdb_id).in_(relevant))))
            seen = self._seen_ids(s)

        existing = store.get_metadata()
        stale = sorted(set(existing) - relevant)
        if stale:
            store.delete(stale)
            log.info("removed %d films from the index that no longer belong to this library", len(stale))
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
            "removed": len(stale),
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

    def discard_user_models(self) -> None:
        """Another account's export replaced the data: drop the old taste and blend
        models so they're never served for the new account, even if this run fails."""
        self.settings.taste_model_path.unlink(missing_ok=True)
        self.settings.blend_model_path.unlink(missing_ok=True)

    def recommend(
        self, session: Session, *, limit: int, filters: RecFilters | None = None, mmr: bool = True
    ) -> tuple[RecResult, BlendModel] | None:
        """Everything the recommendation list needs, wired from settings. None
        until there's a taste model and a vector store."""
        taste = self.load_taste()
        if taste is None or self.store is None:
            return None
        cfg = self.settings
        blend = self.load_blend()
        result = recommend(
            session,
            self.store,
            taste,
            limit=limit,
            mode=cfg.embedding_score_mode,
            filters=filters,
            profile=load_profile(session, cfg.shrinkage_k),
            feature_weights=cfg.feature_weights,
            reasons_per_film=cfg.profile_reasons_per_film,
            min_reason_stars=cfg.profile_min_reason_stars,
            collab=load_scorer(cfg.collab_model_path, cfg.collab_min_item_ratings) if cfg.collab_enabled else None,
            collab_min_user_ratings=cfg.collab_min_user_ratings,
            blend=blend,
            watchlist_boost=cfg.watchlist_boost,
            mmr_lambda=cfg.mmr_lambda if mmr else None,
            min_runtime=cfg.recommend_min_runtime,
        )
        return result, blend

    def fetch_omdb(self, run_id: int | None) -> dict[str, object]:
        """IMDb / Rotten Tomatoes / Metacritic for the top `omdb_shortlist_size`
        films by blended score, skipping any looked up within the cache TTL."""
        if self.omdb is None:
            return {"skipped": "OMDB_API_KEY not set"}
        omdb, cfg = self.omdb, self.settings
        with Session(self.engine) as s:
            out = self.recommend(s, limit=cfg.omdb_shortlist_size, mmr=False)
            shortlist = [r.tmdb_id for r in out[0].items] if out else []
            movies = s.exec(select(Movie).where(col(Movie.tmdb_id).in_(shortlist))).all()
        fresh_after = utcnow() - timedelta(seconds=cfg.cache_ttl_omdb) if cfg.cache_ttl_omdb else None
        todo = [
            (m.tmdb_id, m.imdb_id) for m in movies
            if m.imdb_id and (m.omdb_fetched_at is None
                              or (fresh_after and _as_utc(m.omdb_fetched_at) < fresh_after))
        ]
        stats = {"shortlist": len(shortlist), "no_imdb_id": sum(1 for m in movies if not m.imdb_id),
                 "fetched": 0, "with_ratings": 0, "errors": 0}
        exhausted = 0

        def on_done(sess: Session, item: tuple[int, str], got: OmdbRatings | None, exc: Exception | None) -> None:
            nonlocal exhausted
            if isinstance(exc, DailyBudgetExceeded):
                exhausted += 1
                return
            if got is None:
                stats["errors"] += 1
                log.error("OMDb lookup for %s failed: %s", item[1], exc)
                return
            m = sess.get(Movie, item[0])
            if m is None:
                return
            m.imdb_rating, m.rt_score, m.metacritic = got.imdb_rating, got.rt_score, got.metacritic
            m.omdb_fetched_at = utcnow()
            sess.add(m)
            stats["fetched"] += 1
            stats["with_ratings"] += any(v is not None for v in (got.imdb_rating, got.rt_score, got.metacritic))

        self._parallel(run_id, "omdb", todo, lambda item: omdb.ratings(item[1]), on_done, OmdbUnavailable)
        if exhausted:
            stats["budget_exhausted"] = exhausted
        stats["already_had"] = len(movies) - len(todo) - stats["no_imdb_id"]
        log.info("omdb done: %s", stats)
        return stats

    def load_blend(self) -> BlendModel:
        """The fitted blend, or fixed weights if there isn't one yet."""
        return BlendModel.load(self.settings.blend_model_path) or fixed_model(self.settings)

    def fit_blend_model(self, run_id: int | None) -> dict[str, object]:
        """Cross-fit ①②③ on the user's ratings and fit the blend (app/blend.py).
        Skipped when nothing it depends on has changed."""
        assert self.store is not None and self.embedder is not None
        cfg = self.settings
        self._update(run_id, stage="blend", progress_done=0, progress_total=0)
        with Session(self.engine) as s:
            ratings = user_ratings(s)
            pool = sorted(eligible_ids(s))
            wanted = set(ratings) | set(pool)
            movies = {m.tmdb_id: m for m in s.exec(select(Movie).where(col(Movie.tmdb_id).in_(wanted)))}
        scorer = load_scorer(cfg.collab_model_path, cfg.collab_min_item_ratings) if cfg.collab_enabled else None
        fingerprint = inputs_fingerprint(
            ratings, pool, cfg,
            [self.embedder.model_name, scorer.model.meta.get("fingerprint", "") if scorer else ""],
        )
        existing = BlendModel.load(cfg.blend_model_path)
        if existing is not None and existing.fingerprint == fingerprint:
            return {"fitted": False, "mode": existing.mode}
        embeddings = self.store.get_embeddings(wanted)
        rows = out_of_fold(ratings, movies, embeddings, pool, scorer, cfg)
        model = fit_blend(rows, cfg)
        model.fingerprint = fingerprint
        model.save(cfg.blend_model_path)
        learned = model.metrics.get("learned_blend", {})
        return {
            "fitted": True, "mode": model.mode, "ratings": model.n_ratings,
            "with_collab": model.n_with_collab, "rmse": learned.get("rmse"),
            "baseline_rmse": model.metrics.get("baseline_mean", {}).get("rmse"),
        }

    def collab_params(self) -> TrainParams:
        cfg = self.settings
        return TrainParams(
            factors=cfg.collab_factors,
            iterations=cfg.collab_iterations,
            reg=cfg.collab_reg,
            val_fraction=cfg.collab_val_fraction,
            seed=cfg.collab_seed,
        )

    def train_collab(self, run_id: int | None = None, *, force: bool = False) -> dict[str, object]:
        """Make sure the MovieLens base model exists and matches the settings.
        The user is folded in per request, so after the first run this is a no-op."""
        cfg = self.settings
        if not cfg.collab_enabled:
            return {"skipped": "disabled (COLLAB_ENABLED=false)"}
        params = self.collab_params()
        fingerprint = params.fingerprint(cfg.movielens_dataset)
        existing = MFModel.load(cfg.collab_model_path)
        if existing is not None and existing.meta.get("fingerprint") == fingerprint and not force:
            return {"trained": False, "val_rmse": existing.meta.get("val_rmse")}

        self._update(run_id, stage="collab", progress_done=0, progress_total=0)
        if not is_downloaded(cfg.movielens_dir):
            if not cfg.movielens_auto_download:
                raise MovieLensError(
                    f"{cfg.movielens_dataset} is not downloaded and MOVIELENS_AUTO_DOWNLOAD is off (run `make train`)"
                )
            last = [0.0]

            def on_bytes(done: int, total: int) -> None:
                if time.monotonic() - last[0] >= PROGRESS_FLUSH_SECONDS:
                    last[0] = time.monotonic()
                    self._update(run_id, progress_done=done // 1024, progress_total=total // 1024)

            download_dataset(
                cfg.movielens_dataset, cfg.movielens_root,
                transport=self.movielens_transport, on_progress=on_bytes,
            )

        data = load_dataset(cfg.movielens_dir)
        self._update(run_id, progress_done=0, progress_total=params.iterations)
        model = train_als(
            data, params, cfg.movielens_dataset,
            on_iteration=lambda done, total: self._update(run_id, progress_done=done, progress_total=total),
        )
        model.save(cfg.collab_model_path)
        return {
            "trained": True,
            "dataset": cfg.movielens_dataset,
            "val_rmse": model.meta["val_rmse"],
            "val_rmse_global_mean": model.meta["val_rmse_global_mean"],
            "seconds": model.meta["train_seconds"],
        }

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


def _source_job(source: str) -> str:
    """The fetch that produced a source: "similar:603" → "seed:603"; discover and
    collab sources are their own job."""
    kind, _, rest = source.partition(":")
    return f"seed:{rest}" if kind in ("recommendations", "similar") else source


def _as_utc(dt: datetime) -> datetime:
    # SQLite drops tzinfo on round-trip; values are always stored as UTC.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def make_omdb_client(settings: Settings, cache: ResponseCache) -> OmdbClient | None:
    return OmdbClient(settings.omdb_api_key, cache, settings) if settings.omdb_api_key else None


def make_tmdb_client(settings: Settings, cache: ResponseCache) -> TmdbClient | None:
    return TmdbClient(settings.tmdb_api_key, cache, settings) if settings.tmdb_api_key else None


@lru_cache
def get_pipeline() -> Pipeline:
    settings = get_settings()
    engine = get_engine()
    cache = ResponseCache(engine)
    return Pipeline(
        engine,
        settings,
        make_tmdb_client(settings, cache),
        omdb=make_omdb_client(settings, cache),
        embedder=SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size),
        store=ChromaStore(settings.chroma_path, settings.vector_collection),
    )
