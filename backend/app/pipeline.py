"""The recommendation pipeline for one user's library:
matching → enrichment → collab → candidates → embedding → taste → blend → omdb.

The library (films, candidates, models) lives in memory and is dropped with its
session. Everything learned about *films* is shared and persisted, so each user
makes the next one cheaper:
  * every TMDB/OMDb response is cached (`ApiCache`), so a search, list or
    lookup another user already triggered costs no request;
  * enrichment only fetches TMDB ids that have no `Movie` row yet;
  * embedding only embeds films whose document isn't in the shared index;
  * the MovieLens model is trained once and each user is folded in on the fly.
"""

from __future__ import annotations

import logging
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
from app.blend import BlendModel, fit_blend, fixed_model, out_of_fold
from app.cache import ApiError, DailyBudgetExceeded, ResponseCache
from app.collab import MFModel, TrainParams, load_scorer, train_als
from app.config import Settings, get_settings
from app.db import Movie, get_engine, purge_user_data, utcnow
from app.embeddings import Embedder, SentenceTransformerEmbedder, build_document, doc_hash
from app.library import Library
from app.movielens import MovieLensError, download_dataset, is_downloaded, load_dataset
from app.profile import TasteProfile, load_profile
from app.omdb import OmdbClient, OmdbRatings
from app.recommend import RecFilters, RecResult, recommend
from app.sessions import RunState, UserSession
from app.taste import build_taste_model
from app.tmdb import TmdbClient, fetch_movie
from app.vectorstore import ChromaStore, Metadata, VectorStore

log = logging.getLogger(__name__)

STAGES = ["parsing", "matching", "enrichment", "collab", "candidates", "embedding", "taste", "blend", "omdb"]
PROGRESS_FLUSH_SECONDS = 0.5
# Index metadata that described one user's library; stripped at startup.
LEGACY_INDEX_KEYS = ("seen",)

T = TypeVar("T")
R = TypeVar("R")


class ServiceUnavailable(RuntimeError):
    """An API rejected our key (HTTP 401): stop calling it for this run."""


class TmdbUnavailable(ServiceUnavailable):
    pass


class OmdbUnavailable(ServiceUnavailable):
    pass


class RunCancelled(RuntimeError):
    """The session was deleted or expired mid-run."""


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

    # ------------------------------------------------------------ run control

    def purge_user_data(self) -> list[str]:
        """Remove user data that older versions stored (tables, model files, the
        index's `seen` flags). Idempotent; call at startup."""
        removed = purge_user_data(self.engine, self.settings.data_dir)
        if self.store is not None and (n := self.store.strip_metadata_keys(LEGACY_INDEX_KEYS)):
            log.warning("removed per-user flags from %d index entries", n)
            removed.append(f"index flags ({n})")
        return removed

    def run_session(self, session: UserSession) -> None:
        self.run(session.library, session.run)

    def run(self, lib: Library, run: RunState) -> None:
        """Run every stage for `lib`, recording progress and the outcome on `run`."""
        try:
            self._run(lib, run)
        except RunCancelled:
            self._update(run, status="cancelled", finished_at=utcnow())
        except Exception as exc:  # noqa: BLE001 — record any failure on the run
            log.exception("pipeline run failed")
            self._update(run, status="error", message=str(exc), finished_at=utcnow())

    def _run(self, lib: Library, run: RunState) -> None:
        stats: dict[str, object] = {}
        messages: list[str] = []
        if self.tmdb is not None:
            stats["matching"] = self.match_pending(lib, run)
            stats["enrichment"] = self.enrich_pending(lib, run)

        # Before candidates, which use score ③'s top predictions as a source.
        self._check(run)
        try:
            stats["collab"] = self.train_collab(run)
        except MovieLensError as exc:
            # Score ③ is optional: recommendations still work from ① and ②.
            messages.append(f"collaborative score unavailable: {exc}")
            log.warning(messages[-1])
            stats["collab"] = {"skipped": str(exc)}

        if self.tmdb is None:
            messages.append("TMDB_API_KEY not set: matching, enrichment and candidates skipped")
            log.warning(messages[-1])
        else:
            stats["candidates"] = self.generate_candidates(lib, run)

        if self.embedder is None or self.store is None:
            messages.append("no embedder/vector store configured: embedding skipped")
            log.warning(messages[-1])
        else:
            stats["embedding"] = self.embed_pending(lib, run)
            stats["taste"] = self.build_taste(lib, run)
            stats["blend"] = self.fit_blend_model(lib, run)
            self._check(run)
            try:
                stats["omdb"] = self.fetch_omdb(lib, run)
            except OmdbUnavailable as exc:
                # OMDb answers 401 both for a bad key and for "Request limit reached!".
                stats["omdb"] = {"skipped": str(exc)}
                messages.append(f"OMDb ratings skipped: {exc}")
                log.warning(messages[-1])
            if "budget_exhausted" in stats["omdb"]:
                messages.append("OMDb daily request budget reached: some ratings are missing until tomorrow")

        self._update(
            run, stats={**run.stats, **stats}, stage="done", status="done",
            finished_at=utcnow(), message="; ".join(messages) or None,
        )

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _update(run: RunState | None, **fields: object) -> None:
        if run is None:  # CLI `train` runs outside a session
            return
        for k, v in fields.items():
            setattr(run, k, v)

    @staticmethod
    def _check(run: RunState | None) -> None:
        if run is not None and run.cancelled:
            raise RunCancelled()

    def _parallel(
        self,
        run: RunState | None,
        stage: str,
        items: list[T],
        work: Callable[[T], R],
        on_done: Callable[[Session, T, R | None, Exception | None], None],
        unauthorized: type[ServiceUnavailable] = TmdbUnavailable,
    ) -> None:
        """Run `work` over items in a thread pool; apply results on this thread.
        A 401 cancels the rest and raises `unauthorized`; so does the session ending."""
        self._check(run)
        self._update(run, stage=stage, progress_done=0, progress_total=len(items))
        if not items:
            return
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
                if run is not None and run.cancelled:
                    for f in futures:
                        f.cancel()
                    raise RunCancelled()
                on_done(s, item, None if exc else fut.result(), exc)
                # Commit per item: an open write transaction here would hold
                # SQLite's lock and block workers writing to the HTTP cache.
                s.commit()
                self._update(run, progress_done=done)

    # ------------------------------------------------------------ stages

    def match_pending(self, lib: Library, run: RunState | None = None) -> dict[str, int]:
        """Match films that have no result yet (or had a TMDB error). Fixed films
        (manual/ignored) are skipped. Searches are cached, so a title another
        user already matched costs no request."""
        assert self.tmdb is not None
        tmdb = self.tmdb
        threshold = self.settings.match_low_confidence_threshold
        pending = [
            (f.film_key, f.name, f.year)
            for f in lib.films.values()
            if f.match_status is None or f.match_status == matching.ERROR
        ]
        outcomes = (matching.MATCHED, matching.LOW_CONFIDENCE, matching.UNMATCHED, matching.ERROR)
        stats = dict.fromkeys(outcomes, 0)

        def work(item: tuple[str, str, int | None]) -> matching.MatchResult:
            _, name, year = item
            return matching.match_film(tmdb, name, year, threshold)

        def on_done(
            _s: Session,
            item: tuple[str, str, int | None],
            result: matching.MatchResult | None,
            exc: Exception | None,
        ) -> None:
            film = lib.films[item[0]]
            if result is None:
                log.error("a TMDB search failed: %s", exc)
                film.match_status, film.match_note = matching.ERROR, f"TMDB error: {exc}"
            else:
                film.tmdb_id = result.tmdb_id
                film.match_confidence = result.confidence
                film.match_status = result.status
                film.match_note = result.note
            stats[film.match_status] += 1

        self._parallel(run, "matching", pending, work, on_done)
        log.info("matching done: %s", stats)
        return stats

    def enrich_pending(self, lib: Library, run: RunState | None = None) -> dict[str, int]:
        """Fetch TMDB metadata for the user's matched films."""
        return self._enrich_ids(run, "enrichment", lib.own_ids(), lib)

    def _enrich_ids(
        self, run: RunState | None, stage: str, wanted: set[int], lib: Library | None
    ) -> dict[str, int]:
        """Fetch metadata for ids without a `Movie` row. With `lib`, films whose
        lookup failed are marked for review (unless the user set them)."""
        assert self.tmdb is not None
        tmdb = self.tmdb
        with Session(self.engine) as s:
            have = set(s.exec(select(Movie.tmdb_id).where(col(Movie.tmdb_id).in_(wanted))).all())
        pending = sorted(wanted - have)
        stats = {"enriched": 0, "not_found": 0, "errors": 0, "already_had": len(have)}

        def on_done(s: Session, tmdb_id: int, movie: Movie | None, exc: Exception | None) -> None:
            if movie is not None:
                s.merge(movie)
                stats["enriched"] += 1
                return
            stats["errors" if exc else "not_found"] += 1
            note = f"enrichment failed: {exc}" if exc else f"TMDB id {tmdb_id} not found"
            log.error("tmdb %s: %s", tmdb_id, note)
            if lib is None:
                return
            for film in lib.films.values():
                if film.tmdb_id == tmdb_id and film.match_status != matching.MANUAL:
                    film.match_status, film.match_note = matching.ERROR, note

        self._parallel(run, stage, pending, lambda i: fetch_movie(tmdb, i), on_done)
        log.info("%s done: %s", stage, stats)
        return stats

    def generate_candidates(self, lib: Library, run: RunState | None = None) -> dict[str, object]:
        """Build the library's candidate set from several sources:

        - `recommendations:<seed>` / `similar:<seed>`: TMDB lists for the user's
          top-rated films,
        - `discover:genre:<name>` / `discover:lang:<code>`: acclaimed films in the
          genres and languages the taste profile rates highest,
        - `collab`: score ③'s top predictions among films the user hasn't seen.

        Films under `candidate_min_votes` are dropped (after enrichment, for
        sources that don't report vote counts)."""
        assert self.tmdb is not None
        tmdb = self.tmdb
        cfg = self.settings
        ratings = lib.ratings()
        seen = lib.seen_ids()
        liked = {tid: r for tid, r in ratings.items() if r >= cfg.candidate_seed_min_rating}
        with Session(self.engine) as s:
            votes = dict(s.exec(select(Movie.tmdb_id, Movie.vote_count).where(col(Movie.tmdb_id).in_(liked))).all())
            profile = load_profile(s, ratings, cfg.shrinkage_k)
        # Highest-rated first; among equals, better-known films make better seeds.
        ranked = sorted((tid for tid in liked if tid in votes), key=lambda t: (-liked[t], -(votes[t] or 0)))
        seeds = ranked[: cfg.candidate_seed_count]

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
            for source, tid, n_votes in hits:
                if tid in seen:
                    continue
                if n_votes is not None and n_votes < cfg.candidate_min_votes:
                    skipped_low_votes += 1
                    continue
                found.setdefault(tid, set()).add(source)
                kind = source.split(":", 1)[0]
                per_source[kind] = per_source.get(kind, 0) + 1

        self._parallel(run, "candidates", list(jobs), lambda key: jobs[key](), on_done)
        enrich = self._enrich_ids(run, "candidates", set(found), None)

        # Sources without vote counts (collab) are checked now that details are in.
        with Session(self.engine) as s:
            thin = set(s.exec(
                select(Movie.tmdb_id).where(
                    col(Movie.tmdb_id).in_(found), col(Movie.vote_count) < cfg.candidate_min_votes
                )
            ).all())
        lib.candidates = {tid: sorted(srcs) for tid, srcs in found.items() if tid not in thin}
        stats = {
            "seeds": len(seeds),
            "sources": len(jobs),
            "found": len(found),
            "by_source": per_source,
            "failed_sources": len(failed),
            "total": len(lib.candidates),
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

    def embed_pending(self, lib: Library, run: RunState | None = None) -> dict[str, int]:
        """Embed the library's films and candidates that aren't in the shared
        index yet (or whose document changed). The index only holds film data,
        so it's never pruned: another user's films make later runs cheaper."""
        assert self.embedder is not None and self.store is not None
        embedder, store = self.embedder, self.store
        max_reviews = self.settings.doc_max_reviews
        relevant = lib.current_ids()
        with Session(self.engine) as s:
            movies = list(s.exec(select(Movie).where(col(Movie.tmdb_id).in_(relevant))))

        existing = store.get_metadata(relevant)
        docs: dict[int, str] = {}
        metas: dict[int, Metadata] = {}
        for m in movies:
            doc = build_document(m, max_reviews)
            docs[m.tmdb_id] = doc
            meta: Metadata = {
                "tmdb_id": m.tmdb_id,
                "title": m.title,
                "genres": "|".join(m.genres),
                "doc_hash": doc_hash(doc, embedder.model_name),
            }
            if m.year is not None:
                meta["year"] = m.year
            metas[m.tmdb_id] = meta

        to_embed = [i for i, meta in metas.items() if existing.get(i, {}).get("doc_hash") != meta["doc_hash"]]
        self._check(run)
        self._update(run, stage="embedding", progress_done=0, progress_total=len(to_embed))

        batch = self.settings.embedding_batch_size
        for start in range(0, len(to_embed), batch):
            self._check(run)
            ids = to_embed[start : start + batch]
            vecs = embedder.embed([docs[i] for i in ids])
            store.upsert(ids, vecs, [metas[i] for i in ids], [docs[i] for i in ids])
            self._update(run, progress_done=min(start + batch, len(to_embed)))

        stats = {
            "embedded": len(to_embed),
            "unchanged": len(metas) - len(to_embed),
            "index_size": store.count(),
        }
        log.info("embedding done: %s", stats)
        return stats

    def build_taste(self, lib: Library, run: RunState | None = None) -> dict[str, object]:
        assert self.embedder is not None and self.store is not None
        self._check(run)
        self._update(run, stage="taste", progress_done=0, progress_total=0)
        ratings = lib.ratings()
        with Session(self.engine) as s:
            genres = {m.tmdb_id: m.genres for m in s.exec(select(Movie).where(col(Movie.tmdb_id).in_(ratings)))}

        lib.taste = build_taste_model(
            ratings,
            self.store.get_embeddings(ratings),
            genres,
            cluster_min_rating=self.settings.taste_cluster_min_rating,
            k_range=self.settings.taste_cluster_k_range,
            min_cluster_size=self.settings.taste_cluster_min_size,
            embedding_model=self.embedder.model_name,
        )
        if lib.taste is None:
            return {"built": False}
        return {
            "built": True,
            "rated": lib.taste.n_rated,
            "clusters": len(lib.taste.clusters),
            "silhouette": lib.taste.silhouette,
        }

    def recommend(
        self, lib: Library, session: Session, *, limit: int, filters: RecFilters | None = None, mmr: bool = True
    ) -> tuple[RecResult, BlendModel] | None:
        """Everything the recommendation list needs, wired from settings. None
        until the library has a taste model and there's a vector store."""
        if lib.taste is None or self.store is None:
            return None
        cfg = self.settings
        blend = lib.blend or fixed_model(cfg)
        result = recommend(
            session,
            self.store,
            lib,
            lib.taste,
            limit=limit,
            mode=cfg.embedding_score_mode,
            filters=filters,
            profile=load_profile(session, lib.ratings(), cfg.shrinkage_k),
            feature_weights=cfg.feature_weights,
            reasons_per_film=cfg.profile_reasons_per_film,
            min_reason_stars=cfg.profile_min_reason_stars,
            collab=load_scorer(cfg.collab_model_path, cfg.collab_min_item_ratings) if cfg.collab_enabled else None,
            collab_min_user_ratings=cfg.collab_min_user_ratings,
            blend=blend,
            watchlist_boost=cfg.watchlist_boost,
            mmr_lambda=cfg.mmr_lambda if mmr else None,
            min_runtime=cfg.recommend_min_runtime,
            fit_weight=cfg.rank_fit_weight,
        )
        return result, blend

    def fetch_omdb(self, lib: Library, run: RunState | None = None) -> dict[str, object]:
        """IMDb / Rotten Tomatoes / Metacritic for the library's top
        `omdb_shortlist_size` films, skipping any looked up within the cache TTL
        (by any user: the ratings are stored on the shared `Movie` rows)."""
        if self.omdb is None:
            return {"skipped": "OMDB_API_KEY not set"}
        omdb, cfg = self.omdb, self.settings
        with Session(self.engine) as s:
            out = self.recommend(lib, s, limit=cfg.omdb_shortlist_size, mmr=False)
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

        self._parallel(run, "omdb", todo, lambda item: omdb.ratings(item[1]), on_done, OmdbUnavailable)
        if exhausted:
            stats["budget_exhausted"] = exhausted
        stats["already_had"] = len(movies) - len(todo) - stats["no_imdb_id"]
        log.info("omdb done: %s", stats)
        return stats

    def fit_blend_model(self, lib: Library, run: RunState | None = None) -> dict[str, object]:
        """Cross-fit ①②③ on the user's ratings and fit the blend (app/blend.py)."""
        assert self.store is not None and self.embedder is not None
        cfg = self.settings
        self._check(run)
        self._update(run, stage="blend", progress_done=0, progress_total=0)
        ratings = lib.ratings()
        pool = sorted(lib.eligible_ids())
        wanted = set(ratings) | set(pool)
        with Session(self.engine) as s:
            movies = {m.tmdb_id: m for m in s.exec(select(Movie).where(col(Movie.tmdb_id).in_(wanted)))}
        scorer = load_scorer(cfg.collab_model_path, cfg.collab_min_item_ratings) if cfg.collab_enabled else None
        embeddings = self.store.get_embeddings(wanted)
        rows = out_of_fold(ratings, movies, embeddings, pool, scorer, cfg)
        lib.blend = model = fit_blend(rows, cfg)
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

    def train_collab(self, run: RunState | None = None, *, force: bool = False) -> dict[str, object]:
        """Make sure the MovieLens base model exists and matches the settings. It's
        shared: users are folded in per request, so after the first run this is a no-op."""
        cfg = self.settings
        if not cfg.collab_enabled:
            return {"skipped": "disabled (COLLAB_ENABLED=false)"}
        params = self.collab_params()
        fingerprint = params.fingerprint(cfg.movielens_dataset)
        existing = MFModel.load(cfg.collab_model_path)
        if existing is not None and existing.meta.get("fingerprint") == fingerprint and not force:
            return {"trained": False, "val_rmse": existing.meta.get("val_rmse")}

        self._update(run, stage="collab", progress_done=0, progress_total=0)
        if not is_downloaded(cfg.movielens_dir):
            if not cfg.movielens_auto_download:
                raise MovieLensError(
                    f"{cfg.movielens_dataset} is not downloaded and MOVIELENS_AUTO_DOWNLOAD is off (run `make train`)"
                )
            last = [0.0]

            def on_bytes(done: int, total: int) -> None:
                if time.monotonic() - last[0] >= PROGRESS_FLUSH_SECONDS:
                    last[0] = time.monotonic()
                    self._update(run, progress_done=done // 1024, progress_total=total // 1024)

            download_dataset(
                cfg.movielens_dataset, cfg.movielens_root,
                transport=self.movielens_transport, on_progress=on_bytes,
            )

        data = load_dataset(cfg.movielens_dir)
        self._update(run, progress_done=0, progress_total=params.iterations)
        model = train_als(
            data, params, cfg.movielens_dataset,
            on_iteration=lambda done, total: self._update(run, progress_done=done, progress_total=total),
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

    def lookup_movie(self, tmdb_id: int) -> Movie:
        """Metadata for a film the user matched by hand: cached, or fetched and
        cached. Raises LookupError if TMDB has no such film."""
        with Session(self.engine) as s:
            movie = s.get(Movie, tmdb_id)
            if movie is None:
                if self.tmdb is None:
                    raise TmdbUnavailable("TMDB_API_KEY not set")
                fetched = fetch_movie(self.tmdb, tmdb_id)
                if fetched is None:
                    raise LookupError(f"TMDB id {tmdb_id} not found")
                movie = s.merge(fetched)
                s.commit()
                s.refresh(movie)
            s.expunge(movie)
            return movie


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
