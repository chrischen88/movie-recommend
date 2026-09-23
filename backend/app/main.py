"""FastAPI application entry point."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, col, select

from app import matching
from app.blend import BlendModel
from app.cache import ResponseCache
from app.collab import load_scorer
from app.config import get_settings
from app.db import Candidate, IngestRun, Movie, UserFilm, get_session, utcnow
from app.ingest import sync_export
from app.letterboxd import ExportError, parse_export
from app.pipeline import STAGES, Pipeline, TmdbUnavailable, get_pipeline
from app.recommend import RatingSource, RecFilters, eligible_ids

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("app")

app = FastAPI(title="Letterboxd Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SessionDep = Annotated[Session, Depends(get_session)]
PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]


@app.get("/api/health")
def health() -> dict[str, object]:
    s = get_settings()
    return {
        "status": "ok",
        "features": {
            "tmdb": bool(s.tmdb_api_key),
            "omdb": bool(s.omdb_api_key),
            "openai": bool(s.openai_api_key),
        },
    }


# ---------------------------------------------------------------- ingestion


class RunOut(BaseModel):
    run: IngestRun
    stages: list[str]


@app.post("/api/upload")
async def upload_export(
    file: UploadFile, session: SessionDep, pipeline: PipelineDep, background: BackgroundTasks
) -> dict[str, object]:
    settings = get_settings()
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "upload too large")
    if not pipeline.try_acquire():
        raise HTTPException(409, "an ingestion run is already in progress")

    try:
        run = IngestRun(stage="parsing")
        session.add(run)
        session.commit()
        session.refresh(run)
        assert run.id is not None

        try:
            export = parse_export(data, max_csv_bytes=settings.max_uncompressed_csv_bytes)
        except ExportError as exc:
            run.status, run.message, run.finished_at = "error", str(exc), utcnow()
            session.add(run)
            session.commit()
            raise HTTPException(400, str(exc)) from exc

        result = sync_export(session, export)
        if result.reset:
            pipeline.discard_user_models()
        run.stats = {
            **result.summary(),
            "account": result.account,
            "films": len(export.films),
            "rated": len(export.rated),
            "watchlist": len(export.watchlist),
            "warnings": len(export.warnings),
        }
        session.add(run)
        session.commit()
    except BaseException:
        pipeline.release()
        raise

    background.add_task(pipeline.run, run.id)
    return {
        "run_id": run.id,
        "files_found": export.files_found,
        "stats": run.stats,
        "warnings": export.warnings,
    }


@app.post("/api/ingest/resume")
def resume_ingest(
    session: SessionDep, pipeline: PipelineDep, background: BackgroundTasks
) -> dict[str, int]:
    """Re-run the pipeline on already-ingested films (e.g. after adding a key)."""
    if not pipeline.try_acquire():
        raise HTTPException(409, "an ingestion run is already in progress")
    try:
        run = IngestRun(stage="matching")
        session.add(run)
        session.commit()
        session.refresh(run)
        assert run.id is not None
    except BaseException:
        pipeline.release()
        raise
    background.add_task(pipeline.run, run.id)
    return {"run_id": run.id}


@app.get("/api/ingest/latest")
def latest_run(session: SessionDep) -> RunOut | None:
    run = session.exec(select(IngestRun).order_by(col(IngestRun.id).desc())).first()
    return RunOut(run=run, stages=STAGES) if run else None


@app.get("/api/ingest/{run_id}")
def get_run(run_id: int, session: SessionDep) -> RunOut:
    run = session.get(IngestRun, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return RunOut(run=run, stages=STAGES)


# ---------------------------------------------------------------- films & matches


@app.get("/api/films")
def list_films(session: SessionDep) -> list[UserFilm]:
    return list(session.exec(select(UserFilm).order_by(UserFilm.name)).all())


class MovieBrief(BaseModel):
    tmdb_id: int
    title: str
    year: int | None
    poster_path: str | None
    directors: list[str]
    overview: str | None


class MatchRow(BaseModel):
    film_key: str
    name: str
    year: int | None
    rating: float | None
    status: str | None
    confidence: float | None
    note: str | None
    tmdb_id: int | None
    movie: MovieBrief | None


def _match_row(film: UserFilm, movie: Movie | None) -> MatchRow:
    return MatchRow(
        film_key=film.film_key,
        name=film.name,
        year=film.year,
        rating=film.rating,
        status=film.match_status,
        confidence=film.match_confidence,
        note=film.match_note,
        tmdb_id=film.tmdb_id,
        movie=MovieBrief.model_validate(movie, from_attributes=True) if movie else None,
    )


@app.get("/api/matches")
def list_matches(
    session: SessionDep, filter: Literal["review", "all"] = "review"
) -> dict[str, object]:
    stmt = select(UserFilm, Movie).join(
        Movie, col(UserFilm.tmdb_id) == col(Movie.tmdb_id), isouter=True
    )
    if filter == "review":
        stmt = stmt.where(col(UserFilm.match_status).in_(matching.NEEDS_REVIEW))
    rows = session.exec(stmt.order_by(UserFilm.name)).all()

    counts: dict[str, int] = {}
    for status in session.exec(select(UserFilm.match_status)).all():
        key = status or "pending"
        counts[key] = counts.get(key, 0) + 1
    return {"counts": counts, "rows": [_match_row(f, m) for f, m in rows]}


class SetMatchIn(BaseModel):
    film_key: str
    tmdb_ref: str  # numeric id or themoviedb.org URL


class FilmKeyIn(BaseModel):
    film_key: str


def _row_for(session: SessionDep, film_key: str) -> MatchRow:
    film = session.get(UserFilm, film_key)
    assert film is not None
    movie = session.get(Movie, film.tmdb_id) if film.tmdb_id else None
    return _match_row(film, movie)


@app.post("/api/matches/set")
def set_match(body: SetMatchIn, session: SessionDep, pipeline: PipelineDep) -> MatchRow:
    tmdb_id = matching.parse_tmdb_ref(body.tmdb_ref)
    if tmdb_id is None:
        raise HTTPException(400, "enter a TMDB id or a themoviedb.org/movie/… URL")
    try:
        pipeline.set_manual_match(body.film_key, tmdb_id)
    except KeyError as exc:
        raise HTTPException(404, "film not found") from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except TmdbUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return _row_for(session, body.film_key)


@app.post("/api/matches/accept")
def accept_match(body: FilmKeyIn, session: SessionDep, pipeline: PipelineDep) -> MatchRow:
    film = session.get(UserFilm, body.film_key)
    if film is None:
        raise HTTPException(404, "film not found")
    if film.tmdb_id is None:
        raise HTTPException(400, "film has no candidate match to accept")
    pipeline.set_status(body.film_key, matching.MANUAL, f"accepted by user (tmdb {film.tmdb_id})")
    session.expire_all()
    return _row_for(session, body.film_key)


@app.post("/api/matches/ignore")
def ignore_match(body: FilmKeyIn, session: SessionDep, pipeline: PipelineDep) -> MatchRow:
    try:
        pipeline.set_status(body.film_key, matching.IGNORED, "ignored by user")
    except KeyError as exc:
        raise HTTPException(404, "film not found") from exc
    session.expire_all()
    return _row_for(session, body.film_key)


# ---------------------------------------------------------------- recommendations


def _not_ready(pipeline: Pipeline) -> str | None:
    if pipeline.store is None:
        return "No vector store configured."
    if pipeline.load_taste() is None:
        return "No taste model yet: upload an export and let processing finish."
    return None


@app.get("/api/recommendations")
def get_recommendations(
    session: SessionDep,
    pipeline: PipelineDep,
    limit: int = 40,
    min_rating: Annotated[float | None, Query(ge=0, le=100)] = None,
    rating_source: RatingSource = "tmdb",
    hide_low_quality: bool = False,
    genre: Annotated[list[str] | None, Query()] = None,
    decade: Annotated[int | None, Query(ge=1870, le=2100)] = None,
    max_runtime: Annotated[int | None, Query(gt=0)] = None,
    language: str | None = None,
) -> dict[str, Any]:
    reason = _not_ready(pipeline)
    if reason:
        return {"ready": False, "message": reason, "items": []}
    if min_rating is not None and rating_source in ("tmdb", "imdb") and min_rating > 10:
        raise HTTPException(422, f"{rating_source} ratings are 0–10")
    cfg = pipeline.settings
    filters = RecFilters(
        min_rating=min_rating,
        rating_source=rating_source,
        hide_low_quality=hide_low_quality,
        quality_rt=cfg.quality_floor_tomatometer,
        quality_imdb=cfg.quality_floor_imdb,
        genres=tuple(genre or ()),
        decade=decade,
        max_runtime=max_runtime,
        language=language or None,
    )
    out = pipeline.recommend(session, limit=max(1, min(limit, 200)), filters=filters)
    if out is None:
        return {"ready": False, "message": _not_ready(pipeline), "items": []}
    result, blend = out
    return {
        "ready": True,
        "message": None,
        "items": [asdict(r) for r in result.items],
        "total": result.total,
        "matching": result.matching,
        "facets": asdict(result.facets),
        "collab_films": result.collab_films,
        "blend_mode": blend.mode,
    }


@app.get("/api/metrics")
def get_metrics(session: SessionDep, pipeline: PipelineDep) -> dict[str, Any]:
    """How well each score and the blend predict the user's own held-out ratings,
    plus the blend weights and data-source usage."""
    cfg = pipeline.settings
    blend = BlendModel.load(cfg.blend_model_path)
    if blend is None:
        return {"ready": False, "message": "No blend yet: upload an export and let processing finish."}
    scorer = load_scorer(cfg.collab_model_path, cfg.collab_min_item_ratings) if cfg.collab_enabled else None
    sources: Counter[str] = Counter()
    for cand in session.exec(select(Candidate)):
        for kind in {src.split(":", 1)[0] for src in cand.sources}:
            sources[kind] += 1
    start_of_day = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        "ready": True,
        "mode": blend.mode,
        "min_ratings_for_learning": cfg.blend_min_ratings_for_learning,
        "n_ratings": blend.n_ratings,
        "n_with_collab": blend.n_with_collab,
        "trained_at": blend.trained_at,
        "fixed_weights": blend.fixed_weights,
        "full": asdict(blend.full) if blend.full else None,
        "partial": asdict(blend.partial) if blend.partial else None,
        "metrics": blend.metrics,
        "collab_model": scorer.model.meta if scorer else None,
        "omdb": {
            "enabled": pipeline.omdb is not None,
            "used_today": ResponseCache(session.get_bind()).count_created_since("omdb", start_of_day),  # type: ignore[arg-type]
            "daily_limit": cfg.omdb_daily_limit,
        },
        "candidates": {"total": len(eligible_ids(session)), "by_source": dict(sources.most_common())},
    }


@app.get("/api/taste")
def get_taste(session: SessionDep, pipeline: PipelineDep) -> dict[str, Any]:
    model = pipeline.load_taste()
    if model is None:
        return {"ready": False, "message": _not_ready(pipeline), "clusters": []}
    ids = {i for c in model.clusters for i in c.member_ids}
    movies = {m.tmdb_id: m for m in session.exec(select(Movie).where(col(Movie.tmdb_id).in_(ids)))}
    return {
        "ready": True,
        "mean_rating": model.mean_rating,
        "n_rated": model.n_rated,
        "silhouette": model.silhouette,
        "embedding_model": model.embedding_model,
        "clusters": [
            {
                "id": c.id,
                "source": c.source,
                "label": c.label,
                "size": len(c.member_ids),
                "examples": [
                    {"tmdb_id": i, "title": movies[i].title, "year": movies[i].year,
                     "poster_path": movies[i].poster_path}
                    for i in c.member_ids[:6]
                    if i in movies
                ],
            }
            for c in model.clusters
        ],
    }
