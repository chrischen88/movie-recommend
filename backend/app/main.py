"""FastAPI application entry point."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, col, select

from app import matching
from app.config import get_settings
from app.db import IngestRun, Movie, UserFilm, get_session, utcnow
from app.ingest import sync_export
from app.letterboxd import ExportError, parse_export
from app.pipeline import STAGES, Pipeline, TmdbUnavailable, get_pipeline
from app.profile import load_profile
from app.recommend import RecFilters, recommend

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
        run.stats = {
            **result.summary(),
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
    min_rating: Annotated[float | None, Query(ge=0, le=10)] = None,
    genre: Annotated[list[str] | None, Query()] = None,
    decade: Annotated[int | None, Query(ge=1870, le=2100)] = None,
    max_runtime: Annotated[int | None, Query(gt=0)] = None,
    language: str | None = None,
) -> dict[str, Any]:
    reason = _not_ready(pipeline)
    model = pipeline.load_taste()
    if reason or model is None or pipeline.store is None:
        return {"ready": False, "message": reason, "items": []}
    filters = RecFilters(
        min_rating=min_rating,
        genres=tuple(genre or ()),
        decade=decade,
        max_runtime=max_runtime,
        language=language or None,
    )
    result = recommend(
        session,
        pipeline.store,
        model,
        limit=max(1, min(limit, 200)),
        k_per_vector=pipeline.settings.vector_query_k,
        mode=pipeline.settings.embedding_score_mode,
        filters=filters,
        profile=load_profile(session, pipeline.settings.shrinkage_k),
        feature_weights=pipeline.settings.feature_weights,
        reasons_per_film=pipeline.settings.profile_reasons_per_film,
        min_reason_stars=pipeline.settings.profile_min_reason_stars,
    )
    return {
        "ready": True,
        "message": None,
        "items": [asdict(r) for r in result.items],
        "total": result.total,
        "matching": result.matching,
        "facets": asdict(result.facets),
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
