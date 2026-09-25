"""FastAPI application entry point.

Each visitor works in an in-memory session (app/sessions.py): uploading an
export creates one, and every other endpoint finds it by the `X-Session-Id`
header. Nothing about a user is written to disk; only film data is shared.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import secrets
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlmodel import Session, col, select
from starlette.concurrency import run_in_threadpool

from app import matching
from app.cache import ResponseCache
from app.collab import load_scorer
from app.config import get_settings
from app.db import Movie, get_session, utcnow
from app.demo import demo_export_zip
from app.letterboxd import ExportError, parse_export
from app.library import InvalidFixes, Library, LibraryFilm, applied_fixes, library_from_export
from app.pipeline import STAGES, Pipeline, TmdbUnavailable, get_pipeline
from app.recommend import RatingSource, RecFilters
from app.sessions import AlreadyQueued, Busy, RateLimited, SessionStore, UserSession, get_session_store

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
# httpx logs every request URL at INFO: that's search queries (film titles from
# users' exports) and api_key query params. Keep them out of the logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Resolved through the overrides so tests get their own pipeline and database.
    pipeline = app.dependency_overrides.get(get_pipeline, get_pipeline)()
    pipeline.purge_user_data()
    yield
    app.dependency_overrides.get(get_session_store, get_session_store)().shutdown()


app = FastAPI(title="Letterboxd Recommender", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Open without credentials: platform health checks, and waking a stopped machine.
AUTH_EXEMPT_PATHS = {"/api/health"}


def _authorized(header: str | None, username: str, password: str) -> bool:
    scheme, _, encoded = (header or "").partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        user, _, pw = base64.b64decode(encoded, validate=True).decode().partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    # Compare both, always, so timing doesn't reveal which one was wrong.
    user_ok = secrets.compare_digest(user.encode(), username.encode())
    pw_ok = secrets.compare_digest(pw.encode(), password.encode())
    return user_ok and pw_ok


@app.middleware("http")
async def basic_auth(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    s = get_settings()
    if (
        s.auth_password
        and request.url.path not in AUTH_EXEMPT_PATHS
        and not _authorized(request.headers.get("authorization"), s.auth_username, s.auth_password)
    ):
        return Response(
            "authentication required",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Letterboxd Recommender", charset="UTF-8"'},
        )
    return await call_next(request)


DbDep = Annotated[Session, Depends(get_session)]
PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]
StoreDep = Annotated[SessionStore, Depends(get_session_store)]

SESSION_GONE = "Your session has expired or was deleted. Upload your export again."


def current_session(store: StoreDep, x_session_id: Annotated[str | None, Header()] = None) -> UserSession:
    """The caller's session. 410 (not 404) when it's gone, so the UI can tell an
    expired session from a missing film and send the user back to upload."""
    session = store.get(x_session_id)
    if session is None:
        raise HTTPException(410, SESSION_GONE)
    return session


UserSessionDep = Annotated[UserSession, Depends(current_session)]


def _client_ip(request: Request) -> str:
    # Fly's proxy sets Fly-Client-IP (overwriting any client-sent value).
    return request.headers.get("fly-client-ip") or (request.client.host if request.client else "unknown")


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


# ---------------------------------------------------------------- sessions


@app.post("/api/sessions")
async def create_session(
    request: Request,
    file: UploadFile,
    store: StoreDep,
    pipeline: PipelineDep,
    fixes: Annotated[str, Form()] = "{}",
) -> dict[str, object]:
    """Parse an export into a new in-memory session and queue its processing.
    `fixes` is the browser's saved match fixes (see library_from_export)."""
    settings = store.settings
    _check_upload_rate(request, store)
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "upload too large")
    try:
        fix_map = json.loads(fixes)
    except ValueError:
        fix_map = None
    if not isinstance(fix_map, dict):
        raise HTTPException(400, "fixes must be a JSON object")
    return await _start_session(data, fix_map, store, pipeline)


@app.post("/api/sessions/demo")
async def create_demo_session(request: Request, store: StoreDep, pipeline: PipelineDep) -> dict[str, object]:
    """A session on the built-in sample profile (app/demo.py), for visitors
    without an export. The browser's saved fixes belong to their own export, so
    they aren't applied here."""
    _check_upload_rate(request, store, demo=True)
    return await _start_session(demo_export_zip(), {}, store, pipeline)


def _check_upload_rate(request: Request, store: SessionStore, *, demo: bool = False) -> None:
    try:
        store.check_upload_rate(_client_ip(request), demo=demo)
    except RateLimited as exc:
        raise HTTPException(429, str(exc)) from exc


async def _start_session(
    data: bytes, fix_map: dict[str, Any], store: SessionStore, pipeline: Pipeline
) -> dict[str, object]:
    settings = store.settings
    try:
        export = await run_in_threadpool(parse_export, data, max_csv_bytes=settings.max_uncompressed_csv_bytes)
    except ExportError as exc:
        raise HTTPException(400, str(exc)) from exc
    if len(export.films) > settings.max_export_films:
        raise HTTPException(
            413, f"This export has {len(export.films)} films; the limit is {settings.max_export_films}."
        )
    try:
        lib = library_from_export(export, fix_map)
    except InvalidFixes as exc:
        raise HTTPException(400, str(exc)) from exc

    upload = {
        "films": len(export.films),
        "rated": len(export.rated),
        "watchlist": len(export.watchlist),
        "warnings": len(export.warnings),
        "fixes_applied": applied_fixes(lib),
    }
    try:
        session = store.create(lib, upload)
    except Busy as exc:
        raise HTTPException(503, str(exc)) from exc
    try:
        store.submit(session, pipeline.run_session)
    except Busy as exc:
        store.delete(session.id)
        raise HTTPException(503, str(exc)) from exc
    return {
        "session_id": session.id,
        "expires_in": store.expires_in(session),
        "files_found": export.files_found,
        "stats": upload,
        "warnings": export.warnings,
    }


def _session_out(store: SessionStore, session: UserSession) -> dict[str, object]:
    run = {k: v for k, v in asdict(session.run).items() if k != "cancelled"}
    return {
        "run": run,
        "stages": STAGES,
        "queue_position": store.queue_position(session),
        "expires_in": store.expires_in(session),
    }


@app.get("/api/session")
def get_session_state(session: UserSessionDep, store: StoreDep) -> dict[str, object]:
    return _session_out(store, session)


@app.post("/api/session/reprocess")
def reprocess(session: UserSessionDep, store: StoreDep, pipeline: PipelineDep) -> dict[str, object]:
    """Run the pipeline again, e.g. after fixing matches."""
    try:
        store.submit(session, pipeline.run_session)
    except AlreadyQueued as exc:
        raise HTTPException(409, str(exc)) from exc
    except Busy as exc:
        raise HTTPException(503, str(exc)) from exc
    return _session_out(store, session)


@app.delete("/api/session")
def delete_session(store: StoreDep, x_session_id: Annotated[str | None, Header()] = None) -> dict[str, bool]:
    """Forget the session now instead of when it expires."""
    return {"deleted": bool(x_session_id) and store.delete(x_session_id or "")}


# ---------------------------------------------------------------- matches


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


def _match_row(film: LibraryFilm, movie: Movie | None) -> MatchRow:
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
    session: UserSessionDep, db: DbDep, filter: Literal["review", "all"] = "review"
) -> dict[str, object]:
    films = sorted(session.library.films.values(), key=lambda f: f.name)
    counts = Counter(f.match_status or "pending" for f in films)
    if filter == "review":
        films = [f for f in films if f.match_status in matching.NEEDS_REVIEW]
    ids = {f.tmdb_id for f in films if f.tmdb_id is not None}
    movies = {m.tmdb_id: m for m in db.exec(select(Movie).where(col(Movie.tmdb_id).in_(ids)))}
    return {"counts": dict(counts), "rows": [_match_row(f, movies.get(f.tmdb_id or -1)) for f in films]}


class SetMatchIn(BaseModel):
    film_key: str
    tmdb_ref: str  # numeric id or themoviedb.org URL


class FilmKeyIn(BaseModel):
    film_key: str


def _editable(session: UserSession, film_key: str) -> LibraryFilm:
    if session.run.active:
        raise HTTPException(409, "Wait until processing finishes before fixing matches.")
    try:
        return session.library.film(film_key)
    except KeyError as exc:
        raise HTTPException(404, "film not found") from exc


@app.post("/api/matches/set")
def set_match(body: SetMatchIn, session: UserSessionDep, pipeline: PipelineDep) -> MatchRow:
    _editable(session, body.film_key)
    tmdb_id = matching.parse_tmdb_ref(body.tmdb_ref)
    if tmdb_id is None:
        raise HTTPException(400, "enter a TMDB id or a themoviedb.org/movie/… URL")
    try:
        movie = pipeline.lookup_movie(tmdb_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except TmdbUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    film = session.library.set_manual(
        body.film_key, tmdb_id, f"manually set to {movie.title!r} ({movie.year or '?'})"
    )
    return _match_row(film, movie)


@app.post("/api/matches/accept")
def accept_match(body: FilmKeyIn, session: UserSessionDep, db: DbDep) -> MatchRow:
    film = _editable(session, body.film_key)
    if film.tmdb_id is None:
        raise HTTPException(400, "film has no candidate match to accept")
    session.library.set_status(body.film_key, matching.MANUAL, f"accepted by you (tmdb {film.tmdb_id})")
    return _match_row(film, db.get(Movie, film.tmdb_id))


@app.post("/api/matches/ignore")
def ignore_match(body: FilmKeyIn, session: UserSessionDep) -> MatchRow:
    _editable(session, body.film_key)
    film = session.library.set_status(body.film_key, matching.IGNORED, "ignored by you")
    return _match_row(film, None)


# ---------------------------------------------------------------- recommendations


def _not_ready(lib: Library, pipeline: Pipeline) -> str | None:
    if pipeline.store is None:
        return "No vector store configured."
    if lib.taste is None:
        return "No taste model yet: upload an export and let processing finish."
    return None


@app.get("/api/recommendations")
def get_recommendations(
    session: UserSessionDep,
    db: DbDep,
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
    lib = session.library
    reason = _not_ready(lib, pipeline)
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
    out = pipeline.recommend(lib, db, limit=max(1, min(limit, 200)), filters=filters)
    if out is None:
        return {"ready": False, "message": _not_ready(lib, pipeline), "items": []}
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
def get_metrics(session: UserSessionDep, db: DbDep, pipeline: PipelineDep) -> dict[str, Any]:
    """How well each score and the blend predict the user's own held-out ratings,
    plus the blend weights and data-source usage."""
    cfg = pipeline.settings
    lib = session.library
    blend = lib.blend
    if blend is None:
        return {"ready": False, "message": "No blend yet: upload an export and let processing finish."}
    scorer = load_scorer(cfg.collab_model_path, cfg.collab_min_item_ratings) if cfg.collab_enabled else None
    sources: Counter[str] = Counter()
    for srcs in lib.candidates.values():
        for kind in {src.split(":", 1)[0] for src in srcs}:
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
        "rank_fit_weight": cfg.rank_fit_weight,
        "full": asdict(blend.full) if blend.full else None,
        "partial": asdict(blend.partial) if blend.partial else None,
        "metrics": blend.metrics,
        "collab_model": scorer.model.meta if scorer else None,
        "omdb": {
            "enabled": pipeline.omdb is not None,
            "used_today": ResponseCache(db.get_bind()).count_created_since("omdb", start_of_day),  # type: ignore[arg-type]
            "daily_limit": cfg.omdb_daily_limit,
        },
        "candidates": {"total": len(lib.eligible_ids()), "by_source": dict(sources.most_common())},
    }


@app.get("/api/taste")
def get_taste(session: UserSessionDep, db: DbDep, pipeline: PipelineDep) -> dict[str, Any]:
    lib = session.library
    model = lib.taste
    if model is None:
        return {"ready": False, "message": _not_ready(lib, pipeline), "clusters": []}
    ids = {i for c in model.clusters for i in c.member_ids}
    movies = {m.tmdb_id: m for m in db.exec(select(Movie).where(col(Movie.tmdb_id).in_(ids)))}
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


# ---------------------------------------------------------------- frontend

# Vite puts content-hashed files in assets/, so they can be cached for good.
IMMUTABLE_PREFIX = "assets/"


@app.get("/{path:path}", include_in_schema=False)
def frontend(path: str) -> FileResponse:
    """Serve the built UI: a file from `frontend_dist` if one matches, otherwise
    index.html so client-side routes (/matches, /recommendations…) load the app."""
    dist = get_settings().frontend_dist.resolve()
    index = dist / "index.html"
    if path == "api" or path.startswith("api/") or not index.is_file():
        raise HTTPException(404, "Not Found")
    file = (dist / path).resolve()
    if path and file.is_relative_to(dist) and file.is_file():
        if path.startswith(IMMUTABLE_PREFIX):
            return FileResponse(file, headers={"Cache-Control": "public, max-age=31536000, immutable"})
        return FileResponse(file)
    return FileResponse(index, headers={"Cache-Control": "no-cache"})
