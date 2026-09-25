# Letterboxd Recommender

Local app that recommends films from a Letterboxd export by blending a taste-profile score, embedding similarity (Chroma) and collaborative filtering.

- **Spec:** [docs/SPEC.md](docs/SPEC.md), the original brief. It's built in 8 milestones; stop and summarize after each one.
- **Status, decisions, next steps:** [docs/PROGRESS.md](docs/PROGRESS.md). Read it before starting work, and update it when a milestone or notable change lands.

## Commands

- `make test`: backend pytest suite (fast; fully offline)
- `cd frontend && npm run build`: TypeScript check + build
- `make serve`: API on :8000 and the UI on :5173 (the user runs this; see below)
- `make recommend EXPORT=path.zip`: run an export through the pipeline in the terminal
- `make train` (`FORCE=1` to retrain): download MovieLens if needed and train the score ③ model
- `make fly-deploy`, `make fly-push-data`: deploy to Fly.io and upload `backend/data` ([docs/DEPLOY.md](docs/DEPLOY.md))

## Conventions

- Backend: Python 3.13 venv at `backend/.venv`, FastAPI + SQLModel, type hints throughout. Tunables go in `backend/app/config.py`, secrets in `.env` (never commit either keys or `backend/data/`).
- Every external API call goes through `CachedHttpClient` (`backend/app/cache.py`): cached, throttled, retried. Don't call TMDB/OMDb directly.
- Never store user data on disk: a user's library lives in their in-memory session (`app/library.py`, `app/sessions.py`). Only film data (API cache, `Movie`, embeddings, MovieLens) is persisted and shared.
- Pipeline stages must stay incremental over film data: only fetch or embed films that aren't cached yet.
- Tests never hit the network: use `tests/fake_tmdb.py`, `tests/fake_embedder.py` and `InMemoryStore`.
- Don't kill processes on ports 8000/5173, because those are the user's dev servers. For manual verification, run a separate API on :8765 (with a scratch `DATA_DIR`) and `API_PORT=8765 npm run dev -- --port 5199 --strictPort`.
