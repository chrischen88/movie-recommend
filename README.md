# Letterboxd Recommender

A local app that recommends films from your Letterboxd export. It blends three scores into one ranked list: a content-based taste profile, embedding similarity from a vector DB, and collaborative filtering.

> **Status:** milestones 1–2 of 8 are done: scaffold, caching, and export parsing (1); TMDB matching, enrichment, and the match-review UI (2).

## Setup

Requires Python 3.11+ and Node 20+.

```bash
make setup            # creates backend/.venv, installs deps, copies .env.example → .env
# edit .env and add TMDB_API_KEY (OMDB_API_KEY / OPENAI_API_KEY are optional)
make test             # run the backend test suite
```

## Usage

```bash
make sample                                   # writes a synthetic 30-film export ZIP
make ingest EXPORT=sample_letterboxd_export.zip
make ingest EXPORT=~/Downloads/letterboxd-you-2026-09-01-utc.zip
make serve                                    # API on :8000, UI on :5173
```

Re-ingesting a newer export is incremental. Each film is stored with a content hash, so only added or changed films are reprocessed, and films removed from the export are deleted.

## Layout

```
backend/app/
  config.py      tunable weights and params (overridable via env)
  db.py          SQLModel tables (ApiCache, UserFilm, Movie, IngestRun) + auto-migration
  cache.py       SQLite response cache (TTL), rate limiter, retrying HTTP client
  letterboxd.py  export ZIP parser, which merges ratings/watched/diary/watchlist/reviews
  ingest.py      incremental sync into SQLite
  tmdb.py        TMDB client (v3 key or v4 bearer token) + metadata parsing
  matching.py    title/year → TMDB id matching with confidence scores
  pipeline.py    background matching → enrichment run with progress tracking
  main.py        FastAPI app
  cli.py         `python -m app.cli ingest <zip>`
backend/tests/   pytest suite + synthetic export fixture
frontend/        React + Vite + TypeScript + Tailwind
```

## Notes on the Letterboxd export

- Films are merged on a normalized `title|year` key. The `Letterboxd URI` in `diary.csv` and `reviews.csv` identifies the log entry, not the film.
- `ratings.csv` is the source of truth for ratings. When a film has no entry there, the parser falls back to the latest diary rating.
- The `deleted/`, `orphaned/`, `likes/` and `lists/` folders are ignored.
- Bad rows never fail silently. They are skipped, logged, and returned as warnings in the upload response.

## TMDB matching

Each film is searched by title + year. If there are no results, the search is retried with year −1, then year +1, then no year. Each candidate is scored as title similarity × year closeness. A score below `MATCH_LOW_CONFIDENCE_THRESHOLD` (default 0.75) marks the film for review, as does a tie between two popular films with the same title.

On the **Matches** page you can accept a low-confidence match, set a match by TMDB id or `themoviedb.org/movie/…` URL, or ignore a film. These decisions are never overwritten by later runs. Films with a TMDB error are retried on the next upload, or immediately when you click "Retry matching".
