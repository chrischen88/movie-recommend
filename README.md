# Letterboxd Recommender

A local app that recommends films from your Letterboxd export. It blends three scores into one ranked list: a content-based taste profile, embedding similarity from a vector DB, and collaborative filtering.

> **Status:** milestones 1–3 of 8 are done: scaffold, caching, and export parsing (1); TMDB matching, enrichment, and the match-review UI (2); embeddings, the Chroma index, taste vector and clusters, score ②, and a recommendations page (3).

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
make build-index                              # re-run the pipeline (incremental)
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
  pipeline.py    background run: matching → enrichment → candidates → embedding → taste
  embeddings.py  film documents + sentence-transformers embedder
  vectorstore.py VectorStore interface: ChromaStore (default) + InMemoryStore
  taste.py       taste vector, k-means taste clusters, score ②
  recommend.py   assembles the ranked recommendation list
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

## Embeddings and score ②

Each film becomes one document (title, year, genres, director, keywords, overview, and up to 2 TMDB review snippets). It is embedded with `BAAI/bge-small-en-v1.5` and stored in Chroma under `backend/data/chroma`. A film is re-embedded only when its document or the model changes.

- **Taste vector:** Σ (rating − μ) · embedding over your rated films, L2-normalized, so films you disliked push it away.
- **Taste clusters:** k-means over films rated ≥ 4.0. k runs from 3 to 6 and is chosen by cosine silhouette score, but only among values of k where every cluster has at least `TASTE_CLUSTER_MIN_SIZE` films (default 3), so one stray favourite can't become its own cluster.
- **Score ②:** the vector DB is queried once per vector (the taste vector and each cluster centroid) for unseen films. Each candidate's similarity to each vector is z-scored across the candidate pool, and the best one wins. That winning vector is recorded as the cluster the film "matches". Z-scoring matters because raw cosines aren't comparable: the taste vector is a difference direction with small cosines (~0.2), while centroids sit near films (~0.8). Set `EMBEDDING_SCORE_MODE=max` for raw max cosine. The result is converted to a percentile rank from 0 to 1.

Candidates currently come from TMDB's recommendations and similar-film lists for your top-rated films. Films with fewer than 50 votes are skipped.

## Filtering recommendations

The Recommendations page can filter by **minimum rating** (currently the TMDB user score, 0–10), genre, decade, maximum runtime and original language. The same filters work on the API: `GET /api/recommendations?min_rating=7.5&genre=Drama&decade=1990&max_runtime=120&language=ko`.

- Filtering happens before the result limit, so a strict filter still returns a full page.
- A film whose value for a field is unknown fails that filter. For example, a film with no rating never passes a minimum rating.
- A film's score doesn't change when you filter. It's a percentile over all candidates, so filters only hide films.
- The dropdowns only list values present in your candidate pool, with counts, and your filter choices are kept in the URL.
