# Progress & Roadmap

Tracks where the build stands against [SPEC.md](SPEC.md). Update this file at the end of every milestone.

_Last updated: 2026-09-23_

## Status

| # | Milestone | Status |
|---|---|---|
| 1 | Scaffold, config, caching, Letterboxd ZIP parsing, tests | ✅ Done |
| 2 | TMDB matching + enrichment, match-review UI | ✅ Done |
| 3 | Embeddings + Chroma, taste vector/clusters, score ②, basic recs page | ✅ Done |
| — | Recommendation filters (requested outside the plan) | ✅ Done |
| 4 | Score ① (taste profile) with explanations | ⏭ Next |
| 5 | MovieLens ingestion + score ③ | Not started |
| 6 | Candidate gen (discover), OMDb, learned blend, MMR, metrics page | Not started |
| 7 | OpenAI layer: re-rank, explanations, natural-language requests | Not started |
| 8 | UI polish, taste profile page, feedback loop, README | Not started |

Tests: 118 passing (`make test`). The frontend type-checks and builds (`cd frontend && npm run build`).

## What exists (by module)

- `backend/app/letterboxd.py`: parses the ZIP and merges the CSVs on a normalized `title|year` key.
- `backend/app/cache.py`: SQLite response cache with TTL, rate limiter, retry/backoff client, and an optional daily budget (for OMDb).
- `backend/app/ingest.py`: incremental sync into SQLite using a content hash per film.
- `backend/app/tmdb.py`, `matching.py`: TMDB client and title/year matching with confidence scores.
- `backend/app/pipeline.py`: background run: matching → enrichment → candidates → embedding → taste.
- `backend/app/embeddings.py`, `vectorstore.py`, `taste.py`: film documents, the `VectorStore` interface (Chroma + in-memory), the taste vector, clusters, and score ②.
- `backend/app/recommend.py`: builds the ranked list, applies `RecFilters` (min TMDB rating, genre, decade, max runtime, language) and computes facets.
- `backend/app/main.py`: FastAPI. Routes: `/api/health`, `/api/upload`, `/api/ingest/{id|latest|resume}`, `/api/films`, `/api/matches[/set|/accept|/ignore]`, `/api/recommendations`, `/api/taste`.
- `frontend/`: Upload (progress stepper), Matches (review/fix), and Recommendations (poster grid, filter bar, cluster chips) pages.

## Decisions & deviations from the spec

These are deliberate, so don't "fix" them back without reason.

1. **Films are keyed on `title|year`, not the Letterboxd URI.** In real exports, the URI in `diary.csv` and `reviews.csv` points to the log entry, not the film. `ratings.csv` is the source of truth for ratings; the latest diary rating is the fallback.
2. **Score ② is a z-scored max, not a raw max cosine** (`EMBEDDING_SCORE_MODE=zmax`). Raw cosines aren't comparable across vectors: the taste vector is a difference direction (~0.2 cosine) and centroids sit near films (~0.8), so a raw max would almost never pick the taste vector. Each vector's similarities are z-scored across the candidate pool before taking the max. `max` restores the spec's behaviour.
3. **Taste clusters have a minimum size of 3** (`TASTE_CLUSTER_MIN_SIZE`). A k is only eligible if every cluster has ≥ 3 films. Without this, a real run chose k=6 with single-film clusters whose near-duplicates flooded the results.
4. **A basic candidate pool was pulled into M3.** It uses TMDB recommendations and similar lists from the top 25 films rated ≥ 4.0, skipping films with fewer than 50 votes (`CANDIDATE_MIN_VOTES`). M6 adds `/discover/movie`.
5. **Match statuses:** matched / low_confidence / unmatched / manual / error / ignored. Manual and ignored decisions are never overwritten. An ignored film has its `tmdb_id` cleared so it can't leak into training. Errored films are retried on the next run.
6. **Ambiguous matches** (a near-tie between two films that both have real vote counts) are pushed into review. A same-name obscure film is simply outvoted.
7. **Collaborative filtering (M5, planned):** use a numpy/scipy implementation, because `implicit` may lack Python 3.13 wheels.
8. **The DB auto-migrates by adding new columns** (`db._add_missing_columns`), so new model fields don't require deleting the DB.

## Open items / known issues

- [ ] **Fold `/reviews` into the details call** (`append_to_response=credits,keywords,external_ids,reviews`). Reviews are currently a separate request per film, so this halves first-run TMDB calls (~1,181 → ~630 on the sample export). Changing the request format changes cache keys, so responses cached under the old format won't be reused.
- [ ] Gitignore `frontend/tsconfig.tsbuildinfo` (a build artifact) and `git rm --cached` it.
- [ ] Commit M3 + filters + the proxy-port change. Nothing after the init commit is committed yet.
- [ ] Optional: a configurable default minimum rating for recommendations.
- [ ] Weak picks (e.g. *Mission to Mars*) get through while score ② is the only signal. M4 and M6 should fix this.
- [ ] A harmless joblib/loky "leaked semaphore" warning appears when the server is killed after a pipeline run.

## Plan for remaining milestones

### M4: Score ① (taste profile)
- New `app/profile.py`: μ = mean rating. For each feature value (director, genre, actor [top 5 cast], keyword, decade, language, country), take the mean of (rating − μ), shrunk toward 0: `sum / (n + SHRINKAGE_K)`.
- A candidate's score is the sum of `FEATURE_WEIGHTS[type] × value` over its features, converted to a percentile rank from 0 to 1.
- Keep the top contributions for explanations ("Director Denis Villeneuve: +0.9").
- Uses only local data (the `movie` table), so no API calls.
- Show a ① bar and the explanation lines on the cards. Initially the displayed score = mean of ① and ② (the real blend comes in M6).

### M5: MovieLens + score ③
- Download `ml-latest-small` (configurable `ml-32m`) into `backend/data/movielens/`, then map `links.csv` movieId → tmdbId.
- Add the user as a new row, then run explicit matrix factorization (ALS or SVD in numpy/scipy) with bias terms.
- Predict ratings for candidates. Candidates not in MovieLens get a null score, and the blend re-weights the other two scores.
- Add a `make train` target.

### M6: Blending, OMDb, MMR, metrics
- Add `/discover/movie` as another candidate source.
- OMDb only for the final shortlist, via the IMDb id from `external_ids`. `CachedHttpClient` already supports `daily_limit=1000`.
- Hold out 20%, compute ①②③ using the other 80%, then fit Ridge regression (+ optional log vote count) with 5-fold CV. With fewer than 50 ratings, use fixed weights 0.3/0.4/0.3.
- Add the watchlist boost, the quality floor (RT ≥ 60 or IMDb ≥ 6.5), and MMR (λ≈0.7) on the embeddings.
- Add a `/api/metrics` endpoint and page (RMSE + Spearman per method and for the blend).
- Add IMDb/RT/Metacritic to the rating filter.

### M7: OpenAI layer
- An `LLMClient` interface. Re-rank the top 30 → 20, returning a one-sentence "why" per film via Structured Outputs.
- Validate ids, limit moves to ±`LLM_MAX_POSITION_SHIFT`, and fall back to the blended order.
- Natural-language requests → filters + a query string, embedded with `embed_query` (BGE query prefix) and blended with the taste vector.
- Cache responses in SQLite by input hash, log tokens, and show cost on the metrics page. Use template explanations if the key is missing or a call fails.

### M8: Polish and feedback
- Taste profile page (top/bottom directors/genres/actors, clusters, rating histogram).
- "Seen it — rate it" and "Not interested" buttons: stored locally, fed into training, with re-ranking without a full rebuild.
- An adventurousness slider (lowers MMR λ, down-weights ①). README pass.

## Working notes

- **Dev servers:** the user runs `make serve` on :8000 (API) and :5173 (Vite). For verification, run separate instances instead: API on :8765 with `DATA_DIR` pointing at a scratch dir, and `API_PORT=8765 npm run dev -- --port 5199 --strictPort`. Never kill processes on :8000 or :5173.
- **Tests use fakes, never the network:** `tests/fake_tmdb.py` (an httpx MockTransport TMDB), `tests/fake_embedder.py` (a hashing embedder) and `InMemoryStore`. `tests/test_pipeline.py::make_pipeline` wires them together.
- **Sample data:** `make sample` writes a synthetic 30-film export (real titles, made-up user). `backend/data/` holds the user's real data and is gitignored.
