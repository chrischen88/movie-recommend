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
| 4 | Score ① (taste profile) with explanations | ✅ Done |
| 5 | MovieLens ingestion + score ③ | ⏭ Next |
| 6 | Candidate gen (discover), OMDb, learned blend, MMR, metrics page | Not started |
| 7 | OpenAI layer: re-rank, explanations, natural-language requests | Not started |
| 8 | UI polish, taste profile page, feedback loop, README | Not started |

Tests: 128 passing (`make test`). The frontend type-checks and builds (`cd frontend && npm run build`).

## What exists (by module)

- `backend/app/letterboxd.py`: parses the ZIP and merges the CSVs on a normalized `title|year` key.
- `backend/app/cache.py`: SQLite response cache with TTL, rate limiter, retry/backoff client, and an optional daily budget (for OMDb).
- `backend/app/ingest.py`: incremental sync into SQLite using a content hash per film.
- `backend/app/tmdb.py`, `matching.py`: TMDB client and title/year matching with confidence scores.
- `backend/app/pipeline.py`: background run: matching → enrichment → candidates → embedding → taste.
- `backend/app/embeddings.py`, `vectorstore.py`, `taste.py`: film documents, the `VectorStore` interface (Chroma + in-memory), the taste vector, clusters, and score ②.
- `backend/app/profile.py`: score ①. Builds the taste profile (shrunk mean rating deviation per director/genre/actor/keyword/decade/language/country), scores candidates, and keeps the top contributions as explanations. Also `user_ratings()`, which the taste-vector build shares.
- `backend/app/recommend.py`: builds the ranked list (score = mean of ① and ② until M6), applies `RecFilters` (min TMDB rating, genre, decade, max runtime, language) and computes facets.
- `backend/app/main.py`: FastAPI. Routes: `/api/health`, `/api/upload`, `/api/ingest/{id|latest|resume}`, `/api/films`, `/api/matches[/set|/accept|/ignore]`, `/api/recommendations`, `/api/taste`.
- `frontend/`: Upload (progress stepper), Matches (review/fix), and Recommendations (poster grid, filter bar, cluster chips, ①/② bars, up to 3 "why" lines per card) pages.

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
9. **Score ① divides each feature type's sum by √(values of that type).** The spec says a plain weighted sum, but TMDB films carry anywhere from 3 to 40+ keywords, and each keyword seen on even one rated film has a nonzero shrunk value. A plain sum lets keyword count swamp the director. The √n scaling keeps multi-valued types comparable while still rewarding several matches.
10. **Score ① is computed per request, not stored.** It needs only the local `movie` table and takes milliseconds for a few hundred rated films, so it's always in sync with manual match fixes, with no pipeline stage or file to go stale.
11. **Weak explanations are hidden.** Contributions under `PROFILE_MIN_REASON_STARS` (0.05★) still count toward the score but aren't shown. Features on every rated film (e.g. all Drama) have a zero value by construction; float noise is snapped to 0 so they don't show up as "−0.0★".

## Open items / known issues

- [ ] **Fold `/reviews` into the details call** (`append_to_response=credits,keywords,external_ids,reviews`). Reviews are currently a separate request per film, so this halves first-run TMDB calls (~1,181 → ~630 on the sample export). Changing the request format changes cache keys, so responses cached under the old format won't be reused.
- [ ] Gitignore `frontend/tsconfig.tsbuildinfo` (a build artifact) and `git rm --cached` it.
- [ ] Optional: a configurable default minimum rating for recommendations.
- [ ] Weak picks: score ① now helps demote them (it's nearly independent of ②: Spearman 0.11 over the real 717-film pool). *Mission to Mars* is no longer in the pool, so that example can't be re-checked. M6's quality floor is the real fix.
- [ ] Score ① values are small with 126 ratings and `SHRINKAGE_K=3` (top directors ≈ +0.3★). That's fine for ranking, since percentiles are used, but revisit k once M6's holdout metrics exist.
- [ ] **Stale candidates are never pruned.** The `candidate` table and the vector index only grow. After uploading an export whose top-rated films differ (e.g. someone else's ZIP), films found through the old seeds can still be recommended. Scores ① and ② rank them down, but they should be dropped: remove candidates whose sources no longer include a current seed (keeping the watchlist), and remove them from the index too. There's one profile per app; a new ZIP replaces the old films (`sync_export` deletes missing ones) and match decisions carry over by `title|year`.
- [ ] A harmless joblib/loky "leaked semaphore" warning appears when the server is killed after a pipeline run.

## Plan for remaining milestones

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
- **Review queries for score ② (your own review text).** `UserFilm.review_text` is parsed but unused (the real export has 38 reviews on 128 films, ~156 chars on average). Embed each review, then query the index with it, weighted by (rating − μ) like the taste vector, as one more source in score ②'s z-scored max, e.g. `source="review:<tmdb_id>"`, labelled "Like your review of X". Keep it behind a config flag and a small weight, and keep it only if the holdout metrics improve. **Leakage:** the holdout films' reviews must be excluded while computing the holdout scores. Reviews are opinions while film docs are plot and metadata, so use `embed_query` (from M7) if it lands first.

### M7: OpenAI layer
- An `LLMClient` interface. Re-rank the top 30 → 20, returning a one-sentence "why" per film via Structured Outputs.
- Validate ids, limit moves to ±`LLM_MAX_POSITION_SHIFT`, and fall back to the blended order.
- Natural-language requests → filters + a query string, embedded with `embed_query` (BGE query prefix) and blended with the taste vector.
- Cache responses in SQLite by input hash, log tokens, and show cost on the metrics page. Use template explanations if the key is missing or a call fails.
- **Review aspects → score ①.** Use Structured Outputs to extract liked and disliked aspects from each of the user's reviews (e.g. +"atmospheric", +"strong score", −"slow pacing"). Cache the result per review by hash, so each review costs one call ever. Add these to the profile as an `aspect` feature type with its own weight, getting the same shrinkage as other features. Match candidates via their TMDB keywords and review text (embedding similarity to the aspect phrase above a threshold, or the LLM re-rank step). The holdout exclusion rule from M6 applies here too.
- **Explanations that quote the user.** Give the re-ranker the most relevant snippet of the user's own review, so it can write "You praised the score in *Whiplash*". Only quote reviews of films that actually drove the match (the matched cluster's members or top ① contributors), so quotes aren't random.

### M8: Polish and feedback
- Taste profile page (top/bottom directors/genres/actors, clusters, rating histogram).
- "Seen it — rate it" and "Not interested" buttons: stored locally, fed into training, with re-ranking without a full rebuild.
- An adventurousness slider (lowers MMR λ, down-weights ①). README pass.

## Working notes

- **Dev servers:** the user runs `make serve` on :8000 (API) and :5173 (Vite). For verification, run separate instances instead: API on :8765 with `DATA_DIR` pointing at a scratch dir, and `API_PORT=8765 npm run dev -- --port 5199 --strictPort`. Never kill processes on :8000 or :5173.
- **Tests use fakes, never the network:** `tests/fake_tmdb.py` (an httpx MockTransport TMDB), `tests/fake_embedder.py` (a hashing embedder) and `InMemoryStore`. `tests/test_pipeline.py::make_pipeline` wires them together.
- **Sample data:** `make sample` writes a synthetic 30-film export (real titles, made-up user). `backend/data/` holds the user's real data and is gitignored.
