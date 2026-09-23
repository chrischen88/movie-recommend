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
| 5 | MovieLens ingestion + score ③ | ✅ Done |
| 6 | Candidate gen (discover), OMDb, learned blend, MMR, metrics page | ✅ Done |
| 7 | OpenAI layer: re-rank, explanations, natural-language requests | ⏭ Next |
| 8 | UI polish, taste profile page, feedback loop, README | Not started |

Tests: 201 passing (`make test`). The frontend type-checks and builds (`cd frontend && npm run build`).

## What exists (by module)

- `backend/app/letterboxd.py`: parses the ZIP and merges the CSVs on a normalized `title|year` key.
- `backend/app/cache.py`: SQLite response cache with TTL, rate limiter, retry/backoff client, and an optional daily budget (for OMDb).
- `backend/app/ingest.py`: syncs an export into SQLite. Same account: incremental, using a content hash per film. Different account (from `profile.csv`'s Username): full replace (decision 15).
- `backend/app/tmdb.py`, `matching.py`: TMDB client and title/year matching with confidence scores.
- `backend/app/pipeline.py`: background run: matching → enrichment → collab → candidates → embedding → taste → blend → omdb. The collab stage trains once, then is a no-op until the settings change; if MovieLens can't be fetched it's skipped with a message, and the run still succeeds. Candidates come from four sources: TMDB recommendations and similar lists for the top 60 films rated ≥ 4.0 (`CANDIDATE_SEED_COUNT`, was 25), `/discover/movie` for the profile's best genres and non-English languages, and ③'s top predictions. The blend stage refits only when its inputs' fingerprint changes. The OMDb stage fetches the top-150 shortlist once per `CACHE_TTL_OMDB`; a bad key or an exhausted budget skips it without failing the run.
- `backend/app/embeddings.py`, `vectorstore.py`, `taste.py`: film documents, the `VectorStore` interface (Chroma + in-memory), the taste vector, clusters, and score ②.
- `backend/app/profile.py`: score ①. Builds the taste profile (shrunk mean rating deviation per director/genre/actor/keyword/decade/language/country), scores candidates, and keeps the top contributions as explanations. Also `user_ratings()`, which the taste-vector build shares.
- `backend/app/movielens.py`: downloads the MovieLens ZIP (streamed, atomic) and loads ratings plus the movieId → tmdbId links.
- `backend/app/blend.py`: cross-fitted out-of-fold ①②③ for your rated films, a non-negative Ridge blend (with and without ③), fixed-weight fallback, and the metrics (decision 17).
- `backend/app/omdb.py`: OMDb client (IMDb rating, Rotten Tomatoes, Metascore) over `CachedHttpClient` with the daily budget.
- `backend/app/collab.py`: score ③. Explicit ALS with biases (numpy/scipy), a saved base model, the user folded in per request, and predicted stars per candidate.
- `backend/app/recommend.py`: scores every eligible film, ranks by the blend's predicted rating (or the fixed-weight score), adds the watchlist boost, applies `RecFilters` (min rating from TMDB/IMDb/RT/Metacritic, quality floor, genre, decade, max runtime, language), hides shorts, then re-ranks with MMR.
- `backend/app/main.py`: FastAPI. Routes: `/api/health`, `/api/upload`, `/api/ingest/{id|latest|resume}`, `/api/films`, `/api/matches[/set|/accept|/ignore]`, `/api/recommendations`, `/api/taste`, `/api/metrics`.
- `frontend/`: Upload (progress stepper), Matches (review/fix), Recommendations (poster grid, filter bar, cluster chips, predicted "~4.5★ for you", TMDB/IMDb/RT/Metacritic ratings, ①②③ bars, up to 3 "why" lines per card) and Metrics (blend weights, held-out accuracy per method, candidate sources, MovieLens and OMDb usage) pages.

## Decisions & deviations from the spec

These are deliberate, so don't "fix" them back without reason.

1. **Films are keyed on `title|year`, not the Letterboxd URI.** In real exports, the URI in `diary.csv` and `reviews.csv` points to the log entry, not the film. `ratings.csv` is the source of truth for ratings; the latest diary rating is the fallback.
2. **Score ② is a z-scored max, not a raw max cosine** (`EMBEDDING_SCORE_MODE=zmax`). Raw cosines aren't comparable across vectors: the taste vector is a difference direction (~0.2 cosine) and centroids sit near films (~0.8), so a raw max would almost never pick the taste vector. Each vector's similarities are z-scored across the candidate pool before taking the max. `max` restores the spec's behaviour.
3. **Taste clusters have a minimum size of 3** (`TASTE_CLUSTER_MIN_SIZE`). A k is only eligible if every cluster has ≥ 3 films. Without this, a real run chose k=6 with single-film clusters whose near-duplicates flooded the results.
4. **A basic candidate pool was pulled into M3.** It uses TMDB recommendations and similar lists from the top 25 films rated ≥ 4.0, skipping films with fewer than 50 votes (`CANDIDATE_MIN_VOTES`). M6 adds `/discover/movie`.
5. **Match statuses:** matched / low_confidence / unmatched / manual / error / ignored. Manual and ignored decisions are never overwritten. An ignored film has its `tmdb_id` cleared so it can't leak into training. Errored films are retried on the next run.
6. **Ambiguous matches** (a near-tie between two films that both have real vote counts) are pushed into review. A same-name obscure film is simply outvoted.
7. **Collaborative filtering is our own numpy/scipy ALS,** because `implicit` may lack Python 3.13 wheels and targets implicit feedback anyway. It uses weighted-λ regularization. On ml-latest-small it reaches a validation RMSE of 0.844 against 1.054 for the global mean (reg 0.1, k 32, 15 iterations, 2.5 s).
8. **The DB auto-migrates by adding new columns** (`db._add_missing_columns`), so new model fields don't require deleting the DB.
9. **Score ① divides each feature type's sum by √(values of that type).** The spec says a plain weighted sum, but TMDB films carry anywhere from 3 to 40+ keywords, and each keyword seen on even one rated film has a nonzero shrunk value. A plain sum lets keyword count swamp the director. The √n scaling keeps multi-valued types comparable while still rewarding several matches.
10. **Score ① is computed per request, not stored.** It needs only the local `movie` table and takes milliseconds for a few hundred rated films, so it's always in sync with manual match fixes, with no pipeline stage or file to go stale.
11. **Weak explanations are hidden.** Contributions under `PROFILE_MIN_REASON_STARS` (0.05★) still count toward the score but aren't shown. Features on every rated film (e.g. all Drama) have a zero value by construction; float noise is snapped to 0 so they don't show up as "−0.0★".
12. **The MovieLens ZIP bypasses `CachedHttpClient`.** It's a static file download (1–240 MB), not a JSON API; the cache stores JSON bodies in SQLite. It's streamed to disk with retries, extracted to a staging dir and renamed, so a failed download never looks complete. It's fetched once. `MOVIELENS_AUTO_DOWNLOAD=false` stops the pipeline from fetching it (tests set this).
13. **The user is folded in, not trained in.** The spec says to add the user as a new row and train. Instead, the base model is trained once on MovieLens and saved (`data/movielens/model-<dataset>.npz`, keyed by a fingerprint of dataset plus hyperparameters). The user's factors and bias come from one ALS user step against the frozen item factors. That's identical to the new-row result for the user's own vector, minus their negligible pull on 9.7k item vectors. It takes microseconds, so it runs per request, always matches the current ratings, and M6's 80/20 holdout needs no retraining.
14. **Missing ③ means averaging the others.** Films not in MovieLens, or rated by fewer than `COLLAB_MIN_ITEM_RATINGS` (5) users there, get `collab_score = null`, and their blended score uses ① and ② only (since M6, the blend's *partial* model; decision 17). Score ③ is percentile-ranked only among candidates that have it. If fewer than `COLLAB_MIN_USER_RATINGS` (5) of your films are in MovieLens, ③ is off entirely.
15. **One library at a time; a different account's export replaces it.** The export's `profile.csv` Username is stored (`appstate` table). Uploading a *different* account deletes all user films and candidates before syncing, and discards `taste_model.json`. Matching, candidates, index membership and the taste model are then rebuilt, which is cheap because TMDB responses, `movie` metadata, embeddings and the MovieLens model are user-independent and stay cached. The same account (a newer export) stays incremental and keeps manual match fixes. A DB with films but no recorded account (from before this) is reset on the next upload. An export without `profile.csv` is assumed to be the same account. `test_other_account_upload_equals_fresh_install` asserts that uploading A then B matches a fresh install of B: same films, matches, candidates, scores and taste. Also: the candidate table is now *replaced* on each run (only current seeds' results survive, except those from seeds whose fetch failed). The index holds only the user's films plus current candidates; other `movie` rows are metadata cache only. `recommend()` also intersects the pool with the current library, so a failed or partial run can't surface stale films. Trade-off: switching accounts re-embeds films that were pruned from the index (local compute, no API calls).
16. **Every eligible film is scored; there's no nearest-neighbour cutoff.** Until now the pool was the union of each taste vector's 300 nearest index neighbours (`VECTOR_QUERY_K`, now removed). That made sense when ② was the only score. Once ① and ③ count equally, it dropped films those scores rate highly: on the real data it scored 583 of 792 eligible films. Now every unseen current candidate or watchlist film is scored. That's one embedding fetch plus a few matrix products, trivial at this size. Revisit only if the candidate set grows to tens of thousands.

17. **The blend is fitted on cross-fitted scores, not an 80/20 split.** The spec says hold out 20%, score it from the other 80%, and fit Ridge with 5-fold CV. With ~126 ratings a single 20% holdout gives only 25 training rows. Instead, every rated film gets out-of-fold ①②③: split the ratings into 5 folds, and for each fold rebuild the profile, taste vector/clusters and ③'s fold-in from the other four, then score the held-out fold as percentiles *within the real candidate pool* (so the features mean the same thing at training and serving time). Ridge is then fitted on all 126 rows, with α picked by CV, and the reported metrics are CV predictions too. `positive=True`: a score that doesn't help gets weight 0 rather than a negative weight nobody can explain. Two models are fitted: *full* (①②③ + log votes) for films in MovieLens and *partial* (①② + log votes) for the rest, since ③ is missing for ~16% of the pool. Below `BLEND_MIN_RATINGS_FOR_LEARNING` (50) the fixed 0.3/0.4/0.3 weights are used (a missing ③ spreads its weight over ①②). Metrics are still reported from 10 ratings.
18. **OMDb is shortlist-only and its failure is non-fatal.** Only the top `OMDB_SHORTLIST_SIZE` (150) films by blend are looked up, each once per TTL. A 401 raises `OmdbUnavailable` (not `TmdbUnavailable`) so a bad OMDb key skips that stage only. The daily budget is enforced race-free: parallel workers reserve a slot under a lock before sending, and release it if the request fails.
19. **The quality floor falls back to TMDB.** "Hide low quality" keeps a film if RT ≥ 60 or IMDb ≥ 6.5. A film without OMDb data (outside the shortlist, or no key) is judged by TMDB ≥ 6.5 instead of being hidden.
20. **③-sourced candidates need 100 MovieLens ratings** (`CANDIDATE_COLLAB_MIN_RATINGS`). ③'s top unseen predictions were dominated by films with 5–20 ratings from self-selected fans (a 5-rating DC animated TV cut, a Pink Floyd concert): 36 of 59 had < 50 ratings. Weighted-λ regularization doesn't shrink thin items harder, so their biases are noisy. The floor applies only to the candidate *source*; ③ still scores any pool film with ≥ 5 ratings.
21. **MMR rescales relevance within its pool.** Scores are percentiles over ~1,500 films, so the top 180 all sit in 0.95–1.0 and the similarity penalty (up to 0.3) swamped them: MMR was ranking by novelty and put a K-pop concert film at #2. Relevance is min-max rescaled over the pool MMR considers before applying λ.
22. **Shorts are hidden** (`RECOMMEND_MIN_RUNTIME=40` minutes) unless they're on your watchlist. MovieLens users rate some classic shorts very highly (*Rabbit of Seville*), and they crowded the top.

23. **The ranking mixes taste fit back in** (`RANK_FIT_WEIGHT=0.5`). In learned mode, films are ranked by the percentile of ½·pct(predicted rating) + ½·pct(taste fit), where taste fit is the mean of ① and ②. The "~4.4★ for you" on each card is still the learned prediction. Why: the learned blend puts ③ at 2.7★ and ① and ② at 0 (see the open item below), so 38 of the top 60 were below the median on ① or ② (acclaimed canon, not the user's kind of film). With the mix, that's 1 of 60, and the median predicted rating of the top 60 only drops from 4.44★ to 4.29★. Cost on held-out ratings: Spearman 0.52 for the ranking vs 0.62 for the prediction alone. That's expected, because the holdout measures *how you rate films you chose to watch*, while ② mostly captures *what you choose*. The metrics page shows both rows. Fixed mode (< 50 ratings) is unchanged, since ①② already carry 70% of the weight there.

## Open items / known issues

- [ ] **Fold `/reviews` into the details call** (`append_to_response=credits,keywords,external_ids,reviews`). Reviews are currently a separate request per film, so this halves first-run TMDB calls (~1,181 → ~630 on the sample export). Changing the request format changes cache keys, so responses cached under the old format won't be reused.
- [ ] Gitignore `frontend/tsconfig.tsbuildinfo` (a build artifact) and `git rm --cached` it.
- [ ] Optional: a configurable default minimum rating for recommendations.
- [ ] Weak picks: score ① now helps demote them (it's nearly independent of ②: Spearman 0.11 over the real 717-film pool). *Mission to Mars* is no longer in the pool, so that example can't be re-checked. M6's quality floor is the real fix.
- [ ] Score ① values are small with 126 ratings and `SHRINKAGE_K=3` (top directors ≈ +0.3★). That's fine for ranking, since percentiles are used, but revisit k once M6's holdout metrics exist.
- [x] ~~Stale candidates are never pruned~~: fixed along with full replacement on account change (decision 15).
- [x] ~~Use ml-32m for real data~~: done in `.env`. The default stays ml-latest-small (what the spec asks for in development). With ml-32m, 101 of 126 rated films and 84% of the pool get ③; 80/20 RMSE 0.48 vs 0.69 for your mean.
- [x] ~~**The learned blend is almost all ③**~~: the ranking now mixes in taste fit (decision 23). Still true of the prediction itself: Cross-fitted on 126 ratings: ③ alone RMSE 0.454 / Spearman 0.71 (101 films), ① 0.641 / 0.34, ② 0.677 / 0.02, your mean 0.675, fixed weights 0.621 / 0.39, learned blend 0.472 / 0.62. The full model's weights are ③ 2.7★, ① 0, ② 0, votes 0.04; the partial model uses ① (0.58★). So ② barely predicts *how you rate* films you chose to watch (it's more about *what* you watch: a selection effect the rating target can't see), and ranking by it alone filled the top with acclaimed canon. M8's adventurousness slider could expose `RANK_FIT_WEIGHT`. The review-query idea for ② (below) is untested.
- [ ] Some MovieLens links point at stale TMDB ids (25 "not found" enrichment errors from ③-sourced candidates per run). Harmless, but they log at ERROR and are retried each run.
- [ ] A harmless joblib/loky "leaked semaphore" warning appears when the server is killed after a pipeline run.

## Plan for remaining milestones

### M6: Blending, OMDb, MMR, metrics
_Done: everything below except the review-query idea, which is still open (keep it only if it beats the metrics above)._
- Add `/discover/movie` as another candidate source. Also consider score ③ as a source: the top predicted MovieLens films you haven't seen (they need TMDB enrichment and embedding like any candidate).
- Holdout for ③ is cheap: fold the user in on the 80% (decision 13), no retraining.
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
