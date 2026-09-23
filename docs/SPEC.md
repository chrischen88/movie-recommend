# Build: Personal Movie Recommender from a Letterboxd Export

> Original project brief, saved verbatim. Progress and deviations are tracked in [PROGRESS.md](PROGRESS.md).

You are an expert full-stack and ML engineer. Build a local web app that recommends movies to me based on my Letterboxd data export. It should combine three recommendation methods (a content-based taste profile, embedding similarity stored in a vector DB, and collaborative filtering) and blend them into one ranked list, with a short explanation for each pick.

Work in the milestones below. Finish and test each one before starting the next, and stop after each milestone to summarize what works.

---

## Inputs & data sources

1. **Letterboxd export (ZIP)**, uploaded through the UI. Parse these files:
   - `ratings.csv`: `Date, Name, Year, Letterboxd URI, Rating` (Rating runs 0.5–5.0 in half-star steps)
   - `watched.csv`: `Date, Name, Year, Letterboxd URI`
   - `diary.csv`: `Date, Name, Year, Letterboxd URI, Rating, Rewatch, Tags, Watched Date`
   - `watchlist.csv`: `Date, Name, Year, Letterboxd URI`
   - `reviews.csv` (optional, my own review text)
   - Handle missing files and extra columns gracefully.
2. **TMDB API** (key in `.env` as `TMDB_API_KEY`):
   - Match each Letterboxd film using `/search/movie` with title + year. If there's no match, retry with year ±1. Record a match confidence, and log unmatched films instead of crashing.
   - Enrich each film with: genres, director(s), top 5 cast, keywords, overview, runtime, original language, production countries, release year, poster path, and vote average/count.
   - Get candidate films from `/movie/{id}/recommendations`, `/movie/{id}/similar` (seeded from my highest-rated films), and `/discover/movie`.
   - Get user review text from `/movie/{id}/reviews` (first page is enough; truncate each review to about 500 chars).
3. **OMDb API** (key in `.env` as `OMDB_API_KEY`):
   - Use the IMDb ID from TMDB's `external_ids` to fetch the Rotten Tomatoes Tomatometer, IMDb rating, and Metascore from the `Ratings` array. These fields are often missing, so handle that.
   - The free tier allows **1,000 requests/day**, so only call OMDb for the final shortlisted candidates, and cache every result.
4. **MovieLens** (`ml-latest-small` for development; make `ml-32m` configurable):
   - Use `links.csv` to map `movieId` to `tmdbId`.

**Caching:** Cache every external API response in SQLite, keyed by endpoint + params, with a TTL. Respect rate limits: use a throttle plus retry with exponential backoff. The app must never refetch data it already has.

---

## Tech stack

- **Backend:** Python 3.11+, FastAPI, pandas, SQLite (via SQLModel or SQLAlchemy)
- **Vector DB:** Chroma, persistent and stored locally. Hide it behind a small interface so Qdrant or pgvector could be swapped in later.
- **Embeddings:** `sentence-transformers` with `BAAI/bge-small-en-v1.5` (configurable)
- **Collaborative filtering:** matrix factorization (e.g. the `implicit` library's ALS, or explicit SVD via numpy/scipy). Pick the most reliable option and justify the choice briefly.
- **Blending:** scikit-learn (Ridge regression)
- **Frontend:** React + Vite + TypeScript, styled cleanly with Tailwind
- **Config:** `.env` for keys, plus a `config.py` for tunable weights and parameters
- Include a `README.md` with setup steps, and a `Makefile` or scripts for `setup`, `ingest`, `build-index`, `train`, and `serve`.

---

## Recommendation engine

Every candidate gets three scores. Normalize each score to 0–1 using percentile rank across the candidate set.

### ① Taste-profile score (content-based)
- Compute my mean rating μ. For each feature value (each director, genre, actor, keyword, decade, language, country), compute the average of (my rating − μ) over films that have it.
- Apply Bayesian shrinkage toward 0 based on how many films support each value, so one film can't dominate.
- A candidate's score is the weighted sum of its features' values, with separate weights per feature type (configurable).
- Keep the top contributing features so the UI can explain the score (e.g. "Director Denis Villeneuve: +0.9").

### ② Embedding similarity score (vector DB)
- Build a document for each film: `title + year + genres + director + keywords + overview + top review snippets`. Embed it and store it in Chroma, with metadata (tmdb_id, year, genres, seen flag).
- Build my **taste vector** as Σ (rating − μ) · embedding over my rated films, then L2-normalize it. Films I disliked push the vector away from them.
- Also support **multiple taste clusters**: run k-means over the embeddings of my films rated ≥ 4.0 (k from 3–6, chosen by silhouette score) and query once per cluster, so recommendations don't collapse onto a single genre.
- A candidate's score is the max cosine similarity across the taste vector and the cluster centroids. Record which cluster it matched.

### ③ Collaborative score
- Map my ratings to MovieLens `movieId`s via `tmdbId`, then add me as a new user.
- Train the matrix-factorization model and predict ratings for the candidates.
- For candidates that aren't in MovieLens (recent releases), set this score to null. The blend must handle that by re-weighting the other two scores.

### Blending
- **Learned weights:** Hold out 20% of my rated films. Compute ①②③ for them using only the other 80%, then fit a Ridge regression that predicts my rating from the three scores (plus an optional small feature: log TMDB vote count). Use 5-fold CV.
- Report per-method and blended performance (RMSE and Spearman correlation) in a `/metrics` endpoint and on a UI page.
- **Fallback:** if I have fewer than 50 ratings, use fixed weights from config (default 0.3 / 0.4 / 0.3).
- **Post-processing:**
  - Exclude everything in `watched.csv`.
  - Apply a small configurable boost to films on my watchlist.
  - Apply an optional quality floor (e.g. Tomatometer ≥ 60% or IMDb ≥ 6.5), toggleable in the UI.
  - Apply MMR re-ranking (λ ≈ 0.7) on the embeddings so the top 20 are varied.

### LLM layer (OpenAI)
- Use the official `openai` Python SDK. Read the key from `OPENAI_API_KEY` in `.env`, and make the model configurable via `OPENAI_MODEL`.
- Put all LLM calls behind a small `LLMClient` interface so the provider can be swapped later.
- **Re-rank + explain:** send the top 30 blended candidates (with their ①②③ breakdowns, genres, director, overview, and RT/IMDb scores) plus my top 15 and bottom 10 rated films.
  - The model returns the top 20 in its preferred order, each with a one-sentence "why you'll like it."
  - Use Structured Outputs (a JSON schema) so the response parses reliably.
- **Guardrails:**
  - The model may only reorder and explain the candidates it was given; it must never add films.
  - Validate every returned `tmdb_id` against the input list, drop anything unknown, and fall back to the blended order if validation fails.
  - Limit how far the model can move a film (configurable, e.g. at most ±10 positions).
- **Natural-language requests:** add a text box in the UI ("something like Arrival but lighter", "a 90-minute thriller for tonight").
  - The LLM turns the request into structured filters (genres, runtime, decade, mood keywords) plus a short query string.
  - Embed the query string and blend it with my taste vector for the vector DB search, then run the normal scoring pipeline.
- **Cost and caching:**
  - Cache LLM responses in SQLite, keyed by a hash of the input.
  - Log token usage per request and show the running cost on the metrics page.
- If the key is missing or the call fails, the app must still work: use the blended order and template explanations built from the top contributing features.

---

## Frontend

1. **Upload page:** drag-and-drop the Letterboxd ZIP, then show ingestion progress (parsing → TMDB matching → enrichment → embedding → training) streamed via SSE or polling.
2. **Match review:** a table of films that didn't match or matched with low confidence, where I can fix each match by entering the TMDB ID.
3. **Recommendations:** a poster grid. Each card shows:
   - title, year, and director
   - blended score, plus a small bar for each of ①②③
   - RT / IMDb / Metacritic badges
   - the explanation text
   - which taste cluster it came from

   Filters: genre, decade, runtime, language, and hide-low-quality. Add a slider for how adventurous the results are (lowers the MMR λ and down-weights the taste-profile score).
4. **Taste profile page:** my top/bottom directors, genres, and actors, the taste clusters (with example films in each), and a rating histogram.
5. **Metrics page:** holdout performance per method and the learned blend weights.
6. **Feedback:** on each card, "Seen it — rate it" and "Not interested" buttons. Store these locally, add them to the training data, and allow re-ranking without a full rebuild.

---

## Milestones

1. Project scaffold, config, caching layer, and Letterboxd ZIP parsing, with unit tests on a sample export (create a fixture with about 30 films).
2. TMDB matching and enrichment, plus the match-review UI.
3. Embeddings + Chroma index, taste vector/clusters, and score ②, with a basic recommendations page.
4. Score ① with explanations.
5. MovieLens ingestion and score ③.
6. Candidate generation, OMDb enrichment, blending with learned weights, MMR, and the metrics page.
7. OpenAI LLM layer (re-ranking, explanations, and natural-language requests).
8. Full UI polish, feedback loop, and README.

## Quality bar

- Use type hints throughout. Test the parsing, matching, scoring, and blending code with pytest.
- There must be no API keys in the code, and the app must still run (with reduced features) if the OMDb or OpenAI keys are missing.
- Rebuilds must be incremental: re-uploading a newer export only processes new or changed films.
- Log clearly, and never fail silently on a bad match or a missing field.
