# Deploying to Fly.io

The whole app runs as one Fly Machine. FastAPI serves both the API and the built UI on one port. Shared film data (SQLite with the API cache and film metadata, the Chroma index, the MovieLens model) lives on a volume mounted at `/data`. User data never touches it: each upload lives in server memory for one visit.

| File | Role |
|---|---|
| [`Dockerfile`](../Dockerfile) | Builds the UI, then a Python image with ONNX Runtime and the embedding model's ONNX export baked in |
| [`fly.toml`](../fly.toml) | 2 GB `performance-1x` (a dedicated CPU), the `/data` volume, suspend when idle, health check |
| [`deploy/start.sh`](../deploy/start.sh) | Entrypoint. Applies an uploaded data import, then starts uvicorn |
| [`scripts/fly-push-data.sh`](../scripts/fly-push-data.sh) | Uploads your local `backend/data` to the server |

## First deploy

1. Install [flyctl](https://fly.io/docs/flyctl/install/) and run `fly auth login`.
2. Pick an app name (names are global on Fly) and set it as `app` in `fly.toml`. Then create the app and its volume:
   ```bash
   fly apps create <your-app-name>
   fly volumes create data --region iad --size 3 -y
   ```
3. Set the secrets. To keep the app private, also set `AUTH_PASSWORD` (a shared login, user `letterboxd` or `AUTH_USERNAME`). Without it the app is public: see [Public mode](#public-mode).
   ```bash
   fly secrets set TMDB_API_KEY=... OMDB_API_KEY=...   # plus AUTH_PASSWORD=... for a private app
   ```
4. Deploy: `make fly-deploy`. The build takes a few minutes.
5. Copy your film data up: `make fly-push-data`. It uploads the API cache, film metadata, the index and the MovieLens model, so the server starts with everything your local runs already fetched and embedded. User tables are stripped from the snapshot before it leaves your machine.
6. Open `https://<your-app-name>.fly.dev` and log in.

`MOVIELENS_DATASET` in `fly.toml` must match the model you upload. Training `ml-32m` needs more than 2 GB of RAM, so train it locally (`make train`) and push the model.

## Day to day

- **Code changes:** `make fly-deploy`. A deploy discards the suspend snapshot, so the next visit is a full cold start (seconds, not milliseconds).
- **A newer Letterboxd export:** upload it in the UI. Films already in the shared cache cost no API calls.
- **Replacing the server's film data with your local copy:** `make fly-push-data`. This **overwrites** the server's cache and index, including films other users brought in. There's no user data to lose.
- **Logs and a shell:** `fly logs`, `fly ssh console`.

## How it behaves

- **Suspend when idle.** With no traffic, Fly suspends the machine (a RAM snapshot). The next request resumes it in a few hundred ms, and a pipeline run in progress carries on. You pay for storage only while it's suspended.
- **Sessions end on restart.** A deploy, crash or out-of-memory kill drops every session (they're only in memory). Users see "Your session ended" and upload again; thanks to the shared cache that's quick.
- **One machine only.** SQLite, the volume, the in-memory sessions and the one-at-a-time run queue all assume a single process. Don't `fly scale count` above 1.
- **`/api/health` is open** without a login, for Fly's health checks and so the push script can wake the machine. It reveals only which API keys are configured.
- **Backups.** Fly snapshots volumes daily. List them with `fly volumes snapshots list <volume-id>`.

## Public mode

Without `AUTH_PASSWORD`, anyone with the URL can use the app. What protects it:

- **Uploads per IP:** 10 an hour (`UPLOADS_PER_IP_PER_HOUR`), keyed on Fly's `Fly-Client-IP` header. Over the limit is HTTP 429.
- **Capacity:** at most 20 live sessions (`MAX_SESSIONS`) and 5 runs waiting or processing (`MAX_QUEUED_RUNS`); runs go one at a time. Over either limit is HTTP 503 "try again shortly".
- **Export size:** 50 MB and 5,000 films (`MAX_EXPORT_FILMS`).
- **OMDb quota:** the 1,000-a-day budget is enforced; when it runs out, ratings are skipped until the next day and runs still finish.
- **Cost:** one fixed-size machine, so abuse can slow the app down but not raise the bill.

Privacy: an export is processed in memory and dropped after an hour idle (`SESSION_TTL_SECONDS`), on "Forget my data now", or on restart. Logs carry no film titles or API keys (`httpx` request logging is off). The only user-derived strings on disk are TMDB search queries (film titles) in the API cache, not linked to anyone. The footer credits TMDB (required by its terms), OMDb and MovieLens. MovieLens and OMDb's free tier are non-commercial, so the app must stay free.

## Cost

The machine has a dedicated CPU because embedding new films is sustained CPU work, and a shared CPU is throttled to a crawl under it (measured ~0.2 films/s). On `performance-1x` the ONNX model loads in ~4 s and embeds ~4.5 films/s (PyTorch managed 3/s after a 13 s load). A test upload with 10 films the server hadn't seen, which brought in ~200 new candidates, took 70 s. You're billed per second while it runs; suspended, it costs storage only.


These are Fly's base prices (Ashburn/Amsterdam) as of September 2026.

| | Per month |
|---|---|
| Machine, `performance-1x` 2 GB, while running ($0.0447/h) | ~$2.70 at 2 h/day; $32.15 always on |
| Volume, 3 GB × $0.15 | $0.45 |
| Suspended machine's disk | ~$0.25 |
| Outbound data, snapshots (first 10 GB free), shared IPv4, TLS | ~$0 |
| **Typical use (a few people a week)** | **~$2–5** |

## Troubleshooting

- **Out of memory during a run** (`fly logs` shows the process killed): `fly scale memory 4096`. Fly can't suspend machines larger than 2 GB, so it will stop them instead, and cold starts return.
- **Every page returns 401 with the right password:** check `fly secrets list` for `AUTH_PASSWORD` and `AUTH_USERNAME`.
- **The push script can't reach the app:** it expects `https://<app>.fly.dev`, with the app name from `fly.toml` or its first argument.
