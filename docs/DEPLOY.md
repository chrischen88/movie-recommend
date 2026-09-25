# Deploying to Fly.io

The whole app runs as one Fly Machine. FastAPI serves both the API and the built UI on one port. The data (SQLite, the Chroma index and the trained models) lives on a volume mounted at `/data`.

| File | Role |
|---|---|
| [`Dockerfile`](../Dockerfile) | Builds the UI, then a Python image with CPU-only PyTorch and the embedding model baked in |
| [`fly.toml`](../fly.toml) | 2 GB `shared-cpu-1x`, the `/data` volume, suspend when idle, health check |
| [`deploy/start.sh`](../deploy/start.sh) | Entrypoint. Applies an uploaded data import, then starts uvicorn |
| [`scripts/fly-push-data.sh`](../scripts/fly-push-data.sh) | Uploads your local `backend/data` to the server |

## First deploy

1. Install [flyctl](https://fly.io/docs/flyctl/install/) and run `fly auth login`.
2. Pick an app name (names are global on Fly) and set it as `app` in `fly.toml`. Then create the app and its volume:
   ```bash
   fly apps create <your-app-name>
   fly volumes create data --region iad --size 3 -y
   ```
3. Set the secrets. `AUTH_PASSWORD` turns on the login (user `letterboxd`, or set `AUTH_USERNAME`). **Without it, anyone with the URL can use your API keys.**
   ```bash
   fly secrets set TMDB_API_KEY=... OMDB_API_KEY=... AUTH_PASSWORD=...
   ```
4. Deploy: `make fly-deploy`. The first build takes a few minutes, mostly PyTorch.
5. Copy your data up: `make fly-push-data`. This skips re-running matching, embedding and training on the server. Alternatively, upload your export in the UI and let the server process it, which is slow on a shared CPU.
6. Open `https://<your-app-name>.fly.dev` and log in.

`MOVIELENS_DATASET` in `fly.toml` must match the model you upload. Training `ml-32m` needs more than 2 GB of RAM, so train it locally (`make train`) and push the model.

## Day to day

- **Code changes:** `make fly-deploy`. A deploy discards the suspend snapshot, so the next visit is a full cold start (seconds, not milliseconds).
- **A newer Letterboxd export:** upload it in the UI. Runs are incremental, so only new or changed films are processed.
- **Replacing the server's data with your local data:** `make fly-push-data`. This **overwrites** everything on the server, including match fixes you made there.
- **Logs and a shell:** `fly logs`, `fly ssh console`.

## How it behaves

- **Suspend when idle.** With no traffic, Fly suspends the machine (a RAM snapshot). The next request resumes it in a few hundred ms, and a pipeline run in progress carries on. You pay for storage only while it's suspended.
- **Interrupted runs.** A deploy, crash or out-of-memory kill ends any pipeline run. At startup the app marks such runs as failed, and the Upload page offers "Run processing again".
- **One machine only.** SQLite, the volume and the pipeline's in-memory run lock all assume a single process. Don't `fly scale count` above 1.
- **`/api/health` is open** without a login, for Fly's health checks and so the push script can wake the machine. It reveals only which API keys are configured.
- **Backups.** Fly snapshots volumes daily. List them with `fly volumes snapshots list <volume-id>`.

## Cost

These are Fly's base prices (Ashburn/Amsterdam) as of September 2026.

| | Per month |
|---|---|
| Machine, 2 GB, while running ($0.0154/h) | ~$0.90 at 2 h/day; $11.07 always on |
| Volume, 3 GB × $0.15 | $0.45 |
| Suspended machine's disk | ~$0.25 |
| Outbound data, snapshots (first 10 GB free), shared IPv4, TLS | ~$0 |
| **Typical personal use** | **~$2–4** |

## Troubleshooting

- **Out of memory during a run** (`fly logs` shows the process killed): `fly scale memory 4096`. Fly can't suspend machines larger than 2 GB, so it will stop them instead, and cold starts return.
- **Every page returns 401 with the right password:** check `fly secrets list` for `AUTH_PASSWORD` and `AUTH_USERNAME`.
- **The push script can't reach the app:** it expects `https://<app>.fly.dev`, with the app name from `fly.toml` or its first argument.
