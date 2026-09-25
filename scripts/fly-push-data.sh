#!/usr/bin/env bash
# Copy the local film data (backend/data) to the Fly app, replacing what's there.
#
# Uploads a snapshot of the shared film data: the API cache and TMDB/OMDb
# metadata (SQLite), the Chroma index and the MovieLens model. That saves the
# server the API calls, embedding and training on its slow shared CPU. The raw
# MovieLens download is skipped: only the trained model is needed. User data
# isn't part of it: a database from an older version still has user tables, and
# the server drops them (and VACUUMs) at startup.
# Don't run this while a local pipeline run is in progress.
#
# Usage: scripts/fly-push-data.sh [app-name]   (default: `app` in fly.toml)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/backend/data"
PY="$ROOT/backend/.venv/bin/python"
APP="${1:-$(sed -n 's/^app *= *"\(.*\)"/\1/p' "$ROOT/fly.toml")}"

[ -f "$DATA/app.sqlite3" ] || { echo "no local data in $DATA: run an ingest first" >&2; exit 1; }
command -v fly >/dev/null || { echo "flyctl not found: https://fly.io/docs/flyctl/install/" >&2; exit 1; }

local_ds="$(sed -n 's/^MOVIELENS_DATASET=//p' "$ROOT/.env" 2>/dev/null | tr -d '"'"'" || true)"
fly_ds="$(sed -n 's/^ *MOVIELENS_DATASET *= *"\(.*\)"/\1/p' "$ROOT/fly.toml")"
if [ "${local_ds:-ml-latest-small}" != "$fly_ds" ]; then
  echo "warning: local MOVIELENS_DATASET is ${local_ds:-ml-latest-small} but fly.toml sets $fly_ds;" >&2
  echo "         the server will retrain score 3 on $fly_ds. Edit fly.toml and redeploy to match." >&2
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/data/movielens"

echo "snapshotting $DATA"
cp -R "$DATA/chroma" "$STAGE/data/chroma"
# SQLite's backup API gives a consistent copy even if the local server has the DB open.
"$PY" - "$DATA" "$STAGE/data" <<'PYEOF'
import sqlite3, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for rel in ("app.sqlite3", "chroma/chroma.sqlite3"):
    if (src / rel).exists():
        (dst / rel).unlink(missing_ok=True)
        with sqlite3.connect(src / rel) as a, sqlite3.connect(dst / rel) as b:
            a.backup(b)
PYEOF
rm -f "$STAGE"/data/chroma/chroma.sqlite3-{wal,shm}

# Strip user data from the snapshot before it leaves this machine: the same
# purge the server runs at startup (legacy user tables + VACUUM, index flags).
(cd "$ROOT/backend" && "$PY" - "$STAGE/data" <<'PYEOF'
import sys
from pathlib import Path
from sqlalchemy import text
from app.db import make_engine, purge_user_data
from app.vectorstore import ChromaStore
data = Path(sys.argv[1])
engine = make_engine(data / "app.sqlite3")
removed = purge_user_data(engine, data)
with engine.connect() as conn:
    conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
engine.dispose()
flags = ChromaStore(data / "chroma").strip_metadata_keys(["seen"])
print(f"stripped from snapshot: {', '.join(removed) or 'no user tables'}; index flags on {flags} films")
PYEOF
)
rm -f "$STAGE"/data/app.sqlite3-{wal,shm}
cp "$DATA"/movielens/model-*.npz "$STAGE/data/movielens/" 2>/dev/null || true

tar -czf "$STAGE/import.tgz" -C "$STAGE/data" .
echo "archive: $(du -h "$STAGE/import.tgz" | cut -f1)"

echo "waking $APP"
curl -fsS --retry 5 --retry-delay 3 --retry-all-errors "https://$APP.fly.dev/api/health" >/dev/null

echo "uploading"
fly ssh console -a "$APP" -C "rm -f /data/import.tgz"
fly ssh sftp put -a "$APP" "$STAGE/import.tgz" /data/import.tgz

echo "restarting $APP to apply the import"
fly apps restart "$APP"
curl -fsS --retry 10 --retry-delay 3 --retry-all-errors "https://$APP.fly.dev/api/health" >/dev/null
echo "done: https://$APP.fly.dev"
