#!/bin/sh
# Container entrypoint. Before starting the API, applies a data import that
# scripts/fly-push-data.sh uploaded to $DATA_DIR/import.tgz. That has to happen
# here, while nothing holds the SQLite database or the Chroma index open.
set -eu

DATA_DIR="${DATA_DIR:-/data}"
IMPORT="$DATA_DIR/import.tgz"
STAGE="$DATA_DIR/.import"

if [ -f "$IMPORT" ]; then
  echo "start: importing $IMPORT"
  # Unpack beside the live data first: if this fails, the old data is untouched
  # and the import is retried on the next start.
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  tar -xzf "$IMPORT" -C "$STAGE"
  rm -rf "$DATA_DIR/chroma" "$DATA_DIR"/app.sqlite3 "$DATA_DIR"/app.sqlite3-wal "$DATA_DIR"/app.sqlite3-shm
  cp -a "$STAGE"/. "$DATA_DIR"/
  rm -rf "$STAGE" "$IMPORT"
  echo "start: import done"
fi

# One process: the pipeline's run lock and background jobs live in memory.
exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --proxy-headers --forwarded-allow-ips='*'
