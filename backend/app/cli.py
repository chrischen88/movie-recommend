"""Command-line entry points (used by the Makefile).

    python -m app.cli ingest path/to/letterboxd-export.zip   # parse + run the pipeline
    python -m app.cli build-index                              # (re)run the pipeline only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sqlmodel import Session

from app.db import IngestRun, get_engine
from app.ingest import sync_export
from app.letterboxd import ExportError, parse_export


def cmd_ingest(args: argparse.Namespace) -> int:
    path = Path(args.zip)
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 1
    try:
        export = parse_export(path)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with Session(get_engine()) as session:
        result = sync_export(session, export)
    print(f"files: {', '.join(export.files_found)}")
    print(f"films: {len(export.films)} ({len(export.rated)} rated, {len(export.watchlist)} watchlist)")
    print(f"sync:  {result.summary()}")
    if export.warnings:
        print(f"{len(export.warnings)} warning(s):")
        for w in export.warnings:
            print(f"  - {w}")
    return run_pipeline()


def run_pipeline() -> int:
    """Run matching → … → taste synchronously. Incremental: cheap when nothing changed."""
    from app.pipeline import get_pipeline

    pipeline = get_pipeline()
    if not pipeline.try_acquire():
        print("error: pipeline is busy", file=sys.stderr)
        return 1
    with Session(get_engine()) as session:
        run = IngestRun(stage="matching")
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id
    assert run_id is not None
    pipeline.run(run_id)
    with Session(get_engine()) as session:
        done = session.get(IngestRun, run_id)
        assert done is not None
        for stage, stats in done.stats.items():
            print(f"{stage:>11}: {stats}")
        if done.message:
            print(f"note: {done.message}")
        return 0 if done.status == "done" else 1


def cmd_build_index(_args: argparse.Namespace) -> int:
    return run_pipeline()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="parse a Letterboxd export ZIP into the local DB")
    p_ingest.add_argument("zip")
    p_ingest.set_defaults(func=cmd_ingest)
    p_index = sub.add_parser("build-index", help="run matching/enrichment/embedding/taste")
    p_index.set_defaults(func=cmd_build_index)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
