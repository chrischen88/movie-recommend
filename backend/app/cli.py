"""Command-line entry points (used by the Makefile).

    python -m app.cli ingest path/to/letterboxd-export.zip
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sqlmodel import Session

from app.db import get_engine
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
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="parse a Letterboxd export ZIP into the local DB")
    p_ingest.add_argument("zip")
    p_ingest.set_defaults(func=cmd_ingest)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
