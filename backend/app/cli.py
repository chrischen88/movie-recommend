"""Command-line entry points (used by the Makefile).

    python -m app.cli recommend path/to/letterboxd-export.zip [--top N]   # process + print picks
    python -m app.cli train [--force]                                      # MovieLens model for score ③

`recommend` processes the export in memory, like a web session: only film data
(the API cache, metadata, embeddings) is kept.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sqlmodel import Session

from app.db import get_engine
from app.letterboxd import ExportError, parse_export
from app.library import library_from_export
from app.sessions import RunState


def cmd_recommend(args: argparse.Namespace) -> int:
    from app.pipeline import get_pipeline

    path = Path(args.zip)
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 1
    try:
        export = parse_export(path)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"files: {', '.join(export.files_found)}")
    print(f"films: {len(export.films)} ({len(export.rated)} rated, {len(export.watchlist)} watchlist)")
    if export.warnings:
        print(f"{len(export.warnings)} warning(s):")
        for w in export.warnings:
            print(f"  - {w}")

    pipeline = get_pipeline()
    lib, run = library_from_export(export), RunState(status="running")
    pipeline.run(lib, run)
    for stage, stats in run.stats.items():
        print(f"{stage:>11}: {stats}")
    if run.message:
        print(f"note: {run.message}")
    if run.status != "done":
        return 1
    with Session(get_engine()) as session:
        out = pipeline.recommend(lib, session, limit=args.top)
    if out is None:
        print("no recommendations: no taste model could be built", file=sys.stderr)
        return 1
    for n, r in enumerate(out[0].items, start=1):
        stars = f"~{r.predicted_rating:.1f}★" if r.predicted_rating is not None else f"{r.score:.2f}"
        print(f"{n:>3}. {r.title} ({r.year or '?'})  {stars}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Download MovieLens if needed and train the collaborative model.
    A no-op when a model with the current settings already exists (unless --force)."""
    from app.movielens import MovieLensError
    from app.pipeline import get_pipeline

    pipeline = get_pipeline()
    try:
        stats = pipeline.train_collab(force=args.force)
    except MovieLensError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if stats.get("trained") is False:
        print(f"model is up to date (validation RMSE {stats['val_rmse']:.4f}); use --force to retrain")
    else:
        print(f"collab: {stats}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # request URLs carry titles and API keys
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p_rec = sub.add_parser("recommend", help="process a Letterboxd export ZIP and print recommendations")
    p_rec.add_argument("zip")
    p_rec.add_argument("--top", type=int, default=20, help="how many films to print")
    p_rec.set_defaults(func=cmd_recommend)
    p_train = sub.add_parser("train", help="download MovieLens and train the collaborative model")
    p_train.add_argument("--force", action="store_true", help="retrain even if the model is current")
    p_train.set_defaults(func=cmd_train)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
