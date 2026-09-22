"""Command-line interface: one entry point, one subcommand per pipeline stage.

Installed as the `f1se` console script via `[project.scripts]` in pyproject.toml.

Every handler returns an exit code, and `main` returns it unchanged: the return
value of a console-script entry point *is* the process exit status, which is
what make, CI and any shell wrapper read to decide whether a backfill worked.
"""

import argparse
import logging
import sys

import fastf1

from .config import get_config
from .constants import Session
from .ingest.sources.fastf1_archive import load_session
from .ingest.writer import write_session

log = logging.getLogger(__name__)

INGEST_SESSION = Session.R


def get_event_schedule_rounds(season: int, include_testing: bool = False) -> list[int]:
    """Round numbers for a season, in calendar order."""
    schedule = fastf1.get_event_schedule(year=season, include_testing=include_testing)
    return schedule["RoundNumber"].to_list()


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"{args.command}: not implemented", file=sys.stderr)
    return 1


def _ingest(args: argparse.Namespace) -> int:
    """Backfill sessions into the raw layer, one parquet file per session table."""
    config = get_config()

    seasons = [args.season] if args.season is not None else list(config.train_seasons)

    written = skipped = failed = 0

    for season in seasons:
        rounds = (
            [args.round]
            if args.round is not None
            else get_event_schedule_rounds(season)
        )
        log.info("season %d: %d round(s) to consider", season, len(rounds))

        for round_number in rounds:
            label = f"{season} round {round_number} {INGEST_SESSION}"
            try:
                tables = load_session(season, round_number, INGEST_SESSION)
                for table, df in tables.items():
                    path = write_session(
                        df, table, season, round_number, INGEST_SESSION, args.force
                    )
                    if path is None:
                        skipped += 1
                        log.info("%s %s: already on disk, skipped", label, table)
                    else:
                        written += 1
                        log.info(
                            "%s %s: wrote %d rows to %s", label, table, len(df), path
                        )
            except Exception:
                failed += 1
                log.exception("%s: failed, continuing", label)

    log.info(
        "ingest summary: %d written, %d skipped, %d failed", written, skipped, failed
    )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1se",
        description="F1 tyre-degradation models and pit-strategy engine.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Log at DEBUG instead of INFO."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    ingest = subparsers.add_parser(
        "ingest", help="Fetch FastF1 sessions into the raw (bronze) layer."
    )
    ingest.add_argument(
        "--season",
        type=int,
        help="Season to ingest. Defaults to every season in config.train_seasons.",
    )
    ingest.add_argument(
        "--round",
        type=int,
        help="Single round. Defaults to the full schedule of each selected season.",
    )
    ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch and overwrite sessions already present in the raw layer.",
    )
    ingest.set_defaults(func=_ingest)

    train = subparsers.add_parser(
        "train", help="Fit degradation models and score them on held-out races."
    )
    train.set_defaults(func=_not_implemented)

    recommend = subparsers.add_parser(
        "recommend", help="Recommend a pit lap for a driver at a given lap."
    )
    recommend.set_defaults(func=_not_implemented)

    leakage_report = subparsers.add_parser(
        "leakage-report", help="Measure how much leakage inflates held-out error."
    )
    leakage_report.set_defaults(func=_not_implemented)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the chosen subcommand. Returns an exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)
