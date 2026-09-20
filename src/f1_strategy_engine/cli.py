"""Command-line interface: one entry point, one subcommand per pipeline stage.

Installed as the `f1se` console script via `[project.scripts]` in pyproject.toml.
"""
import argparse
import sys

import fastf1

from .config import get_config
from .constants import Session
from f1_strategy_engine.ingest.writer import write_session
from f1_strategy_engine.ingest.sources.fastf1_archive import load_session


def get_event_schedule_rounds(season: int, include_testing: bool = False) -> list[int]:
    schedule = fastf1.get_event_schedule(year=season, include_testing=include_testing)
    return schedule["RoundNumber"].to_list()


def _not_implemented(args: argparse.Namespace) -> None:
    print(f"{args.command}: not implemented", file=sys.stderr)


def _ingest(args: argparse.Namespace) -> None:
    config = get_config()

    arg_dict = vars(args)

    force = arg_dict.get("force") is not None

    season = arg_dict.get("season")
    seasons = [int(season)] if season is not None else config.train_seasons

    round_number = arg_dict.get("round")
    rounds = [int(round_number)] if round_number is not None else get_event_schedule_rounds(season)
    
    for season in seasons:
        for round_number in rounds:
            session = load_session(season, round_number)
            write_session(session["laps"], "laps", season, round_number, Session.R, force)
            write_session(session["weather"], "weather", season, round_number, Session.R, force)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f1se",
        description="F1 tyre-degradation models and pit-strategy engine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    ingest = subparsers.add_parser(
        "ingest", help="Fetch FastF1 sessions into the raw (bronze) layer."
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


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the chosen subcommand."""
    args = build_parser().parse_args(argv)
    args.func(args)
