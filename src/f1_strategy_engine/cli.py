"""Command-line interface: one entry point, one subcommand per pipeline stage.

Installed as the `f1se` console script via `[project.scripts]` in pyproject.toml.

Every handler returns an exit code, and `main` returns it unchanged: the return
value of a console-script entry point *is* the process exit status, which is
what make, CI and any shell wrapper read to decide whether a backfill worked.
"""

import argparse
import logging

import fastf1
import pandas as pd

from f1_strategy_engine.config import get_config
from f1_strategy_engine.constants import PointInTimeMode, Session
from f1_strategy_engine.evaluation.leakage import run_leakage
from f1_strategy_engine.evaluation.report import train_and_score
from f1_strategy_engine.evaluation.splits import race_split
from f1_strategy_engine.features.build import build_features, build_modelling_table
from f1_strategy_engine.ingest.reader import read_raw
from f1_strategy_engine.ingest.sources.fastf1_archive import enable_cache, load_session
from f1_strategy_engine.ingest.writer import session_file_path, write_session
from f1_strategy_engine.models.baselines import PerCompoundSlopeModel
from f1_strategy_engine.serving.recommend import (
    TOO_CLOSE_S,
    current_stint,
    recommend,
)
from f1_strategy_engine.simulation.parameters import (
    compound_offsets,
    max_stint_laps,
    pit_loss_by_circuit,
    pit_stops,
)

log = logging.getLogger(__name__)

INGEST_SESSION = Session.R
INGEST_TABLES = ("laps", "weather")


def get_event_schedule_rounds(season: int, include_testing: bool = False) -> list[int]:
    """Round numbers for a season, in calendar order."""
    schedule = fastf1.get_event_schedule(year=season, include_testing=include_testing)
    return schedule["RoundNumber"].to_list()


def _ingest(args: argparse.Namespace) -> int:
    """Backfill sessions into the raw layer, one parquet file per session table."""
    config = get_config()
    enable_cache()

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

            all_exist = all(
                session_file_path(table, season, round_number, INGEST_SESSION).exists()
                for table in INGEST_TABLES
            )

            if not args.force and all_exist:
                skipped += len(INGEST_TABLES)
                log.info("%s: already on disk, skipped", label)
                continue

            try:
                tables = load_session(season, round_number, INGEST_SESSION)
                for table, df in tables.items():
                    file_path = write_session(
                        df, table, season, round_number, INGEST_SESSION, args.force
                    )
                    if file_path is None:
                        skipped += 1
                        log.info("%s %s: already on disk, skipped", label, table)
                    else:
                        written += 1
                        log.info(
                            "%s %s: wrote %d rows to %s",
                            label,
                            table,
                            len(df),
                            file_path,
                        )
            except Exception:
                failed += 1
                log.exception("%s: failed, continuing", label)

    log.info(
        "ingest summary: %d written, %d skipped, %d failed", written, skipped, failed
    )
    return 1 if failed else 0


def _train(args: argparse.Namespace) -> int:
    """Fit every model on the training races and print its held-out scores.

    The full results, residual plot included, live in
    notebooks/04_baseline_results.ipynb.
    """
    pit_mode = get_config().pit_mode if args.pit_mode is None else args.pit_mode
    run = train_and_score(pd.Timestamp.now(tz="UTC"), pit_mode)
    log.info(
        "split: %d train laps (rounds %s), %d test laps (rounds %s)",
        len(run.train),
        sorted(run.train["round"].unique().tolist()),
        len(run.test),
        sorted(run.test["round"].unique().tolist()),
    )

    for name, model in run.models.items():
        if isinstance(model, PerCompoundSlopeModel):
            log.info("%s degradation slopes (s/lap):\n%s", name, model.slopes_.round(3))

    print(run.results.round(3).to_string())
    return 0


def _leakage_report(args: argparse.Namespace) -> int:
    """Run the forecasts under both point-in-time modes and print the gap."""
    table = run_leakage(pd.Timestamp.now(tz="UTC"))
    print(table.round(3).to_string())
    return 0


CAVEAT = (
    "NOTE: deterministic, single-car projection in fuel-corrected time. "
    "No safety cars, traffic, rivals or tyre cliff."
)


def _recommend(args: argparse.Namespace) -> int:
    """Print the best one-stop strategies for a driver after a given lap."""
    now = pd.Timestamp.now(tz="UTC")
    train, _ = race_split(build_modelling_table(now, PointInTimeMode.ROWS_ONLY))
    model = PerCompoundSlopeModel().fit(train)
    stops = pit_stops(train)
    pit_losses = pit_loss_by_circuit(stops)

    laps = read_raw("laps", [args.season], [args.round])
    weather = read_raw("weather", [args.season], [args.round])
    this_lap = laps[(laps["driver"] == args.driver) & (laps["lap_number"] == args.lap)]
    this_lap = this_lap.iloc[0]

    # Only what was known when the driver finished this lap
    t = this_lap["knowledge_time_utc"]
    known = build_features(laps, weather, t, PointInTimeMode.ROWS_ONLY)
    state = current_stint(known, args.driver, args.lap)
    laps_left = int(this_lap["scheduled_laps"]) - args.lap

    pit_loss = pit_losses.get(state["circuit"], stops["loss"].median())
    compound = state["compound"]
    print(
        f"Race: {args.season} R{args.round} {state['circuit']} · "
        f"Driver {args.driver} · Lap {args.lap} of {args.lap + laps_left} · "
        f"On {compound}, stint lap {state['stint_lap']}"
    )
    print(
        f"Degradation: {compound} {model.slopes_[compound]:.3f} s/lap · "
        f"pit loss {pit_loss:.1f} s\n"
    )

    if pd.isna(state["baseline"]):
        print("No stint baseline yet (too few clean laps on these tyres).")
        return 1

    options = recommend(
        state,
        args.lap,
        laps_left,
        model,
        pit_loss,
        compound_offsets(train),
        max_stint_laps(train),
    )
    if options.empty:
        print("No one-stop strategy fits within the stint lengths seen in training.")
        return 1

    for strategy, total in options.head(5).itertuples(index=False):
        print(f"  {strategy:<24} projected remaining time {total:8.1f} s")

    best, runner_up = options.iloc[0], options.iloc[1]
    margin = runner_up["total"] - best["total"]
    if margin < TOO_CLOSE_S:
        print(
            f"\nToo close to call: {best['strategy']} vs {runner_up['strategy']}, "
            f"{margin:.1f} s apart."
        )
    else:
        print(
            f"\nRecommendation: {best['strategy']}, "
            f"{margin:.1f} s ahead of {runner_up['strategy']}."
        )
    print(CAVEAT)
    return 0


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
    train.add_argument(
        "--pit-mode",
        type=PointInTimeMode,
        choices=list(PointInTimeMode),
        help="Point-in-time gate for the features. Defaults to config.pit_mode.",
    )
    train.set_defaults(func=_train)

    recommend_parser = subparsers.add_parser(
        "recommend", help="Recommend a pit lap for a driver at a given lap."
    )
    recommend_parser.add_argument("--season", type=int, required=True)
    recommend_parser.add_argument("--round", type=int, required=True)
    recommend_parser.add_argument("--driver", required=True, help="e.g. VER")
    recommend_parser.add_argument("--lap", type=int, required=True)
    recommend_parser.set_defaults(func=_recommend)

    leakage_report = subparsers.add_parser(
        "leakage-report", help="Measure how much leakage inflates held-out error."
    )
    leakage_report.set_defaults(func=_leakage_report)

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
