"""Historical backfill from FastF1's post-hoc archive.

Fetches one session, converts every timestamp to tz-aware UTC *at the boundary*,
attaches both time axes, and hands back a DataFrame.

This module does no cleaning, no filtering and no dropping of rows. Inaccurate
laps, deleted laps and FastF1-synthesised laps are all written, flagged rather
than removed -- deciding what to exclude is the silver layer's job, and a
decision made there can be revisited without re-fetching 24 races.
"""

import fastf1
import pandas as pd

from f1_strategy_engine.config import get_config
from f1_strategy_engine.ingest.knowledge_time import lap_knowledge_time

SOURCE = "fastf1_archive"

_PASSTHROUGH = {
    "Driver": "driver",
    "DriverNumber": "driver_number",
    "Team": "team",
    "LapNumber": "lap_number",
    "Stint": "stint",
    "Compound": "compound",
    "TyreLife": "tyre_life",
    "FreshTyre": "fresh_tyre",
    "TrackStatus": "track_status",
    "Position": "position",
    "IsAccurate": "is_accurate",
    "Deleted": "deleted",
    "DeletedReason": "deleted_reason",
    "FastF1Generated": "fastf1_generated",
    "SpeedI1": "speed_i1",
    "SpeedI2": "speed_i2",
    "SpeedFL": "speed_fl",
    "SpeedST": "speed_st",
}

_DTYPES = {
    "lap_number": "Int32",
    "stint": "Int32",
    "tyre_life": "Int32",
    "position": "Int32",
    "fresh_tyre": "boolean",
    "is_accurate": "boolean",
    "deleted": "boolean",
}


def _to_utc(td: pd.Series, t0: pd.Timestamp) -> pd.Series:
    """Session-clock timedelta -> tz-aware UTC microsecond timestamp."""
    return (t0 + td).dt.tz_localize("UTC").astype("datetime64[us, UTC]")


def _enable_cache() -> None:
    """Saves cached fastf1 data in designated directory."""
    cache_dir = get_config().cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def _build_laps(
    session: fastf1.core.Session, season: int, round_number: int, session_code: str
    ) -> pd.DataFrame:
    """Retrieve laps from fastf1 API and transform dataframes into desired structure."""
    t0 = session.t0_date
    if t0 is None:
        raise RuntimeError(
            f"FastF1 could not determine t0_date for {season} round {round_number} "
            f"{session_code}. Every UTC timestamp would be unanchored; refusing to write."
        )

    laps = session.laps
    out = pd.DataFrame(index=laps.index)

    # --- identity ---
    out["season"] = season
    out["round"] = round_number
    out["session"] = session_code
    out["driver"] = laps["Driver"]
    out["lap_number"] = laps["LapNumber"]

    # --- bitemporal part ---
    out["event_time_utc"] = _to_utc(laps["Time"], t0)
    out["knowledge_time_utc"] = lap_knowledge_time(out["event_time_utc"])

    # --- info ---
    out["ingested_at_utc"] = pd.Timestamp.now(tz="UTC").floor("us")
    out["source"] = SOURCE

    # --- payload: durations, as seconds ---
    out["lap_time_s"] = laps["LapTime"].dt.total_seconds()
    for n in (1, 2, 3):
        out[f"sector{n}_s"] = laps[f"Sector{n}Time"].dt.total_seconds()

    # --- payload: timestamps ---
    out["lap_start_time_utc"] = _to_utc(laps["LapStartTime"], t0)
    out["pit_in_time_utc"] = _to_utc(laps["PitInTime"], t0)
    out["pit_out_time_utc"] = _to_utc(laps["PitOutTime"], t0)

    # Session-clock originals, kept as the audit trail for the conversion above.
    out["session_time_s"] = laps["Time"].dt.total_seconds()
    out["lap_start_session_time_s"] = laps["LapStartTime"].dt.total_seconds()

    # --- payload: everything else, unmodified ---
    for src, dst in _PASSTHROUGH.items():
        if dst not in out.columns:
            out[dst] = laps[src]

    return out.astype(_DTYPES).reset_index(drop=True)


def load_session_laps(
    season: int, round_number: int, session_code: str = "R"
) -> pd.DataFrame:
    """Fetch one session's laps as a bronze-layer DataFrame.

    telemetry=True is required for `t0_date` (the UTC anchor) and messages=True
    for the `deleted` / `deleted_reason` columns, which come from race control.
    Both make the fetch slower and the cache larger; both buy columns that
    cannot be reconstructed later without re-fetching.
    """
    _enable_cache()
    session = fastf1.get_session(season, round_number, session_code)
    session.load(laps=True, telemetry=True, weather=False, messages=True)
    return _build_laps(session, season, round_number, session_code)
