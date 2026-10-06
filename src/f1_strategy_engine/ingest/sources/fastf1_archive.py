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
from f1_strategy_engine.ingest.knowledge_time import (
    LAP_RULE,
    WEATHER_RULE,
    lap_knowledge_time,
    weather_knowledge_time,
)

SOURCE = "fastf1_archive"

_PASSTHROUGH_LAPS = {
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

_DTYPES_LAPS = {
    "lap_number": "Int32",
    "stint": "Int32",
    "tyre_life": "Int32",
    "position": "Int32",
    "scheduled_laps": "Int32",
    "fresh_tyre": "boolean",
    "is_accurate": "boolean",
    "deleted": "boolean",
}

_PASSTHROUGH_WEATHER = {
    "Time": "relative_time",
    "AirTemp": "air_temp",
    "Humidity": "humidity",
    "Pressure": "pressure",
    "Rainfall": "rainfall",
    "TrackTemp": "track_temp",
    "WindDirection": "wind_direction",
    "WindSpeed": "wind_speed",
}


def _to_utc(td: pd.Series, t0: pd.Timestamp) -> pd.Series:
    """Session-clock timedelta -> tz-aware UTC microsecond timestamp."""
    return (t0 + td).dt.tz_localize("UTC").astype("datetime64[us, UTC]")


def enable_cache() -> None:
    """Point FastF1's HTTP cache at `config.cache_dir`.

    Must run before *any* FastF1 request, including the event schedule:
    otherwise FastF1 silently enables its default cache in ~/.cache/fastf1.
    """
    cache_dir = get_config().cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def _build_laps(
    session: fastf1.core.Session,
    season: int,
    round_number: int,
    session_code: str,
    t0: pd.Timestamp,
) -> pd.DataFrame:
    """
    Retrieve laps from fastf1 API and transform dataframes into desired structure.
    Returns laps as bronze-layer dataframe.
    """
    laps = session.laps
    out = pd.DataFrame(index=laps.index)

    # --- identity ---
    out["season"] = season
    out["round"] = round_number
    out["session"] = session_code
    out["driver"] = laps["Driver"]
    out["lap_number"] = laps["LapNumber"]

    # --- session facts known before the start ---
    out["scheduled_laps"] = session.total_laps
    out["circuit"] = session.event["Location"]

    # --- bitemporal part ---
    out["event_time_utc"] = _to_utc(laps["Time"], t0)
    out["knowledge_time_utc"] = lap_knowledge_time(out["event_time_utc"])

    # --- provenance ---
    out["ingested_at_utc"] = pd.Timestamp.now(tz="UTC").floor("us")
    out["source"] = SOURCE
    out["knowledge_time_rule"] = LAP_RULE

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
    for src, dst in _PASSTHROUGH_LAPS.items():
        if dst not in out.columns:
            out[dst] = laps[src]

    return out.astype(_DTYPES_LAPS).reset_index(drop=True)


def _build_weather(
    session: fastf1.core.Session,
    season: int,
    round_number: int,
    session_code: str,
    t0: pd.Timestamp,
) -> pd.DataFrame:
    weather = session.weather_data
    out = pd.DataFrame(index=weather.index)

    # --- identity ---
    out["season"] = season
    out["round"] = round_number
    out["session"] = session_code
    out["sample_index"] = pd.RangeIndex(start=0, stop=weather.shape[0])

    # --- bitemporal part ---
    out["event_time_utc"] = _to_utc(weather["Time"], t0)
    out["knowledge_time_utc"] = weather_knowledge_time(out["event_time_utc"])

    # --- provenance ---
    out["ingested_at_utc"] = pd.Timestamp.now(tz="UTC").floor("us")
    out["source"] = SOURCE
    out["knowledge_time_rule"] = WEATHER_RULE

    for src, dst in _PASSTHROUGH_WEATHER.items():
        if dst not in out.columns:
            out[dst] = weather[src]

    return out


def load_session(
    season: int, round_number: int, session_code: str = "R"
) -> dict[str, pd.DataFrame]:
    """Returns {"laps": ..., "weather: ..."} from one fetch.
    telemetry=True is required for `t0_date` (the UTC anchor) and messages=True
    for the `deleted` / `deleted_reason` columns, which come from race control."""
    enable_cache()
    session = fastf1.get_session(season, round_number, session_code)
    session.load(laps=True, telemetry=True, weather=True, messages=True)

    t0 = session.t0_date
    if t0 is None:
        raise RuntimeError(
            f"FastF1 could not determine t0_date for {season} round {round_number} "
            f"{session_code}."
        )

    laps = _build_laps(session, season, round_number, session_code, t0)
    weather = _build_weather(session, season, round_number, session_code, t0)
    return {"laps": laps, "weather": weather}
