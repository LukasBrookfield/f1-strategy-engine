"""
Assemble the modelling table
    raw -> as_of (laps and weather) -> silver flags -> features -> one tidy DataFrame
"""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from f1_strategy_engine.config import get_config
from f1_strategy_engine.constants import PointInTimeMode
from f1_strategy_engine.features import definitions as fd
from f1_strategy_engine.features.silver import add_flags
from f1_strategy_engine.ingest.as_of import LAP_KEYS, as_of
from f1_strategy_engine.ingest.reader import read_raw

log = logging.getLogger(__name__)

WEATHER_KEYS = ("season", "round", "session", "sample_index")

LapFeature = Callable[[pd.DataFrame], pd.Series]
WeatherFeature = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]

# Feature set. Order here is column order in the output
LAP_FEATURES: tuple[LapFeature, ...] = (
    fd.compound,
    fd.tyre_age,
    fd.stint_lap_index,
    fd.fuel_corrected_lap_time,
    fd.rolling_median_last_3,
    fd.delta_to_stint_baseline,
    fd.circuit,
)
WEATHER_FEATURES: tuple[WeatherFeature, ...] = (fd.track_temp, fd.air_temp)

# Carried through unchanged
PASSTHROUGH_COLUMNS = [
    *LAP_KEYS,  # season, round, session, driver, lap_number
    "stint",
    "event_time_utc",
    "knowledge_time_utc",
    "is_lap1",
    "is_inlap",
    "is_outlap",
    "is_safety_car_lap",
    "is_vsc_lap",
    "is_accurate",
    "deleted",
    "fastf1_generated",
]


def build_features(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
    t: pd.Timestamp,
    mode: PointInTimeMode,
) -> pd.DataFrame:
    """The modelling table as it could have been built at time t.

    Takes raw (bronze) laps and weather, gates both through `as_of`, adds the
    silver flags and computes every feature over the gated rows.

    Returns one row per lap, sorted by lap key, with a fresh RangeIndex.
    """
    gated_laps = as_of(laps, t, mode, keys=LAP_KEYS).reset_index(drop=True)
    gated_weather = as_of(weather, t, mode, keys=WEATHER_KEYS).reset_index(drop=True)

    silver = add_flags(gated_laps)

    features = [feature(silver) for feature in LAP_FEATURES]
    features += [feature(silver, gated_weather) for feature in WEATHER_FEATURES]

    table = pd.concat([silver[PASSTHROUGH_COLUMNS], *features], axis=1)
    log.debug("built %d rows at t=%s (mode=%s)", len(table), t, mode)
    return table.sort_values(list(LAP_KEYS)).reset_index(drop=True)


def build_modelling_table(
    t: pd.Timestamp,
    mode: PointInTimeMode,
    seasons: Sequence[int] | None = None,
    rounds: Sequence[int] | None = None,
    *,
    include_holdout: bool = False,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Read the raw layer and build the modelling table at time t.

    `seasons` defaults to `config.train_seasons`. Asking for the held-out season
    raises unless `include_holdout=True`, so it can't be loaded by accident.
    """
    config = get_config()
    seasons = config.train_seasons if seasons is None else tuple(seasons)
    if config.holdout_season in seasons and not include_holdout:
        raise ValueError(
            f"Season {config.holdout_season} is held out; "
            "pass include_holdout=True only for the final evaluation."
        )

    laps = read_raw("laps", seasons, rounds, raw_dir)
    weather = read_raw("weather", seasons, rounds, raw_dir)
    return build_features(laps, weather, t, mode)
