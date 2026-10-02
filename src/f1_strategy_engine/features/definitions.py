"""
Feature definitions: one function per feature, all shaped `(laps, as_of) -> Series`.
"""

import pandas as pd

from f1_strategy_engine.config import get_config
from f1_strategy_engine.features.silver import fuel_correct

RACE_KEY = ["season", "round", "session"]
STINT_KEY = [*RACE_KEY, "driver", "stint"]
BASELINE_LAPS = 3
WINDOW_SIZE = 3

# --- Tyre ---------------------------------------------------------------


def compound(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Tyre compound of each lap, as a categorical."""
    return laps["compound"].astype("category").rename("compound")


def tyre_age(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Laps on this set of tyres (the degradation clock).

    TODO:
    - Decide tyre_life vs stint_lap_index for B1 (workshop 06 §7), note in ADR 0005
    - Handle used sets and NaN tyre life
    """
    raise NotImplementedError


def stint_lap_index(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Lap number within the current stint."""
    laps_sorted = laps.sort_values([*STINT_KEY, "lap_number"])
    # add 1 to cumcount, so that the first lap after pitting is stint_lap_index = 1
    stint_lap_index = laps_sorted.groupby(STINT_KEY)["lap_number"].cumcount() + 1
    return stint_lap_index.rename("stint_lap_index")


# --- Pace ---------------------------------------------------------------


def fuel_corrected_lap_time(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Lap time with the fuel-burn effect removed."""
    fuel = get_config().fuel
    laps_remaining = laps["scheduled_laps"] - laps["lap_number"]
    corrected = fuel_correct(
        laps["lap_time_s"], laps_remaining, fuel.kg_per_lap, fuel.s_per_kg
    )
    return corrected.rename("fuel_corrected_lap_time")


def rolling_median_last_3(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Median of the last 3 fuel-corrected laps in the stint.

    min_periods=1, so the median will also be calculated on 1 or 2 values if 3
    values are not available. The window includes the current lap. Flagged laps
    carry the value from the stint's last representative lap (NaN if none yet).
    """
    pace = fuel_corrected_lap_time(laps, as_of)
    df = laps.assign(pace=pace)

    representative = ~(
            df["is_outlap"] | df["is_inlap"] | df["is_lap1"] | df["is_safety_car_lap"]
        ) & df["pace"].notna()
    eligible = df[representative].sort_values([*STINT_KEY, "lap_number"])

    rolling_median = eligible.groupby(STINT_KEY)["pace"].transform(
        lambda s: s.rolling(WINDOW_SIZE, min_periods=1).median()
    )

    ordered = df.assign(rolling=rolling_median).sort_values([*STINT_KEY, "lap_number"])
    per_lap = ordered.groupby(STINT_KEY)["rolling"].ffill()

    return per_lap.reindex(laps.index).rename("rolling_median_last_3")


def delta_to_stint_baseline(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Fuel-corrected pace relative to the stint's baseline.  

    The baseline is NaN until the stint has BASELINE_LAPS representative laps 
    in the rows given.
    """
    pace = fuel_corrected_lap_time(laps, as_of)
    df = laps.assign(pace=pace)

    representative = ~(
        df["is_outlap"] | df["is_inlap"] | df["is_lap1"] | df["is_safety_car_lap"]
    ) & df["pace"].notna()
    eligible = df[representative].sort_values([*STINT_KEY, "lap_number"])
    first_k = eligible.groupby(STINT_KEY).head(BASELINE_LAPS)

    stats = first_k.groupby(STINT_KEY)["pace"].agg(["median", "count"])
    baseline = stats["median"].where(stats["count"] == BASELINE_LAPS)

    baseline_per_lap = df[STINT_KEY].join(baseline.rename("baseline"), on=STINT_KEY)
    return (df["pace"] - baseline_per_lap["baseline"]).rename(
        "delta_to_stint_baseline"
    )

# --- Track --------------------------------------------------------------


def track_temp(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Latest track temperature known when the lap finished.

    TODO:
    - Needs weather data as well as laps, so decide how to get it in
    - merge_asof, direction="backward", by season/round/session
    - Sort both sides by the join key
    - Filter weather through as_of too
    """
    raise NotImplementedError


def air_temp(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Latest air temperature known when the lap finished.

    TODO:
    - Same join as track_temp, maybe share a helper
    """
    raise NotImplementedError


# --- Circuit ------------------------------------------------------------


def circuit(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series: 
    """Circuit of each lap, as a categorical."""
    return laps["circuit"].astype("category").rename("circuit")
