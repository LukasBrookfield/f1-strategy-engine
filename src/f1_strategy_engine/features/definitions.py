"""
Feature definitions: one function per feature, all shaped `(laps, as_of) -> Series`.
"""

import pandas as pd

from f1_strategy_engine.config import get_config
from f1_strategy_engine.features.silver import fuel_correct

# --- Tyre ---------------------------------------------------------------


def compound(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:  # 🤖
    """Tyre compound of each lap, as a categorical."""
    return laps["compound"].astype("category").rename("compound")


def tyre_age(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:  # ✍️
    """Laps on this set of tyres (the degradation clock).

    TODO:
    - Decide tyre_life vs stint_lap_index for B1 (workshop 06 §7), note in ADR 0005
    - Handle used sets and NaN tyre life
    """
    raise NotImplementedError


def stint_lap_index(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:  # ✍️
    """Lap number within the current stint.

    TODO:
    - Group by driver + stint, cumcount
    - Pick 0- or 1-based and document it
    - Only use info from the lap itself and earlier
    """
    raise NotImplementedError


# --- Pace ---------------------------------------------------------------


def fuel_corrected_lap_time(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:  # 🔍
    """Lap time with the fuel-burn effect removed.

    `laps_remaining` is `scheduled_laps - lap_number`, so the final scheduled lap
    gets zero correction. `scheduled_laps` is known before the session starts, so
    it is identical under every point-in-time mode. It is never derived from the
    laps in the frame (that would leak the future and break for DNFs).

    Every lap is corrected, including in/out/SC laps: flag, don't drop.
    `as_of` is unused: the inputs are already point-in-time rows.
    """
    fuel = get_config().fuel
    laps_remaining = laps["scheduled_laps"] - laps["lap_number"]
    corrected = fuel_correct(
        laps["lap_time_s"], laps_remaining, fuel.kg_per_lap, fuel.s_per_kg
    )
    return corrected.rename("fuel_corrected_lap_time")


def rolling_median_last_3(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:  # ✍️
    """Median of the last 3 fuel-corrected laps in the stint.

    TODO:
    - Decide: does the window include the current lap? (target leakage, workshop 08 §3)
    - groupby driver + stint, rolling, min_periods
    - Decide what to do with flagged laps (SC, in/out laps, inaccurate)
    """
    raise NotImplementedError


def delta_to_stint_baseline(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:  # ✍️
    """Lap pace relative to the stint's baseline pace.

    TODO:
    - Pick a baseline (first k representative laps?), note in ADR 0005
    - Baseline aggregates only over the rows given, so it's honest under ROWS_ONLY
    - Decide the value before k laps exist (NaN?)
    """
    raise NotImplementedError


# --- Track --------------------------------------------------------------


def track_temp(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:  # ✍️
    """Latest track temperature known when the lap finished.

    TODO:
    - Needs weather data as well as laps, so decide how to get it in
    - merge_asof, direction="backward", by season/round/session
    - Sort both sides by the join key
    - Filter weather through as_of too
    """
    raise NotImplementedError


def air_temp(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:  # ✍️
    """Latest air temperature known when the lap finished.

    TODO:
    - Same join as track_temp, maybe share a helper
    """
    raise NotImplementedError


# --- Circuit ------------------------------------------------------------


def circuit(laps: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:  # 🤖
    """Circuit of each lap, as a categorical."""
    return laps["circuit"].astype("category").rename("circuit")
