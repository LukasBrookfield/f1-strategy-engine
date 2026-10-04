"""
Feature definitions: one function per feature.
All shaped `laps -> Series` apart from features that require weather data, which are
shaped `(laps, weather) -> Series`
Point-in-time correctness is enforced upstream.
"""

import pandas as pd

from f1_strategy_engine.config import get_config
from f1_strategy_engine.constants import CIRCUIT_ALIASES, CIRCUITS, Compound
from f1_strategy_engine.features.silver import fuel_correct

RACE_KEY = ["season", "round", "session"]
STINT_KEY = [*RACE_KEY, "driver", "stint"]
BASELINE_LAPS = 3
WINDOW_SIZE = 3

# Fixed categories, so every frame (train, eval, serve) encodes values identically
COMPOUND_DTYPE = pd.CategoricalDtype([c.value for c in Compound])
CIRCUIT_DTYPE = pd.CategoricalDtype(CIRCUITS)


def _representative(df: pd.DataFrame) -> pd.Series:
    """Laps whose pace reflects the tyres, used by the pace aggregates.

    Excludes lap 1, in/out laps, SC and VSC laps, laps FastF1 couldn't time
    accurately or synthesised, laps deleted by race control, and missing times.
    A missing FastF1 flag excludes the lap.
    """
    excluded = (
        df["is_lap1"]
        | df["is_inlap"]
        | df["is_outlap"]
        | df["is_safety_car_lap"]
        | df["is_vsc_lap"]
        | ~df["is_accurate"].fillna(False)
        | df["deleted"].fillna(True)
        | df["fastf1_generated"].fillna(True)
    )
    return ~excluded & df["pace"].notna()


def _to_category(values: pd.Series, dtype: pd.CategoricalDtype) -> pd.Series:
    """Cast to a fixed categorical, raising on unknown values.

    `astype` would silently turn a value outside the categories into NaN.
    """
    present = values.dropna()
    unknown = present[~present.isin(dtype.categories)].unique()
    if len(unknown) > 0:
        raise ValueError(f"Unknown {values.name} values: {sorted(unknown)}")
    return values.astype(dtype)


# --- Tyre ---------------------------------------------------------------


def compound(laps: pd.DataFrame) -> pd.Series:
    """Tyre compound of each lap, as a categorical with fixed categories."""
    return _to_category(laps["compound"], COMPOUND_DTYPE).rename("compound")


def tyre_age(laps: pd.DataFrame) -> pd.Series:
    """Laps on this set of tyres. Includes laps from previous sessions."""
    return laps["tyre_life"].rename("tyre_age")


def stint_lap_index(laps: pd.DataFrame) -> pd.Series:
    """Lap number within the current stint; the stint's first lap is 1.

    Counted from lap numbers, not rows, so a missing lap row can't shift the index
    of later laps.
    """
    stint_start = laps.groupby(STINT_KEY)["lap_number"].transform("min")
    return (laps["lap_number"] - stint_start + 1).rename("stint_lap_index")


# --- Pace ---------------------------------------------------------------


def fuel_corrected_lap_time(laps: pd.DataFrame) -> pd.Series:
    """Lap time with the fuel-burn effect removed."""
    fuel = get_config().fuel
    laps_remaining = laps["scheduled_laps"] - laps["lap_number"]
    corrected = fuel_correct(
        laps["lap_time_s"], laps_remaining, fuel.kg_per_lap, fuel.s_per_kg
    )
    return corrected.rename("fuel_corrected_lap_time")


def rolling_median_last_3(laps: pd.DataFrame) -> pd.Series:
    """Median of the last 3 fuel-corrected laps in the stint.

    min_periods=1, so the median will also be calculated on 1 or 2 values if 3
    values are not available. The window includes the current lap. Laps that
    aren't representative are skipped by the window and carry the value from the
    stint's last representative lap (NaN if none yet).
    """
    pace = fuel_corrected_lap_time(laps)
    df = laps.assign(pace=pace)

    representative = _representative(df)
    eligible = df[representative].sort_values([*STINT_KEY, "lap_number"])

    rolling_median = eligible.groupby(STINT_KEY)["pace"].transform(
        lambda s: s.rolling(WINDOW_SIZE, min_periods=1).median()
    )

    ordered = df.assign(rolling=rolling_median).sort_values([*STINT_KEY, "lap_number"])
    per_lap = ordered.groupby(STINT_KEY)["rolling"].ffill()

    return per_lap.reindex(laps.index).rename("rolling_median_last_3")


def delta_to_stint_baseline(laps: pd.DataFrame) -> pd.Series:
    """Fuel-corrected pace relative to the stint's baseline.  

    The baseline is NaN until the stint has BASELINE_LAPS representative laps 
    in the rows given.
    """
    pace = fuel_corrected_lap_time(laps)
    df = laps.assign(pace=pace)

    representative = _representative(df)
    eligible = df[representative].sort_values([*STINT_KEY, "lap_number"])
    first_k = eligible.groupby(STINT_KEY).head(BASELINE_LAPS)

    stats = first_k.groupby(STINT_KEY)["pace"].agg(["median", "count"])
    baseline = stats["median"].where(stats["count"] == BASELINE_LAPS)

    baseline_per_lap = df[STINT_KEY].join(baseline.rename("baseline"), on=STINT_KEY)
    return (df["pace"] - baseline_per_lap["baseline"]).rename(
        "delta_to_stint_baseline"
    )


# --- Track --------------------------------------------------------------


def _latest_weather(laps: pd.DataFrame, weather: pd.DataFrame, col: str) -> pd.Series:
    """
    Helper method for getting the latest weather columns when the lap became known.
    """
    lap_cols = [*RACE_KEY, "knowledge_time_utc"]
    weather_cols = [*lap_cols, col]
    laps_sorted = laps[lap_cols].reset_index(names="_row").sort_values(
        "knowledge_time_utc"
    )
    weather_sorted = weather[weather_cols].sort_values("knowledge_time_utc")
    df = pd.merge_asof(
        laps_sorted,
        weather_sorted,
        on="knowledge_time_utc",
        by=RACE_KEY,
        direction="backward",
        tolerance=pd.Timedelta("2min"),
    )

    return df.set_index("_row")[col].reindex(laps.index).rename(col)


def track_temp(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.Series:
    """Latest track temperature known when the lap became known.

    A sample older than 2 min (twice the ~60 s feed interval) gives NaN, not stale data
    """
    return _latest_weather(laps, weather, "track_temp")


def air_temp(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.Series:
    """Latest air temperature known when the lap became known.

    A sample older than 2 min (twice the ~60 s feed interval) gives NaN, not stale data
    """
    return _latest_weather(laps, weather, "air_temp")


# --- Circuit ------------------------------------------------------------


def circuit(laps: pd.DataFrame) -> pd.Series:
    """Circuit of each lap, as a categorical with fixed categories.

    Aliases are mapped first, so one circuit is one category across seasons.
    """
    canonical = laps["circuit"].replace(CIRCUIT_ALIASES)
    return _to_category(canonical, CIRCUIT_DTYPE).rename("circuit")
