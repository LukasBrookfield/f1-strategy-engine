import pandas as pd

SAFETY_CAR_STATUS = "4"
VSC_STATUSES = "[67]"  # VSC deployed, VSC ending


def add_flags(df: pd.DataFrame):
    """Returns a new Dataframe with added silver layer columns.
    Columns added: is_lap1, is_inlap, is_outlap, is_safety_car_lap, is_vsc_lap
    The original Dataframe is unchanged
    """
    df = df.copy()
    df["is_lap1"] = df["lap_number"] == 1
    df["is_inlap"] = df["pit_in_time_utc"].notna()
    df["is_outlap"] = df["pit_out_time_utc"].notna()
    df["is_safety_car_lap"] = df["track_status"].str.contains(
        SAFETY_CAR_STATUS, regex=False, na=False
    )
    df["is_vsc_lap"] = df["track_status"].str.contains(VSC_STATUSES, na=False)
    return df


def is_representative(laps: pd.DataFrame, pace: pd.Series) -> pd.Series:
    """Laps whose pace reflects the tyres: the rows to aggregate or fit pace on.

    Excludes lap 1, in/out laps, SC and VSC laps, laps FastF1 couldn't time
    accurately or synthesised, laps deleted by race control, and laps with no pace.
    A missing FastF1 flag excludes the lap.

    `pace` is passed in rather than read from `laps`, so callers don't depend on
    a column name.
    """
    excluded = (
        laps["is_lap1"]
        | laps["is_inlap"]
        | laps["is_outlap"]
        | laps["is_safety_car_lap"]
        | laps["is_vsc_lap"]
        | ~laps["is_accurate"].fillna(False)
        | laps["deleted"].fillna(True)
        | laps["fastf1_generated"].fillna(True)
    )
    return ~excluded & pace.notna()


def fuel_correct(
    lap_time_s: pd.Series,
    laps_remaining: pd.Series,
    kg_per_lap: float,
    s_per_kg: float,
) -> pd.Series:
    """Corrects lap time for amount of fuel left in car at time t

    The amount of fuel in the car is negatively correlated with lap time, so if we
    didn't do fuel correction then the degradation curve would include this correlation
    which would hurt the model.
    """
    corrected = lap_time_s - laps_remaining * kg_per_lap * s_per_kg
    return corrected
