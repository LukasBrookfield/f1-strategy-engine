"""
Race parameters measured from the training races: pit loss, compound pace
offsets and how long a stint on each compound can last.
"""

import pandas as pd

from f1_strategy_engine.constants import DRIVER_KEYS, SLICK_COMPOUNDS, STINT_KEYS
from f1_strategy_engine.evaluation.leakage import stint_baselines
from f1_strategy_engine.features.silver import is_representative

PACE = "fuel_corrected_lap_time"


def pit_stops(table: pd.DataFrame) -> pd.DataFrame:
    """Every normal pit stop, with the time it cost compared to staying out.

    loss = in-lap + out-lap - 2 * the driver's median representative lap that race.
    Stops where the in- or out-lap was under SC/VSC are dropped (those are cheap
    stops), and so are stops to or from wet tyres (the track is changing).
    """
    laps = table.sort_values([*DRIVER_KEYS, "lap_number"]).reset_index(drop=True)
    next_lap = laps.groupby(list(DRIVER_KEYS)).shift(-1)

    representative = is_representative(laps, laps[PACE])
    reference = laps[PACE].where(representative)
    reference = reference.groupby([laps[k] for k in DRIVER_KEYS]).transform("median")

    neutralised = laps["is_safety_car_lap"] | laps["is_vsc_lap"]
    next_neutralised = next_lap["is_safety_car_lap"] | next_lap["is_vsc_lap"]
    is_stop = (
        laps["is_inlap"]
        & (next_lap["is_outlap"] == True)  # noqa: E712, NaN after the last lap
        & ~neutralised
        & (next_neutralised == False)  # noqa: E712
        & laps["compound"].isin(SLICK_COMPOUNDS)
        & next_lap["compound"].isin(SLICK_COMPOUNDS)
    )

    loss = laps[PACE] + next_lap[PACE] - 2 * reference
    stops = laps.assign(loss=loss)[is_stop]
    return stops[[*DRIVER_KEYS, "circuit", "lap_number", "loss"]].dropna(subset="loss")


def pit_loss_by_circuit(stops: pd.DataFrame) -> pd.Series:
    """Median pit loss per circuit."""
    return stops.groupby("circuit", observed=True)["loss"].median()


def compound_offsets(table: pd.DataFrame) -> pd.Series:
    """How much the baseline pace changes from one stint to the next, per compound pair.

    Median over consecutive stints of the same driver in the same race, so the car
    and the track are roughly the same. Indexed by (from compound, to compound).
    """
    baselines = stint_baselines(table).rename("baseline")
    compounds = table.groupby(list(STINT_KEYS))["compound"].first().astype(str)
    stints = pd.concat([baselines, compounds], axis=1).reset_index()

    stints["next_baseline"] = stints.groupby(list(DRIVER_KEYS))["baseline"].shift(-1)
    stints["next_compound"] = stints.groupby(list(DRIVER_KEYS))["compound"].shift(-1)
    stints["change"] = stints["next_baseline"] - stints["baseline"]

    slick = stints["compound"].isin(SLICK_COMPOUNDS) & stints["next_compound"].isin(
        SLICK_COMPOUNDS
    )
    offsets = stints[slick].groupby(["compound", "next_compound"])["change"].median()
    return offsets.astype(float)


def max_stint_laps(table: pd.DataFrame) -> pd.Series:
    """The longest stint we trust the model for on each compound (95th percentile)."""
    stints = table.groupby(list(STINT_KEYS)).agg(
        compound=("compound", "first"), length=("stint_lap_index", "max")
    )
    stints = stints[stints["compound"].isin(SLICK_COMPOUNDS)]
    stints["compound"] = stints["compound"].astype(str)
    return stints.groupby("compound")["length"].quantile(0.95).round().astype(int)
