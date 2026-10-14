"""
Pit recommendation: try every one-stop strategy from the current lap and rank them.
"""

import pandas as pd

from f1_strategy_engine.constants import SLICK_COMPOUNDS
from f1_strategy_engine.models.baselines import PerCompoundSlopeModel
from f1_strategy_engine.simulation.deterministic import strategy_time

PACE = "fuel_corrected_lap_time"

# Margins smaller than this are inside the model's noise: about 2x B1's
# held-out RMSE per lap (~1.0 s)
TOO_CLOSE_S = 2.0


def current_stint(known: pd.DataFrame, driver: str, lap: int) -> dict:
    """Where the driver is at the end of `lap`, from what was known at that time."""
    laps = known[known["driver"] == driver]
    now = laps[laps["lap_number"] == lap].iloc[0]
    stint = laps[laps["stint"] == now["stint"]]

    # The baseline is the same on every lap of the stint (NaN if not known yet)
    baseline = (stint[PACE] - stint["delta_to_stint_baseline"]).mean()
    used = set(laps["compound"].astype(str)) & set(SLICK_COMPOUNDS)
    return {
        "compound": str(now["compound"]),
        "stint_lap": int(now["stint_lap_index"]),
        "baseline": baseline,
        "circuit": str(now["circuit"]),
        "compounds_used": used,
    }


def recommend(
    state: dict,
    lap: int,
    laps_left: int,
    model: PerCompoundSlopeModel,
    pit_loss: float,
    offsets: pd.Series,
    max_stint: pd.Series,
) -> pd.DataFrame:
    """Every candidate strategy and its projected remaining race time, best first."""

    def delta(compound, stint_lap):
        return model.intercepts_[compound] + model.slopes_[compound] * stint_lap

    compound = state["compound"]
    stint_lap = state["stint_lap"]
    baseline = state["baseline"]
    options = []

    # Staying out is only legal once two different dry compounds have been used
    can_stay_out = stint_lap + laps_left <= max_stint[compound]
    if len(state["compounds_used"]) >= 2 and can_stay_out:
        total = strategy_time(delta, baseline, compound, stint_lap, laps_left, None)
        options.append({"strategy": "Stay out to the end", "total": total})

    for laps_to_pit in range(1, laps_left):
        # Don't extrapolate the current stint past the longest stints we've seen
        if laps_to_pit > 1 and stint_lap + laps_to_pit > max_stint[compound]:
            break
        for new_compound in map(str, SLICK_COMPOUNDS):
            if laps_left - laps_to_pit > max_stint[new_compound]:
                continue
            new_baseline = baseline + offsets.get((compound, new_compound), 0.0)
            total = strategy_time(
                delta,
                baseline,
                compound,
                stint_lap,
                laps_left,
                laps_to_pit,
                new_compound,
                new_baseline,
                pit_loss,
            )
            strategy = f"Box lap {lap + laps_to_pit} -> {new_compound}"
            options.append({"strategy": strategy, "total": total})

    return pd.DataFrame(options, columns=["strategy", "total"]).sort_values(
        "total", ignore_index=True
    )
