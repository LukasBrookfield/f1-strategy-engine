"""
The leakage experiment: the same forecasts made under two point-in-time modes.

A decision is made each time the race leader finishes a lap. At that moment we
forecast the rest of every driver's current stint from the stint baseline that
was known then. Only the mode passed to `as_of` differs between the two arms.
"""

import pandas as pd
from sklearn.metrics import root_mean_squared_error as rmse

from f1_strategy_engine.constants import STINT_KEYS, PointInTimeMode
from f1_strategy_engine.evaluation.report import TARGET, scoring_mask
from f1_strategy_engine.evaluation.splits import race_split
from f1_strategy_engine.features.build import build_features, build_modelling_table
from f1_strategy_engine.ingest.reader import read_raw
from f1_strategy_engine.models.baselines import PerCompoundSlopeModel

ARMS = (PointInTimeMode.NONE, PointInTimeMode.ROWS_ONLY)


def stint_baselines(table: pd.DataFrame) -> pd.Series:
    """The baseline pace of each stint, NaN if it doesn't have one yet."""
    baseline = table[TARGET] - table["delta_to_stint_baseline"]
    return table.assign(baseline=baseline).groupby(list(STINT_KEYS))["baseline"].first()


def race_forecasts(
    laps: pd.DataFrame,
    weather: pd.DataFrame,
    mode: PointInTimeMode,
    now: pd.Timestamp,
) -> pd.DataFrame:
    """One row per (decision, later lap of the same stint) in one race."""
    # What actually happened: the targets, and which lap each driver was on
    full = build_features(laps, weather, now, PointInTimeMode.ROWS_ONLY)
    decision_times = full.groupby("lap_number")["knowledge_time_utc"].min()

    pairs = []
    for t in decision_times:
        # The only thing that depends on the mode: what the baseline looked like at t
        known = build_features(laps, weather, t, mode)
        baselines = stint_baselines(known)

        so_far = full[full["knowledge_time_utc"] <= t]
        current = so_far.loc[so_far.groupby("driver")["lap_number"].idxmax()]
        current = current[["driver", "stint", "lap_number"]].rename(
            columns={"lap_number": "decision_lap"}
        )

        targets = full.merge(current, on=["driver", "stint"])
        targets = targets[targets["lap_number"] > targets["decision_lap"]]
        targets = targets.join(baselines, on=list(STINT_KEYS))

        # predict() reads the baseline as pace - delta, so setting delta this way
        # makes the model forecast from the baseline known at t
        targets["delta_to_stint_baseline"] = targets[TARGET] - targets["baseline"]
        pairs.append(targets)

    return pd.concat(pairs, ignore_index=True)


def run_leakage(now: pd.Timestamp) -> pd.DataFrame:
    """B1's forecast error under each mode, on the held-out races."""
    predictions, masks = {}, {}
    for mode in ARMS:
        train, test = race_split(build_modelling_table(now, mode))
        model = PerCompoundSlopeModel().fit(train)

        forecasts = []
        for season, round_number in test[["season", "round"]].drop_duplicates().values:
            laps = read_raw("laps", [season], [round_number])
            weather = read_raw("weather", [season], [round_number])
            forecasts.append(race_forecasts(laps, weather, mode, now))
        forecasts = pd.concat(forecasts, ignore_index=True)

        predictions[mode] = model.predict(forecasts)
        masks[mode] = scoring_mask(forecasts, predictions[mode]).to_numpy()

    # Both arms forecast the same rows in the same order; only the baselines differ
    actual = forecasts[TARGET].to_numpy()
    leaky, honest = PointInTimeMode.NONE, PointInTimeMode.ROWS_ONLY
    shared = masks[leaky] & masks[honest]

    rows = []
    for label, leaky_mask, honest_mask in [
        ("each arm's own laps", masks[leaky], masks[honest]),
        ("shared laps", shared, shared),
    ]:
        leaky_rmse = rmse(actual[leaky_mask], predictions[leaky][leaky_mask])
        honest_rmse = rmse(actual[honest_mask], predictions[honest][honest_mask])
        rows.append(
            {
                "scored on": label,
                "n NONE": leaky_mask.sum(),
                "RMSE NONE": leaky_rmse,
                "n ROWS_ONLY": honest_mask.sum(),
                "RMSE ROWS_ONLY": honest_rmse,
                "inflation %": 100 * (honest_rmse - leaky_rmse) / honest_rmse,
            }
        )
    return pd.DataFrame(rows).set_index("scored on")
