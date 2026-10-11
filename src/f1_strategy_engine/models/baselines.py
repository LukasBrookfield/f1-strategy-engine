from typing import Self

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from f1_strategy_engine.constants import SLICK_COMPOUNDS
from f1_strategy_engine.features.silver import is_representative


def _training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Representative slick laps with a stint baseline: the rows slopes are fit on."""
    representative = is_representative(df, df["fuel_corrected_lap_time"])
    slick = df["compound"].isin(SLICK_COMPOUNDS)
    return df[representative & slick & df["delta_to_stint_baseline"].notna()]


class NoDegradationModel:
    """Flat forecast at the stint's baseline pace, i.e. no degradation."""

    def fit(self, df: pd.DataFrame) -> Self:
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        baseline = df["fuel_corrected_lap_time"] - df["delta_to_stint_baseline"]
        return baseline.to_numpy(np.float64)


class PerCompoundSlopeModel:
    """Stint baseline plus a per-compound degradation slope."""

    def fit(self, df: pd.DataFrame) -> Self:
        train = _training_rows(df)

        slopes, intercepts = {}, {}
        for c, group in train.groupby("compound", observed=True):
            lr = LinearRegression().fit(
                group[["stint_lap_index"]], group["delta_to_stint_baseline"]
            )
            slopes[c] = lr.coef_[0]
            intercepts[c] = lr.intercept_

        self.slopes_ = pd.Series(slopes, dtype=np.float64)
        self.intercepts_ = pd.Series(intercepts, dtype=np.float64)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        baseline = df["fuel_corrected_lap_time"] - df["delta_to_stint_baseline"]
        compound = df["compound"].astype(str)
        delta = (
            compound.map(self.intercepts_)
            + compound.map(self.slopes_) * df["stint_lap_index"]
        )
        return (baseline + delta).to_numpy(np.float64)
