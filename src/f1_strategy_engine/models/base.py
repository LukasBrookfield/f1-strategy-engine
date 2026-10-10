"""
The interface every degradation model implements.
Follows sklearn's fit/predict convention: hyperparameters go in `__init__`,
learned values get a trailing underscore and are set only in `fit`.
"""

from typing import Protocol, Self

import numpy as np
import pandas as pd


class DegradationModel(Protocol):
    """Anything with these two methods is a degradation model; no inheritance needed."""

    def fit(self, df: pd.DataFrame) -> Self:
        """Learn from the modelling table (as built by `build_features`).

        Returns self, so `model = B1().fit(train)` chains.
        """
        ...

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """One prediction per row of df, in df's row order.

        No side effects: never refits, never mutates df.
        """
        ...
