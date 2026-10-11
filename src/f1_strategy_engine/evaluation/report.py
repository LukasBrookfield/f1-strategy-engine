"""
Fit the baseline ladder and score it on held-out races: results table, residual plot.

Shared by `f1se train` and notebooks/04_baseline_results.ipynb, so both report
exactly the same numbers.

Residual = actual - predicted, so a positive residual means the lap was slower
than the model expected.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from f1_strategy_engine.constants import SLICK_COMPOUNDS, PointInTimeMode
from f1_strategy_engine.evaluation.splits import race_split
from f1_strategy_engine.features.build import build_modelling_table
from f1_strategy_engine.features.silver import is_representative
from f1_strategy_engine.models.base import DegradationModel
from f1_strategy_engine.models.baselines import (
    NoDegradationModel,
    PerCompoundSlopeModel,
)

TARGET = "fuel_corrected_lap_time"
ALL = "ALL"  # compound label for the whole-test-set row
# The baseline ladder, in the order it's reported
MODELS: dict[str, type[DegradationModel]] = {
    "B0": NoDegradationModel,
    "B1": PerCompoundSlopeModel,
}
PANEL_TITLES = [
    "vs lap in stint (median, IQR)",
    "vs lap number: fuel check (median, IQR)",
    "by compound",
    "by circuit",
]
X_LABELS = ["Lap in stint", "Lap number", "Compound", "Circuit"]


@dataclass(frozen=True)
class TrainResult:
    train: pd.DataFrame
    test: pd.DataFrame
    models: dict[str, DegradationModel]
    residuals: pd.DataFrame
    results: pd.DataFrame


def train_and_score(t: pd.Timestamp, pit_mode: PointInTimeMode) -> TrainResult:
    """Build the modelling table at t, split by race, fit every model, score it."""
    table = build_modelling_table(t, pit_mode)
    train, test = race_split(table)
    return score_models(train, test)


def score_models(train: pd.DataFrame, test: pd.DataFrame) -> TrainResult:
    """Fit every model on train and score it on test."""
    models = {name: model_cls().fit(train) for name, model_cls in MODELS.items()}
    predictions = {name: model.predict(test) for name, model in models.items()}

    residuals = residual_frame(test, predictions)
    return TrainResult(train, test, models, residuals, results_table(residuals))


def scoring_mask(df: pd.DataFrame, predicted: np.ndarray) -> pd.Series:
    """Laps that get scored: representative slick laps the model made a prediction for.

    Excluding laps by their own flags is evaluation hindsight, not a feature, so it
    doesn't leak. Predictions are NaN until a stint has its baseline laps.
    """
    representative = is_representative(df, df[TARGET])
    slick = df["compound"].isin(SLICK_COMPOUNDS)
    return representative & slick & np.isfinite(predicted)


def residual_frame(
    df: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    """One row per (model, scored lap), with the columns the plots break down by."""
    context = ["round", "circuit", "compound", "lap_number", "stint_lap_index"]
    frames = []
    for model, predicted in predictions.items():
        mask = scoring_mask(df, predicted)
        scored = df.loc[mask, [*context, TARGET]].copy()
        scored["predicted"] = predicted[mask.to_numpy()]
        scored["residual"] = scored[TARGET] - scored["predicted"]
        scored["model"] = model
        frames.append(scored)
    return pd.concat(frames, ignore_index=True)


def _metrics(group: pd.DataFrame) -> pd.Series:
    y, y_hat = group[TARGET], group["predicted"]
    return pd.Series(
        {
            "n_scored": len(group),
            "rmse": root_mean_squared_error(y, y_hat),
            "mae": mean_absolute_error(y, y_hat),
        }
    )


def results_table(residuals: pd.DataFrame) -> pd.DataFrame:
    """n, RMSE and MAE per model, overall and per compound."""
    overall = residuals.groupby("model", sort=False).apply(
        _metrics, include_groups=False
    )
    overall = overall.assign(compound=ALL).set_index("compound", append=True)

    residuals = residuals.assign(compound=residuals["compound"].astype(str))
    per_compound = residuals.groupby(["model", "compound"], sort=False).apply(
        _metrics, include_groups=False
    )

    table = pd.concat([overall, per_compound]).sort_index(
        level="model", sort_remaining=False
    )
    return table.astype({"n_scored": int})


def plot_residuals(residuals: pd.DataFrame) -> Figure:
    """One row per model; residuals vs lap in stint, lap number, compound, circuit.

    Uses the Figure API rather than pyplot, so it needs no display and never leaks
    open figures. A notebook renders the returned Figure inline.
    """
    models = residuals["model"].unique()
    fig = Figure(figsize=(18, 4 * len(models)), layout="constrained")
    axes = fig.subplots(len(models), 4, squeeze=False, sharey=True)

    # Clip the y-axis to the bulk of the data; heavy tails show in the boxplot fliers
    low, high = residuals["residual"].quantile([0.01, 0.99])
    compounds = residuals["compound"].astype(str)
    circuits = residuals["circuit"].astype(str)

    for row, model in zip(axes, models, strict=True):
        is_model = residuals["model"] == model
        r = residuals[is_model].assign(
            compound=compounds[is_model], circuit=circuits[is_model]
        )

        for ax, x in zip(row[:2], ["stint_lap_index", "lap_number"], strict=True):
            binned = r.groupby(x)["residual"].quantile([0.25, 0.5, 0.75]).unstack()
            ax.fill_between(binned.index, binned[0.25], binned[0.75], alpha=0.25)
            ax.plot(binned.index, binned[0.5], linewidth=2)

        sns.boxplot(data=r, x="compound", y="residual", ax=row[2], fliersize=2)
        sns.boxplot(data=r, x="circuit", y="residual", ax=row[3], fliersize=2)
        row[3].tick_params(axis="x", labelrotation=30)

        for ax, xlabel in zip(row, X_LABELS, strict=True):
            ax.axhline(0, color="grey", linewidth=1)
            ax.set_ylim(low, high)
            ax.set_xlabel(xlabel)
        row[0].set_ylabel(f"{model}\nresidual (s)")

    for ax, title in zip(axes[0], PANEL_TITLES, strict=True):
        ax.set_title(title)

    return fig
