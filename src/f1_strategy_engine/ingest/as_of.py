"""The point-in-time gate: the only sanctioned way to read raw data.

Every feature, model and evaluation reads bronze rows through `as_of`, so this
is the single place where "what was knowable at time t" is decided. The Phase 4
leakage experiment is nothing more than calling it with two different modes.
"""

from collections.abc import Sequence

import pandas as pd

from f1_strategy_engine.constants import PointInTimeMode

LAP_KEYS = ("season", "round", "session", "driver", "lap_number")


def as_of(
    df: pd.DataFrame,
    t: pd.Timestamp,
    mode: PointInTimeMode,
    keys: Sequence[str] = LAP_KEYS,
) -> pd.DataFrame:
    """Rows as they were known at time t. Latest belief per entity, nothing later.

    Modes:
        ROWS_ONLY:     keep rows with `knowledge_time_utc <= t`, then the latest
                       row per key.
        NONE:          no time filter, still the latest row per key. What a
                       pipeline without this gate sees.
        ROWS_AND_FITS: raises NotImplementedError. 

    Returns a new DataFrame sorted by `knowledge_time_utc`
    """
    if not isinstance(t, pd.Timestamp) or t.tzinfo is None:
        raise TypeError(f"t must be a tz-aware pd.Timestamp, got {t!r}")

    match mode:
        case PointInTimeMode.ROWS_ONLY:
            visible = df[df["knowledge_time_utc"] <= t]
        case PointInTimeMode.NONE:
            visible = df
        case PointInTimeMode.ROWS_AND_FITS:
            raise NotImplementedError(
                "ROWS_AND_FITS not yet implemented, use ROWS_ONLY."
            )
        case _:
            raise ValueError(f"Unhandled PointInTimeMode: {mode!r}")

    return (
        visible.sort_values("knowledge_time_utc", kind="stable")
        .groupby(list(keys), as_index=False, dropna=False)
        .tail(1)
    )
