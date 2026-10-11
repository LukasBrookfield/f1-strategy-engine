"""
Project the rest of a race for one strategy. Pure functions: no I/O, no config.

All times are fuel-corrected. Every strategy covers the same laps with the same
fuel, so fuel adds the same amount to every total and can't change which wins.
"""

from collections.abc import Callable

# (compound, stint lap index) -> seconds slower than the stint's baseline pace
DeltaFn = Callable[[str, int], float]


def stint_time(
    baseline: float, delta: DeltaFn, compound: str, first_lap: int, n_laps: int
) -> float:
    """Total time for n_laps of one stint, starting at stint lap first_lap."""
    return sum(baseline + delta(compound, first_lap + i) for i in range(n_laps))


def strategy_time(
    delta: DeltaFn,
    baseline: float,
    compound: str,
    stint_lap: int,
    laps_left: int,
    laps_to_pit: int | None,
    new_compound: str | None = None,
    new_baseline: float | None = None,
    pit_loss: float = 0.0,
) -> float:
    """Time for the remaining laps, starting after stint lap stint_lap.

    laps_to_pit: laps still driven on the current tyres, the last one being the
    in-lap. None means no stop.
    """
    if laps_to_pit is None:
        return stint_time(baseline, delta, compound, stint_lap + 1, laps_left)

    before = stint_time(baseline, delta, compound, stint_lap + 1, laps_to_pit)
    after = stint_time(new_baseline, delta, new_compound, 1, laps_left - laps_to_pit)
    return before + pit_loss + after
