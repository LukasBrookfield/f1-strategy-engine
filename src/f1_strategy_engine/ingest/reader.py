"""
No module outside `ingest/` reads `data/raw/` directly: everything goes through
`read_raw` and then through `as_of`. One reader, one writer, one schema.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from f1_strategy_engine.config import get_config
from f1_strategy_engine.ingest.writer import Table


def read_raw(
    table: Table,
    seasons: Sequence[int] | None = None,
    rounds: Sequence[int] | None = None,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Every raw row of `table`, optionally restricted to some seasons and rounds."""
    raw_dir = get_config().raw_dir if raw_dir is None else raw_dir

    filters = []
    if seasons is not None:
        filters.append(("season", "in", list(seasons)))
    if rounds is not None:
        filters.append(("round", "in", list(rounds)))

    return pd.read_parquet(
        raw_dir / table,
        partitioning=None,
        filters=filters or None,
    )
