import os
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from f1_strategy_engine.config import get_config
from f1_strategy_engine.constants import Session
from f1_strategy_engine.ingest.schema import SCHEMAS

Table = Literal["laps", "weather"]

DATA_FILE_NAME = "data.parquet"
# Dot-prefixed so pyarrow dataset discovery skips it: a crash before the rename
# leaves a harmless leftover instead of a file that breaks every dataset read.
TEMP_FILE_NAME = ".data.parquet.tmp"


def session_file_path(
    table: Table,
    season: int,
    round_number: int,
    session_code: Session,
    raw_dir: Path | None = None,
) -> Path:
    """Where one session's raw file lives (Hive-style partitioning)."""
    raw_dir = get_config().raw_dir if raw_dir is None else raw_dir
    return (
        raw_dir
        / table
        / f"season={season}"
        / f"round={round_number}"
        / f"session={session_code.value}"
        / DATA_FILE_NAME
    )


def write_session(
    df: pd.DataFrame,
    table: Table,
    season: int,
    round_number: int,
    session_code: Session,
    force: bool = False,
    raw_dir: Path | None = None,
) -> Path | None:
    """Write one session table to the raw layer. Returns the file path, or None.

    Returns None when the file already exists and `force` is False.

    The frame is validated against `schema.SCHEMAS[table]` at write time: wrong
    dtypes, missing or extra columns, or nulls in non-nullable fields raise.
    """
    file_path = session_file_path(table, season, round_number, session_code, raw_dir)

    if file_path.exists() and not force:
        return None

    schema = SCHEMAS[table]
    # from_pandas silently drops columns not in the schema, so check explicitly.
    extra = set(df.columns) - set(schema.names)
    if extra:
        raise ValueError(f"{table}: columns not in schema: {sorted(extra)}")
    arrow_table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = file_path.parent / TEMP_FILE_NAME
    pq.write_table(arrow_table, tmp_path)
    os.replace(tmp_path, file_path)

    return file_path
