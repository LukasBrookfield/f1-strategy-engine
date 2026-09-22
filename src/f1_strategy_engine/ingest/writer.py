import os
from pathlib import Path
from typing import Literal

import pandas as pd

from f1_strategy_engine.config import get_config
from f1_strategy_engine.constants import Session

DATA_FILE_NAME = "data.parquet"
TEMP_FILE_NAME = "data.tmp"


def write_session(
    df: pd.DataFrame,
    table: Literal["laps", "weather"],
    season: int,
    round_number: int,
    session_code: Session,
    force: bool = False,
) -> Path | None:
    """Write the raw data of an F1 session to parquet into the data/raw folder."""
    # create path using hive style partitioning
    path = (
        get_config().raw_dir
        / table
        / f"season={season}"
        / f"round={round_number}"
        / f"session={session_code.value}"
    )
    file_path = path / DATA_FILE_NAME

    if os.path.exists(file_path) and not force:
        return None

    path.mkdir(parents=True, exist_ok=True)

    # temporary path for saving data, so if loading is interrupted the file is recognized as corrupted
    tmp_path = path / TEMP_FILE_NAME

    df.to_parquet(tmp_path)

    # when the data has been fully loaded, the actual finished file can be created
    os.replace(tmp_path, file_path)

    return path
