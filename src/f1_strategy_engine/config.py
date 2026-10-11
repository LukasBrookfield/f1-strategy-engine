from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from f1_strategy_engine.constants import CONFIG_PATH, PointInTimeMode


@dataclass(frozen=True)
class FuelConfig:
    kg_per_lap: float
    s_per_kg: float


@dataclass(frozen=True)
class Config:
    train_seasons: tuple[int, ...]
    holdout_season: int
    raw_dir: Path
    cache_dir: Path
    pit_mode: PointInTimeMode
    fuel: FuelConfig


@lru_cache(maxsize=1)
def get_config() -> Config:
    with open(CONFIG_PATH) as file:
        yaml_config = yaml.safe_load(file)

    train_seasons = tuple(yaml_config["train_seasons"])
    holdout_season = yaml_config["holdout_season"]
    raw_dir = Path(yaml_config["raw_dir"])
    cache_dir = Path(yaml_config["cache_dir"])
    pit_mode = PointInTimeMode(yaml_config["pit_mode"])
    fuel = FuelConfig(**yaml_config["fuel"])
    return Config(train_seasons, holdout_season, raw_dir, cache_dir, pit_mode, fuel)
