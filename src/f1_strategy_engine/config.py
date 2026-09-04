from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from .constants import PointInTimeMode, CONFIG_PATH
import yaml

@dataclass(frozen=True)
class Config:
    train_seasons: tuple[int, ...]
    holdout_season: int
    raw_dir: Path
    cache_dir: Path
    pit_mode: PointInTimeMode


@lru_cache(maxsize=1)
def get_config() -> Config:
    with open(CONFIG_PATH) as file:
        yaml_config = yaml.safe_load(file)
    
    train_seasons = tuple(yaml_config["train_seasons"])
    holdout_season = yaml_config["holdout_season"]
    raw_dir = Path(yaml_config["raw_dir"])
    cache_dir = Path(yaml_config["cache_dir"])
    pit_mode = PointInTimeMode(yaml_config["pit_mode"])
    return Config(train_seasons, holdout_season, raw_dir, cache_dir, pit_mode)

print(get_config())
