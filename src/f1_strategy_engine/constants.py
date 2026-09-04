from enum import StrEnum

CONFIG_PATH = "configs/base.yaml"


class Compound(StrEnum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTER = "INTERMEDIATE"
    WET = "WET"
    UNKNOWN = "UNKNOWN"
    TEST_UNKNOWN = "TEST_UNKNOWN"


class Session(StrEnum):
    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    Q = "Q"
    R = "R"


class PointInTimeMode(StrEnum):
    NONE = "NONE"
    ROWS_ONLY = "ROWS_ONLY"
    ROWS_AND_FITS = "ROWS_AND_FITS"
