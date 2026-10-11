from enum import StrEnum

CONFIG_PATH = "configs/base.yaml"

DRIVER_KEYS = ("season", "round", "session", "driver")
STINT_KEYS = (*DRIVER_KEYS, "stint")
LAP_KEYS = (*DRIVER_KEYS, "lap_number")


class Compound(StrEnum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTER = "INTERMEDIATE"
    WET = "WET"
    UNKNOWN = "UNKNOWN"
    TEST_UNKNOWN = "TEST_UNKNOWN"


# Dry tyres only: on a drying track inters get faster with age, which is the track
# not the tyre, so no degradation model should fit or be scored on them
SLICK_COMPOUNDS = (Compound.SOFT, Compound.MEDIUM, Compound.HARD)

CIRCUITS = (
    "Austin",
    "Baku",
    "Barcelona",
    "Budapest",
    "Imola",
    "Jeddah",
    "Las Vegas",
    "Lusail",
    "Marina Bay",
    "Melbourne",
    "Mexico City",
    "Miami",
    "Monaco",
    "Montréal",
    "Monza",
    "Sakhir",
    "Shanghai",
    "Silverstone",
    "Spa-Francorchamps",
    "Spielberg",
    "Suzuka",
    "São Paulo",
    "Yas Island",
    "Zandvoort",
)

CIRCUIT_ALIASES = {"Miami Gardens": "Miami"}


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
