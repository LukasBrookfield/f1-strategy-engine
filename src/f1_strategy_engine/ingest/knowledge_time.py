"""Knowledge-time derivation rules."""

import pandas as pd

from f1_strategy_engine.ingest.schema import KNOWLEDGE_TIME_DTYPE

# How long after an event occurs before a timing feed publishes it.
FEED_LATENCY = pd.Timedelta(seconds=2)

# Rule labels, derived from FEED_LATENCY
_LATENCY_LABEL = f"{FEED_LATENCY.total_seconds():g}s"
LAP_RULE = f"lap_complete+{_LATENCY_LABEL}"
WEATHER_RULE = f"observed+{_LATENCY_LABEL}"


def lap_knowledge_time(event_time_utc: pd.Series) -> pd.Series:
    """When a lap time became knowable.

    A lap time cannot exist before the lap has finished, and the feed needs a
    moment to publish it. Hence lap completion plus the feed latency.
    """
    return (event_time_utc + FEED_LATENCY).astype(KNOWLEDGE_TIME_DTYPE)


def weather_knowledge_time(event_time_utc: pd.Series) -> pd.Series:
    """When a weather sample became knowable.

    A sample is knowable once it has been taken, so there is nothing to model
    here beyond the same feed latency.
    """
    return (event_time_utc + FEED_LATENCY).astype(KNOWLEDGE_TIME_DTYPE)
