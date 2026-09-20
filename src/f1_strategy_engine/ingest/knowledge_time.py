"""Knowledge-time derivation rules.

This module carries the project's central assumption: *when did each fact
become knowable?* FastF1 is a post-hoc archive and cannot tell us, so we model
it with an explicit, documented rule.

It deliberately imports nothing from FastF1 and touches no I/O, so the rules
can be unit-tested in isolation and reviewed on their own.
"""

import pandas as pd

# How long after an event occurs before a timing feed publishes it.
#
# This is a LOWER BOUND. Real feeds lag more than this, and lag variably. That
# direction is deliberate: understating the delay means our derived knowledge
# times are optimistic, so any leakage this rule fails to catch makes a model
# look *better* than it truly is, never worse. An honest error direction.
FEED_LATENCY = pd.Timedelta(seconds=2)


def lap_knowledge_time(event_time_utc: pd.Series) -> pd.Series:
    """When a lap time became knowable.

    A lap time cannot exist before the lap has finished, and the feed needs a
    moment to publish it. Hence lap completion plus the feed latency.

    Known limitation: this cannot reconstruct lap *deletions*. A lap deleted by
    race control minutes later appears here as knowable at completion time. The
    `deleted` column in the laps table is what lets us measure how often that
    happens rather than merely admit it.
    """
    return event_time_utc + FEED_LATENCY


def weather_knowledge_time(event_time_utc: pd.Series) -> pd.Series:
    """When a weather sample became knowable.

    A sample is knowable once it has been taken, so there is nothing to model
    here beyond the same feed latency. Unlike `lap_knowledge_time`, this rule
    involves no assumption about the shape of the underlying event.
    """
    return event_time_utc + FEED_LATENCY
