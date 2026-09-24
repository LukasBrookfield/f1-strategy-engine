"""Knowledge-time derivation rules.

This module carries the project's central assumption: *when did each fact
become knowable?* FastF1 is a post-hoc archive and cannot tell us, so we model
it with an explicit, documented rule.

It deliberately imports nothing from FastF1 and touches no I/O, so the rules
can be unit-tested in isolation and reviewed on their own.

Each rule returns timestamps at a fixed resolution and publishes a short,
machine-readable label describing itself. The label is written onto every row
as `knowledge_time_rule`, so a row's provenance can be read back off the row
rather than inferred from the commit that produced it.
"""

import pandas as pd

# Every timestamp in the raw layer is stored at microsecond resolution in UTC.
#
# Both time axes must agree on resolution: `event_time_utc` and
# `knowledge_time_utc` sit on the same row and get compared and joined against
# each other constantly, and parquet records the resolution as part of the
# column's physical type. Letting them diverge means relying on pandas to
# silently coerce at every comparison. Adding a `pd.Timedelta` upcasts to
# nanoseconds, so the rules below cast back explicitly.
#
# This belongs in `schema.py` once that exists; it lives here for now because
# these are the only functions that mint a timestamp.
KNOWLEDGE_TIME_DTYPE = "datetime64[us, UTC]"

# How long after an event occurs before a timing feed publishes it.
#
# This is a LOWER BOUND. Real feeds lag more than this, and lag variably. That
# direction is deliberate: understating the delay means our derived knowledge
# times are optimistic, so any leakage this rule fails to catch makes a model
# look *better* than it truly is, never worse. An honest error direction.
FEED_LATENCY = pd.Timedelta(seconds=2)

# Rule labels, derived from FEED_LATENCY rather than hardcoded beside it.
#
# A literal "+2s" here would drift the moment the constant above is retuned,
# and a provenance label that disagrees with the rule that produced the row is
# worse than no label at all -- it is a lie the audit trail cannot detect.
_LATENCY_LABEL = f"{FEED_LATENCY.total_seconds():g}s"
LAP_RULE = f"lap_complete+{_LATENCY_LABEL}"
WEATHER_RULE = f"observed+{_LATENCY_LABEL}"


def lap_knowledge_time(event_time_utc: pd.Series) -> pd.Series:
    """When a lap time became knowable.

    A lap time cannot exist before the lap has finished, and the feed needs a
    moment to publish it. Hence lap completion plus the feed latency.

    Takes a tz-aware series and returns `KNOWLEDGE_TIME_DTYPE`. Naive input
    raises, because a naive timestamp has no defensible interpretation here.

    Known limitation: this cannot reconstruct lap *deletions*. A lap deleted by
    race control minutes later appears here as knowable at completion time. The
    `deleted` column in the laps table is what lets us measure how often that
    happens rather than merely admit it.
    """
    return (event_time_utc + FEED_LATENCY).astype(KNOWLEDGE_TIME_DTYPE)


def weather_knowledge_time(event_time_utc: pd.Series) -> pd.Series:
    """When a weather sample became knowable.

    A sample is knowable once it has been taken, so there is nothing to model
    here beyond the same feed latency. Unlike `lap_knowledge_time`, this rule
    involves no assumption about the shape of the underlying event.

    Same input and output contract as `lap_knowledge_time`.
    """
    return (event_time_utc + FEED_LATENCY).astype(KNOWLEDGE_TIME_DTYPE)
