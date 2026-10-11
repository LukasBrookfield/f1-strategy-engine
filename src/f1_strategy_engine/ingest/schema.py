"""The raw (bronze) layer contract: one Arrow schema per table."""

import pyarrow as pa

KNOWLEDGE_TIME_DTYPE = "datetime64[us, UTC]"

_UTC_US = pa.timestamp("us", tz="UTC")

# Shared by every table
_SESSION_KEY = [
    pa.field("season", pa.int32(), nullable=False),
    pa.field("round", pa.int32(), nullable=False),
    pa.field("session", pa.string(), nullable=False),
]
_BITEMPORAL = [
    pa.field("event_time_utc", _UTC_US, nullable=False),
    pa.field("knowledge_time_utc", _UTC_US, nullable=False),
]
_PROVENANCE = [
    pa.field("ingested_at_utc", _UTC_US, nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("knowledge_time_rule", pa.string(), nullable=False),
]

LAPS_SCHEMA = pa.schema(
    [
        *_SESSION_KEY,
        pa.field("driver", pa.string(), nullable=False),
        pa.field("lap_number", pa.int32(), nullable=False),
        *_BITEMPORAL,
        *_PROVENANCE,
        # --- session facts known before the start ---
        pa.field("scheduled_laps", pa.int32(), nullable=False),
        pa.field("circuit", pa.string(), nullable=False),
        # --- payload: durations, as seconds ---
        pa.field("lap_time_s", pa.float64()),
        pa.field("sector1_s", pa.float64()),
        pa.field("sector2_s", pa.float64()),
        pa.field("sector3_s", pa.float64()),
        # --- payload: timestamps ---
        pa.field("lap_start_time_utc", _UTC_US),
        pa.field("pit_in_time_utc", _UTC_US),
        pa.field("pit_out_time_utc", _UTC_US),
        pa.field("session_time_s", pa.float64()),
        pa.field("lap_start_session_time_s", pa.float64()),
        # --- payload: everything else, as FastF1 gave it ---
        pa.field("driver_number", pa.string()),
        pa.field("team", pa.string()),
        pa.field("stint", pa.int32()),
        pa.field("compound", pa.string()),
        pa.field("tyre_life", pa.int32()),
        pa.field("fresh_tyre", pa.bool_()),
        pa.field("track_status", pa.string()),
        pa.field("position", pa.int32()),
        pa.field("is_accurate", pa.bool_()),
        pa.field("deleted", pa.bool_()),
        pa.field("deleted_reason", pa.string()),
        pa.field("fastf1_generated", pa.bool_()),
        pa.field("speed_i1", pa.float64()),
        pa.field("speed_i2", pa.float64()),
        pa.field("speed_fl", pa.float64()),
        pa.field("speed_st", pa.float64()),
    ]
)

WEATHER_SCHEMA = pa.schema(
    [
        *_SESSION_KEY,
        pa.field("sample_index", pa.int32(), nullable=False),
        *_BITEMPORAL,
        *_PROVENANCE,
        pa.field("relative_time", pa.duration("ns")),
        pa.field("air_temp", pa.float64()),
        pa.field("humidity", pa.float64()),
        pa.field("pressure", pa.float64()),
        pa.field("rainfall", pa.bool_()),
        pa.field("track_temp", pa.float64()),
        pa.field("wind_direction", pa.int32()),
        pa.field("wind_speed", pa.float64()),
    ]
)

SCHEMAS = {"laps": LAPS_SCHEMA, "weather": WEATHER_SCHEMA}
