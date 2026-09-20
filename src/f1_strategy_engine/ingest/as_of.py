def as_of(df, t, keys=("season", "round", "session", "driver", "lap_number")):
    """Rows as they were known at time t. Latest belief per entity, nothing later."""
    return (df[df["knowledge_time_utc"] <= t]
            .sort_values("knowledge_time_utc")
            .groupby(list(keys), as_index=False).tail(1))
