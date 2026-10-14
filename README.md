# F1 tyre degradation modelling with strategy recommendation
This project is an end-to-end ML project that uses the FastF1 SDK in Python to extract historical F1 data, and then it processes it and models tyre degradation and uses a deterministic race predictor in order to optimise F1 team pit strategy.
- Built a bitemporal data pipeline ingesting the 2024 F1 season (26,604 laps) via FastF1 into immutable Parquet, recording
when each lap happened and when it became knowable, with all reads through a point-in-time gate.
- Built a CLI that recommends a pit lap for any historical race state, using pit loss measured per circuit from the data and
reporting the margin over the runner-up strategy.
- Modelled per-compound tyre degradation, scored on whole held-out races to avoid autocorrelation leakage: 0.998 s RMSE vs
1.025 s for a constant-pace baseline, and 1.193 s on an untouched 2025 season.
- Quantified training-serving skew by ablating the point-in-time gate: leaky features understated held-out RMSE by 1.4%,
reported as a lower bound because knowledge times are derived, not observed.
