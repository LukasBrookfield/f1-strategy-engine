# F1 tyre degradation modelling with strategy recommendation
This project is an end-to-end ML project that uses the FastF1 SDK in Python to extract historical F1 data, and then it processes it and models tyre degradation and uses a deterministic race predictor in order to optimise F1 team pit strategy.
# Results
Trained on the 2024 season only. 2025 is the held-out season, evaluated once.

| Result | Value | Measured on |
|---|---|---|
| Lap-time RMSE, held-out races | 0.998 s vs 1.025 s for a constant-pace baseline (2.7% better) | 2024 rounds 19–24 (4,356 laps) |
| Lap-time RMSE, held-out season | 1.193 s vs 1.277 s for a constant-pace baseline (6.6% better) | All of 2025 (21,384 laps) |
| Leakage (training–serving skew) | Without the point-in-time gate, RMSE looks 1.4% better than it really is | 2024 rounds 19–24 |

Every number comes from `notebooks/05_final_report.ipynb`.

# How to run
The `make recommend` command takes arguments --season, --round, --driver and --lap. The default is 2025, round 14, Lando Norris, lap 44.
```bash
make ingest     # ingest raw data with FastF1 API
make recommend  # recommend pit strategy based on season, round, lap and driver
```
# What's next
- Monte-carlo race predictor
- Feature store
- State-space degradation model
- Experiment tracking with MLflow
# Architecture
```  
  FastF1 archive ────► ┌──────────────────────────────────────┐
                       │  ingest/                             │
                       │  two timestamps, no cleaning at all  │
                       └───────────────┬──────────────────────┘
                                       │  
                       ┌───────────────▼──────────────────────┐
                       │  features/  fuel correction, tyre    │
                       │  age, stint state, rolling pace, etc │
                       └───────────────┬──────────────────────┘
                                       │  
                       ┌───────────────▼──────────────────────┐
                       │  models/  linear degradation         │
                       │  "s lost per lap of tyre age"        │
                       └───────────────┬──────────────────────┘
                                       │  
                       ┌───────────────▼──────────────────────┐
                       │  simulation/  deterministic          │
                       │  pit lap A vs lap B vs stay out      │
                       └───────────────┬──────────────────────┘
                                       │
                       ┌───────────────▼──────────────────────┐
                       │  serving/   "box lap 31, +1.3 s"     │
                       └──────────────────────────────────────┘
```
