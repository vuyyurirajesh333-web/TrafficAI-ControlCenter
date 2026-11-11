# Datasets — usage policy

Phase A freeze: feature schemas live in `config/schemas/`.  
Leaky ML tables must move to `datasets/quarantine_leaky/` per `docs/architecture_freeze/QUARANTINE_MOVE_LIST.md`.

## Runtime (used by the live API)

| Path | Role |
|------|------|
| `processed/module1_bus/gtfs_*.parquet` | Delhi GTFS stops/routes/trips/stop_times for `/api/geo/*` |
| `processed/module1_bus/gtfs_node_map_route_*.csv` | Route polylines for map |
| `processed/module1_bus/bus_gps_points.parquet` | Historical GPS hotspots (`/api/geo/hotspots`) |
| `processed/module1_bus/module1_ml_meta.json` | Junction lat/lon seed |

## Offline / engineering only (do **not** quote as live CCTV accuracy)

| Path | Issue |
|------|-------|
| `processed/module1_bus/module1_ml_table*` | Labels from rule engine on ETA/distance features |
| `processed/module2_breakdown/*` | Leak via `has_any_anomaly_in_video`; raw AI City videos not in repo |
| `processed/module3_uturn/*` | NGSIM heuristic labels ≈ features; domain ≠ Delhi CCTV |

See `docs/ML_VS_DL_ACCURACY.md` and Colab retrain notes in `notebooks/`.

## Missing from this checkout

- `datasets/raw/` (AI City videos referenced by `anomaly_events.csv` Colab paths)
- Full NGSIM dumps beyond the processed sample parquet
