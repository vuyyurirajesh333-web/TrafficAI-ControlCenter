# Quarantine Move List (Phase A Freeze)

**Do not delete.** Move (copy on Drive if unsure) into quarantine folders.  
**Do not use quarantined artifacts for new training metrics or live inference.**

## Destination folders

| Destination | Purpose |
|---|---|
| `datasets/quarantine_leaky/` | Old feature/label tables |
| `metrics/quarantine_leaky/` | Old arena / per-model inflated metrics |
| `models/quarantine_leaky/` | Old 33 artifacts trained on leaky labels |

## A. Datasets — MOVE

| Current path | Move to |
|---|---|
| `datasets/processed/module1_bus/module1_ml_table.parquet` | `datasets/quarantine_leaky/module1_bus/module1_ml_table.parquet` |
| `datasets/processed/module1_bus/module1_ml_table_sample.csv` | `datasets/quarantine_leaky/module1_bus/module1_ml_table_sample.csv` |
| `datasets/processed/module2_breakdown/module2_ml_table.parquet` | `datasets/quarantine_leaky/module2_breakdown/module2_ml_table.parquet` |
| `datasets/processed/module2_breakdown/module2_ml_table_sample.csv` | `datasets/quarantine_leaky/module2_breakdown/module2_ml_table_sample.csv` |
| `datasets/processed/module3_uturn/module3_ml_table.parquet` | `datasets/quarantine_leaky/module3_uturn/module3_ml_table.parquet` |
| `datasets/processed/module3_uturn/module3_ml_table_sample.csv` | `datasets/quarantine_leaky/module3_uturn/module3_ml_table_sample.csv` |
| `datasets/processed/module3_uturn/uturn_candidate_labels.csv` | `datasets/quarantine_leaky/module3_uturn/uturn_candidate_labels.csv` |

### Datasets — KEEP in place (not leaky runtime/context)

| Path | Why keep |
|---|---|
| `datasets/processed/module1_bus/gtfs_*.parquet` | Runtime geo |
| `datasets/processed/module1_bus/gtfs_node_map_route_*.csv` | Map polylines |
| `datasets/processed/module1_bus/bus_gps_points.parquet` | Hotspots / future weak labels |
| `datasets/processed/module1_bus/bus_gps_points_sample.csv` | Dev sample |
| `datasets/processed/module1_bus/module1_ml_meta.json` | Junction seed (OK) |
| `datasets/processed/module2_breakdown/anomaly_events.csv` | Real AI City intervals (labels OK; rebuild features later) |
| `datasets/processed/module2_breakdown/anomaly_events.parquet` | Same |
| `datasets/processed/module2_breakdown/module2_ml_meta.json` | Docs (note leak warning inside) |
| `datasets/processed/module3_uturn/ngsim_tracks_sample.parquet` | Raw-ish trajectories for re-label |
| `datasets/processed/module3_uturn/module3_ml_meta.json` | Docs |

## B. Metrics — MOVE entire old arena packs

Move each module folder’s **current** arena/summary JSON/CSV into quarantine, preserving names:

| Current | Move to |
|---|---|
| `metrics/module1/*` (all arena/summary/model json+csv) | `metrics/quarantine_leaky/module1/` |
| `metrics/module2/*` | `metrics/quarantine_leaky/module2/` |
| `metrics/module3/*` | `metrics/quarantine_leaky/module3/` |

After move, leave empty `metrics/module{1,2,3}/` with a `.gitkeep` (or restore only after v2 arena export).

**Note:** Until Agent C lands, `/api/video/analyze` currently **requires** arena JSON. After local quarantine move, analyze may 404 until Phase F stub or Phase D v2 metrics. Prefer: **copy** to quarantine first on local machine, then move only when Agent C removes the hard dependency — OR move now and accept analyze break until stub. Phase A recommendation: **copy today; hard-move after Agent C health no longer requires old arena**.

## C. Models — MOVE old artifacts

| Current | Move to |
|---|---|
| `models/module1/baselines/*.joblib` | `models/quarantine_leaky/module1/baselines/` |
| `models/module1/dl/*` | `models/quarantine_leaky/module1/dl/` |
| `models/module2/baselines/*.joblib` | `models/quarantine_leaky/module2/baselines/` |
| `models/module2/dl/*` | `models/quarantine_leaky/module2/dl/` |
| `models/module3/baselines/*.joblib` | `models/quarantine_leaky/module3/baselines/` |
| `models/module3/dl/*` | `models/quarantine_leaky/module3/dl/` |

## D. Google Drive mirror

Apply the **same relative paths** under:

`MyDrive/TrafficAI_ControlCenter/`

Create if missing:
- `datasets/quarantine_leaky/`
- `metrics/quarantine_leaky/`
- `models/quarantine_leaky/`

## Local shell (when you choose to execute moves)

```bash
# Example — datasets only (safe anytime)
mkdir -p datasets/quarantine_leaky/module1_bus datasets/quarantine_leaky/module2_breakdown datasets/quarantine_leaky/module3_uturn
mv datasets/processed/module1_bus/module1_ml_table.parquet datasets/quarantine_leaky/module1_bus/
mv datasets/processed/module1_bus/module1_ml_table_sample.csv datasets/quarantine_leaky/module1_bus/
mv datasets/processed/module2_breakdown/module2_ml_table.parquet datasets/quarantine_leaky/module2_breakdown/
mv datasets/processed/module2_breakdown/module2_ml_table_sample.csv datasets/quarantine_leaky/module2_breakdown/
mv datasets/processed/module3_uturn/module3_ml_table.parquet datasets/quarantine_leaky/module3_uturn/
mv datasets/processed/module3_uturn/module3_ml_table_sample.csv datasets/quarantine_leaky/module3_uturn/
mv datasets/processed/module3_uturn/uturn_candidate_labels.csv datasets/quarantine_leaky/module3_uturn/
```

**Do not run metrics/models hard-move until you accept temporary Lab/analyze breakage, or wait for Agent C stub.**
