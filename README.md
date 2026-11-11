# Traffic AI Control Center

Government-style **junction operations** console for three CCTV modules:

1. **Module 1** — Public transport (bus) priority / signal request  
2. **Module 2** — Unexpected stoppage / breakdown alert (fixed CCTV; vehicle-level)  
3. **Module 3** — U-turn / turn demand for adaptive signal timing  

Perception: **YOLOv8n + ByteTrack**. Live decisions: **`rules_v1`** (track heuristics) with safety gates (incident overrides bus priority). Colab arena models are engineering-only.

## Quick start

```bash
# Backend
cd app/backend
.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000

# Frontend (other terminal)
cd app/frontend
npm run dev
```

Open the Vite URL. Use **Monitor** (upload or click samples), **Delhi map**, **Alerts**, **History**, **Lab**.

## UI routes

| Route | Purpose |
|-------|---------|
| `/` | Junction monitor + analysis |
| `/map` | Real GTFS routes 425/534 + historical GPS hotspots |
| `/alerts` | Persisted alert inbox (SQLite ACK) |
| `/history` | Past analyses |
| `/lab` | Colab arena leaderboard (not ops decisions) |

## Sample videos

Click chips in the UI, or upload from `data/sample_videos/`.

## Docs

- `PROJECT_IMPROVEMENT_ROADMAP.md` — full audit roadmap  
- `docs/IMPLEMENTATION_STATUS.md` — what is done vs deferred  
- `docs/ML_VS_DL_ACCURACY.md` — leakage / rule-label warning  
- `notebooks/COLAB_RETRAIN_MODULE2_TRACK_FEATURES.md` — retrain Module 2 on track features (run in Colab)  

## Layout

- `app/backend` — FastAPI + perception + geo + SQLite  
- `app/frontend` — React ops console  
- `config/junctions/` — ROI / junction lat-lon  
- `models/` / `metrics/` — Colab exports (engineering)  
- `datasets/processed/` — GTFS/GPS + offline ML tables  
- `data/outputs/` — annotated H.264 results  
- `data/ops/ops.sqlite3` — analysis/alert history (created at runtime) 
