# Implementation status (post-roadmap sprint)

Updated as of the multi-phase implementation pass.

## Phase 1 — Fix existing problems — DONE

Honesty fields, relative URLs, sample analyze, thread offload, CORS, deduped samples, dead asset cleanup, ffmpeg notes, tests.

## Phase 2 — ML/DL improvements — PARTIAL (infra done; Colab retrain for you)

| Item | Status |
|------|--------|
| Junction ROI config | Done (`config/junctions/junction_01.json`) |
| Approach density | Done |
| Conflict-based U-turn gap | Done |
| Track feature export script | Done (`scripts/export_track_features.py`) |
| Module 2 hybrid XGB | **Waiting on your Colab retrain** — see `notebooks/COLAB_RETRAIN_MODULE2_TRACK_FEATURES.md` |
| YOLO ONNX | Not exported yet (documented next step; keep yolov8n.pt) |

## Phase 3 — Datasets / maps — DONE

GTFS stops/routes/hotspots APIs; Delhi map UI; `datasets/README.md` quarantine notes.

## Phase 4 — Advanced features — PARTIAL

| Item | Status |
|------|--------|
| SQLite analysis/alert history | Done |
| Persisted alert ACK | Done |
| Density metrics | Done |
| Live RTSP / RL / forecasting | Deferred (needs feeds / research) |

## Phase 5 — Frontend — DONE (ops shell)

Routes: Monitor, Delhi map, Alerts, History, Lab. No fake congestion layer.

## Phase 6 — Perf / tests — PARTIAL

| Item | Status |
|------|--------|
| `max_frames` / `stride` API params | Done |
| Backend unit tests | 17 passing |
| ONNX / VRAM benchmarks | Pending on your GPU run |
| Requirement pins | Soft (added pyarrow/pytest); full pins optional |

## How to run

```bash
# backend
cd app/backend
.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000

# frontend
cd app/frontend
npm run dev
```

```bash
cd app/backend
.venv/Scripts/python.exe -m pytest tests/ -v
```
