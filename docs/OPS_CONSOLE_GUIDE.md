# Junction Operations Console — Operator Guide

## What this system does

Operators upload fixed-CCTV clips. The system runs **YOLO + ByteTrack**, then three decision modules:

| Module | Question answered | Typical action |
|--------|-------------------|----------------|
| 1 Bus priority | Is a public bus approaching with short ETA? | Request green extension |
| 2 Incident alert | Did a vehicle **move then stop** while peers still move? | Page control room / patrol / V2I |
| 3 U-turn demand | Is trajectory geometry consistent with U-turn demand? | Schedule protected U-turn window |

**Safety rule:** an active Module 2 incident **outranks** Module 1 bus priority (green extension is deferred).

## What it does *not* do

- It does **not** treat the camera as an “object that is stationary.” Fixed CCTV is assumed.
- It does **not** alert on red-light / all-lanes-stopped congestion (peers must still be moving, or stop duration must be long).
- The main UI does **not** show the 11-model Colab arena — that is engineering-only (`/api/arena/{module}`).

## How to test

1. Start API: `uvicorn main:app --reload --port 8000` from `app/backend`  
2. Start UI: `npm run dev` from `app/frontend`  
3. Upload clips from `data/sample_videos/`  

Expected smoke checks:

- `test_module1_bus.mp4` → Module 1 active, bus ID present  
- `test_module2_breakdown.mp4` → Module 2 alert when a vehicle stopped after motion  
- `test_module3_pexels.mp4` → Module 3 demand / U-turn schedule when curvature is strong  

## Alert acknowledgement

Open alerts appear as a red banner. **Acknowledge** dismisses them from the banner for this session (analysis result remains).
