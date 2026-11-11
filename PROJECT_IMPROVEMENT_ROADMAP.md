# Traffic AI Control Center — Project Improvement Roadmap

**Audit date:** 2026-11-08  
**Scope:** Full repository inspection (frontend, backend, ML/CV, datasets, metrics, docs)  
**Constraint honored:** No existing source files were modified during this audit; this document is the sole deliverable.  
**Hardware context:** RTX 4050 6GB VRAM, 24GB system RAM  

---

## Executive verdict

The live system is a **working junction ops demo**, not a full AI traffic-management platform.

| Layer | Reality |
|-------|---------|
| Perception | **Real** — YOLOv8n + ByteTrack on uploaded CCTV clips |
| Module decisions (bus / breakdown / U-turn) | **Heuristics** on track kinematics — not the Colab-trained models |
| Colab 11-model arena (`models/`, `metrics/`) | **Engineering artifact only** — F1 scores decorate API payloads; models are not called for decisions |
| Delhi geography | **Present as GTFS + bus GPS** (no map tiles / GeoJSON / shapefiles in-repo) |
| Frontend | **Single-page React ops console** — real API data after upload; cosmetic “LIVE” cues; no maps/charts/router |
| Database / history / multi-junction / live CCTV | **Missing** |

**Do not claim production accuracy from arena F1.** Documented near-perfect scores (M1 ≈ 0.999, M3 = 1.0) reflect **rule-derived / leaky labels**, already explained in `docs/ML_VS_DL_ACCURACY.md`.

---

## 1. Existing project architecture

```
TrafficAI_ControlCenter/
├── app/
│   ├── backend/          # FastAPI (main.py) + perception.py + yolov8n.pt
│   └── frontend/         # React 19 + Vite 8 + axios — single App.jsx console
├── data/
│   ├── sample_videos/    # Smoke-test CCTV clips
│   ├── outputs/          # Annotated H.264 results
│   └── AboutModules/     # Design briefs (.docx)
├── datasets/processed/   # Offline ML tables (GTFS/GPS, AI City, NGSIM)
├── models/module{1,2,3}/ # Colab baselines (.joblib) + DL (.pt) — unused at runtime
├── metrics/module{1,2,3}/# Arena leaderboards
├── notebooks/            # EMPTY
└── docs/                 # OPS guide + ML vs DL accuracy notes
```

### Runtime data flow (verified)

```
Operator uploads video
        │
        ▼
YOLOv8n.track (ByteTrack)     ← only neural net on the hot path
        │
        ▼
Track kinematics (speed, heading, tortuosity, distance proxy)
        │
        ├── Module 1: bus + ETA/distance RULES
        ├── Module 2: sudden-stop + peer-motion RULES
        └── Module 3: uturn_score / heading RULES
        │
        ▼
Unified actions (incident overrides bus priority)
        + simulated signal_phase string
        + annotated *_web.mp4
        + selected_model name/F1 from arena JSON (cosmetic)
```

### API surface (all endpoints)

| Method | Path | Real behavior |
|--------|------|---------------|
| `GET` | `/api/health` | Status + sample list + `module2_model_loaded` flag |
| `GET` | `/api/samples` | List `data/sample_videos/` with module hints |
| `GET` | `/api/arena/{module}` | Static Colab leaderboard JSON |
| `POST` | `/api/video/analyze` | Sync YOLO+ByteTrack + rules; return ops JSON + video URL |
| `GET` | `/outputs/*` | Static annotated videos |

No WebSocket, no auth, no database, no job queue, no streaming progress.

---

## 2. Current implementation status

### 2.1 Implemented and working

| Capability | Evidence |
|------------|----------|
| FastAPI backend + CORS + static outputs | `app/backend/main.py` |
| YOLOv8n vehicle detect (car/motorcycle/bus/truck) | `perception.py` `get_model()` + class filter |
| ByteTrack multi-object tracking | `model.track(..., tracker="bytetrack.yaml")` |
| Annotated browser-playable H.264 video | ffmpeg / OpenH264 re-encode |
| Module 1 bus detect + priority request heuristics | distance_proxy & ETA thresholds |
| Module 2 sudden-stop incident + alert object | prior motion → stop + peer gates + red-light suppress |
| Module 3 U-turn demand heuristics | net heading + tortuosity `uturn_score` |
| Safety: incident overrides bus priority | `build_module_outputs` unified actions |
| React ops console consuming analyze API | `App.jsx` binds to real `result.*` fields |
| Sample video listing | `/api/samples` |
| Engineering arena endpoint | `/api/arena/{module}` |
| Honest docs on leakage vs live path | `docs/ML_VS_DL_ACCURACY.md`, README |

### 2.2 Partially implemented

| Item | Gap |
|------|-----|
| Module 2 XGBoost | Loaded on startup into `M2_MODEL`; **never `.predict()`’d** |
| Arena `selected_model` / `selected_model_f1` | Attached to module JSON; **does not drive decisions** |
| Signal phase / mode | **Simulated strings** from module outputs — no controller |
| Sample chips in UI | Listed from API; **not clickable** (no onClick / no server serve of samples) |
| Alert acknowledge | Client-only Set; **not persisted** |
| ffmpeg packaging | `bin/ffmpeg.zip` present; code expects `bin/ffmpeg.exe` |
| Design briefs (`AboutModules/*.docx`) | Spec richer than code (ROIs, auto-rickshaw, full TSC) |
| `tracks_preview`, `ops_summary`, `fps` | Returned by API; **UI ignores them** |
| Health/samples fetch | Once on mount; silent fail → OFFLINE / empty chips |

### 2.3 Completely missing

- Live CCTV / RTSP / webcam streaming  
- Database (alerts, analyses, junctions, history)  
- Multi-intersection coordination  
- Geospatial map UI (Leaflet/Mapbox/etc.)  
- Charts / heatmaps / historical analytics  
- Authentication / RBAC  
- Tests (unit, integration, CV regression)  
- Training scripts / reproducible notebooks (folder empty)  
- `datasets/raw/` (referenced by anomaly CSV Colab paths — not in repo)  
- Dedicated Delhi map tiles / GeoJSON / OSM extracts / shapefiles  
- Emergency vehicle priority, accident CV, violation detection  
- Reinforcement learning signal control  
- ONNX/TensorRT export path  
- Background job / async inference queue  
- Config via `.env` (API host hardcoded `127.0.0.1:8000`)  

### 2.4 Dummy / static / mock / cosmetic

| Element | Classification |
|---------|----------------|
| `ops_summary.junction = "Junction-01"` | Hardcoded |
| `ops_summary.camera = "CCTV-FIXED-01"` | Hardcoded |
| `signal_phase` / `signal_mode` | Simulated from rules |
| `selected_model` / F1 on module payloads | Cosmetic arena winner name |
| `incident_probability` | Hand formula, not model posterior |
| U-turn `gap_sec` (9.5 / 4.2) | Hardcoded fake gap |
| Frontend “LIVE ANALYSIS” | Means “has finished result”, not live stream |
| Clock | Client local time only |
| Fallback UI strings (`INACTIVE`, `NORMAL`, `Low`) | Idle placeholders before analysis |
| Distance/ETA meters | Pixel→meter proxy (`(h-y)*0.05`) — not calibrated |

**Not fake:** vehicle counts, track IDs, annotated video, bus detection presence, stop/heading kinematics after analyze.

### 2.5 Unused resources (high level)

See §4 for full inventory. Runtime uses essentially: `yolov8n.pt`, sample videos, outputs, arena JSON (names only). Almost everything under `datasets/processed/` and `models/` is offline/unused.

### 2.6 Inefficient or unnecessarily complex components

| Component | Issue | Recommendation |
|-----------|-------|----------------|
| 33 Colab classifiers (11×3) | Zero ops value; misleading if shown as “the AI” | Keep for thesis archive; isolate from ops claims |
| Module 2 RF ~6.5 MB on 12 features | Overbuilt for leaky tabular task | Retrain leaner if keeping; else drop from runtime |
| DL on 8–16 flat features | Wrong representation for CCTV ops | Prefer track-sequence / video models if DL is needed |
| Dual video encode (mp4v then H.264) | Extra latency/disk | Encode H.264 once when possible |
| Sync YOLO inside `async` route | Blocks event loop | Thread/process pool or job queue |
| Duplicate sample videos | `pexels≡intersection`, `cars≡uturn` (identical MD5) | Deduplicate |
| Unused frontend assets | `App.css`, hero/react/vite SVGs | Remove or ignore |

### 2.7 Performance bottlenecks (observed from code structure)

1. Full YOLO+ByteTrack on up to **400 frames** synchronously in the request path  
2. Double encode of annotated video  
3. Entire upload buffered into memory then temp disk  
4. No frame skipping / ROI crop / batch config exposed  
5. No explicit device pin (GPU may or may not be used depending on Ultralytics defaults/cwd)  
6. Arena JSON re-read on every analyze  
7. Outputs accumulate with no retention policy  

---

## 3. Identified problems and limitations

### Critical product honesty issues

1. API fields imply a **selected trained model** is deciding; live path is **rules**.  
2. Arena F1 must not be presented as CCTV production accuracy.  
3. “LIVE” UI language overstates batch upload analysis.  

### Technical limitations

4. No calibrated camera geometry → distance/speed/ETA are proxies.  
5. No lane/approach polygons → approach is quadrant heuristic.  
6. COCO “bus” ≠ DIMTS route identity (no plate/OCR/GTFS join live).  
7. Breakdown ≠ AI City anomaly definition (whole-frame motion tables unused).  
8. U-turn rules on short clips ≠ NGSIM-trained models.  
9. No persistence → no ops history, no authority workflow.  
10. Hardcoded localhost URLs break non-local deploy.  
11. CORS `allow_origins=["*"]` + `allow_credentials=True` is insecure.  
12. No logging framework; failures become opaque 500s.  
13. Empty `notebooks/` + missing raw videos → **non-reproducible training** from this repo alone.  
14. sklearn/xgboost version drift (GB unloadable in current venv; RF/XGB warn).  

---

## 4. Unused datasets and resources — complete inventory

### 4.1 Perception / ops (USED)

| Resource | Contents | Used how |
|----------|----------|----------|
| `app/backend/yolov8n.pt` (~6.2 MB) | COCO YOLOv8 nano | Live detect/track |
| `data/sample_videos/*.mp4` | Smoke clips | Manual upload / listing |
| `data/outputs/` | Annotated results | Written + served |
| `metrics/*/arena_*_11models.json` | Leaderboards | Names/F1 decoration + `/api/arena` |

### 4.2 Module 1 — Bus / Delhi transit (UNUSED at runtime)

| File | What it contains | Useful for | Preprocess? | Integration path |
|------|------------------|------------|-------------|------------------|
| `gtfs_stops.parquet` | **6,342** Delhi NCR stops; lat 28.42–28.87, lon 76.84–77.41 | Map stops, junction search, geospatial UI | Light: filter active stops | Leaflet markers + stop search API |
| `gtfs_routes.parquet` | **2,964** DIMTS routes (`route_type=3`) | Route picker, PT priority context | Join to trips | Route overlay on map |
| `gtfs_trips.parquet` | **78,515** trips | Schedule context | Join stop_times | ETA vs schedule |
| `gtfs_stop_times.parquet` | **3,276,092** stop times | Schedule adherence targets | Heavy; index by trip | Offline schedule DB / parquet query |
| `bus_gps_points.parquet` | **924,375** GPS/node logs (routes e.g. 425, Kalkaji/Govindpuri…) | Real bus trajectories, hotspot density | Clean times; map-match optional | Historical bus flow + lateness features |
| `bus_gps_points_sample.csv` | 8k-row sample | Dev/demo without full parquet | No | Frontend/backend prototyping |
| `module1_ml_table.parquet` | **419,453** rows; ETA/dist/speed + rule labels | Offline experiments only (labels = rules) | **Must drop rule-circular targets** before ML claims | Retrain only with non-circular targets |
| `module1_ml_meta.json` | Features + **junction_lat=28.5411, junction_lon=77.2492** | Anchor “demo junction” near CR Park / Nehru Place corridor | No | Default map center + junction config |
| `gtfs_node_map_route_425.csv` | 24 ordered stops (Sunehari Pulla → Hamdard Nagar) | Visualize Route 425 | No | Polyline from stop sequence |
| `gtfs_node_map_route_534.csv` | 22 ordered stops (ISBT Sarai Kale Khan → …) | Visualize Route 534 | No | Same |

**Delhi “map” finding (important):**  
There are **no** `.geojson`, `.shp`, `.kml`, `.osm`, or map-tile assets in this repository. Delhi geographical value is **GTFS + GPS + junction coordinates**. That is enough to build a real geospatial layer without inventing map files.

**How Delhi geo should be used**

| Use case | Feasible with current assets? | Approach |
|----------|-------------------------------|----------|
| Traffic intersection visualization | Yes (point) | Plot junction 28.5411, 77.2492 + nearby stops |
| Road network representation | Partial | Derive approximate polylines from ordered GTFS stops; optionally add OSM later |
| Intersection identification | Yes (seed) | Config file of junctions; start with meta junction + Nehru Place / CR Park stops |
| Geographical traffic monitoring | Partial | Overlay GPS density / stop dwell; needs live feeds for “now” |
| Hotspot identification | Yes (historical) | Aggregate GPS speed/dwell by stop or grid |
| Breakdown location mapping | Needs wiring | Map alert `approach` → geofenced junction approach polygons |
| Public transport route visualization | Yes | Routes 425/534 node maps + full GTFS |
| Congestion visualization | Not from current tables alone | Needs speed samples or live detectors; GPS can approximate historical |

### 4.3 Module 2 — Breakdown (UNUSED at runtime)

| File | Contents | Notes |
|------|----------|-------|
| `module2_ml_table.parquet` | **14,110** windows; OpenCV motion feats + `y` | Leak: `has_any_anomaly_in_video` |
| `anomaly_events.csv/.parquet` | **18** anomaly intervals | Paths point to **missing** Colab `datasets/raw/aicity2021_t4/...` |
| `module2_ml_meta.json` | Feature list + AI City 2021 Track4 label source | |

**Use:** Retrain track-level classifiers aligned with live sudden-stop features; do **not** plug current table into ops without de-leaking. Raw AI City videos are not in-repo.

### 4.4 Module 3 — U-turn (UNUSED at runtime)

| File | Contents | Notes |
|------|----------|-------|
| `module3_ml_table.parquet` | **1,097** vehicles; geometry feats | Labels from heading heuristic ≈ features |
| `uturn_candidate_labels.csv` | 1,102 label rows | Same heuristic |
| `ngsim_tracks_sample.parquet` | **250,000** track points / 1,102 vehicles | Good for trajectory research; US freeway domain ≠ Delhi CCTV |
| `module3_ml_meta.json` | Meta + note to refine with SUMO/inD | |

### 4.5 Models (mostly UNUSED)

| Asset | Status |
|-------|--------|
| All `models/*/dl/*.pt` + `seq_*.npy` | Unused |
| All baselines except M2 XGB | Unused |
| M2 `xgboost.joblib` | Loaded, unused |
| Arena metrics JSON/CSV | Read for decoration / engineering API |

### 4.6 Other

| Asset | Status |
|-------|--------|
| `data/AboutModules/*.docx` | Design specs — not loaded by code |
| `notebooks/` | Empty |
| Duplicate samples / outputs | Waste space; cleanup candidate |
| Frontend `App.css`, unused assets | Dead |

---

## 5. ML/DL model improvement recommendations

### Guiding principle

Keep **YOLO + ByteTrack** as the shared perception backbone. Improve each module with **calibrated geometry + better decision models that consume track features**. Do **not** revive the leaky tabular arena as “production AI.” Do **not** add RL / Transformers / GAT unless the data problem is fixed first.

---

### Module 1 — Smart Traffic Signal Priority for Public Transport

| Field | Detail |
|-------|--------|
| **1. Existing approach** | COCO bus class via YOLO; distance proxy `(h−y)×0.05`; ETA from proxy speed; grant if dist&lt;140 m and ETA&lt;60 s; arena name attached |
| **2. Recommended approach** | (A) Keep YOLO bus detect + ByteTrack. (B) Add **approach ROI polygons** + optional homography for meter calibration. (C) Join **GTFS schedule + GPS** when available for true lateness. (D) Decision: keep explicit priority **policy rules** (auditable for authorities); optionally add a small classifier only for *schedule adherence / passenger-proxy / downstream queue*, not for re-learning ETA rules. |
| **3. Why beneficial** | Current ETA is uncalibrated; no route identity; GTFS assets unused; trees in arena only re-learned the rule |
| **4. Suitable datasets** | In-repo: `gtfs_*`, `bus_gps_points`, route 425/534 maps. External: more Delhi bus GPS, junction CCTV with bus annotations |
| **5. Expected improvement** | Fewer false priority grants from distant/slow false geometry; measurable schedule-based priority; **do not quote arena F1 as gain** |
| **6. Compute** | Negligible beyond YOLO; GTFS queries CPU/RAM light |
| **7. Complexity** | Medium (calibration + GTFS service); High if full AVL integration |
| **8. Metrics** | Priority precision/recall vs operator labels; mean granted extension when bus truly in window; false grant rate; schedule lateness MAE if predicting delay |
| **9. Worth it?** | **Yes — High priority** for Module 1 realism |

**YOLO upgrade?** Optional later: fine-tune YOLO on Delhi buses / add autorickshaw. Start with **yolov8s or keep n** + calibration before jumping to larger detectors.

---

### Module 2 — Vehicle Breakdown Detection and Alert

| Field | Detail |
|-------|--------|
| **1. Existing approach** | Track speeds → sudden_stop_candidate; suppress all-stopped scenes; duration + moving peers; hand-crafted `incident_probability` |
| **2. Recommended approach** | Keep rule gates as **safety baseline**. Add a **track-feature classifier** (XGBoost or small TCN on track sequences: prior max speed, stop duration, peer movers, lane occupancy, bbox size stability). Wire the already-loaded XGB **only after** retraining on track features **without** `has_any_anomaly_in_video`. Use model probability **with** rule gates (AND logic), not instead of them. |
| **3. Why beneficial** | Aligns ML with live perception; removes leakage; reduces FP at congested red lights while catching true stoppages |
| **4. Suitable datasets** | Rebuild from ByteTrack on AI City (if raw videos obtained) + local labeled clips; current `module2_ml_table` only after dropping leak features and preferably replacing features |
| **5. Expected improvement** | Better calibrated probabilities; lower FP vs pure rules on edge cases — **must measure on held-out videos**, not old arena F1 |
| **6. Compute** | Tiny vs YOLO; fits easily on RTX 4050 |
| **7. Complexity** | Medium |
| **8. Metrics** | Alert precision/recall @ ops threshold; time-to-alert; FP/hour; vs rule-only baseline |
| **9. Worth it?** | **Yes — High** (after de-leak retrain). Blindly calling current joblib: **No** |

**Not recommended now:** end-to-end video Transformers for breakdown — too heavy for 6GB and weak labels.

---

### Module 3 — Smart U-Turn Management

| Field | Detail |
|-------|--------|
| **1. Existing approach** | Heading unwrap + tortuosity → `uturn_score`; thresholds; hardcoded gap seconds |
| **2. Recommended approach** | Keep geometric score as detector of turn-like motion. Add **U-turn / left-pocket ROI** masks per junction. Replace fake `gap_sec` with **gap estimation from conflicting approach track speeds/counts**. Independent labels (manual or SUMO/inD) if training a classifier; do **not** retrain on NGSIM features that equal labels. Domain: prefer Delhi/junction CCTV over NGSIM freeway. |
| **3. Why beneficial** | Current gap is fictional; NGSIM≠Delhi; F1=1.0 arena is invalid for claims |
| **4. Suitable datasets** | Local annotated U-turns; optional inD/SUMO; NGSIM only for algorithm prototyping |
| **5. Expected improvement** | Fewer false U-turn schedules; gap tied to real conflicting flow |
| **6. Compute** | Negligible |
| **7. Complexity** | Medium |
| **8. Metrics** | U-turn event P/R; scheduled window usefulness (operator score); conflict near-miss proxy |
| **9. Worth it?** | **Yes — Medium-High** for ROI + real gap; **No** for reusing perfect arena trees |

---

### Cross-cutting CV techniques — keep / improve / defer

| Technique | Verdict | Rationale |
|-----------|---------|-----------|
| YOLO vehicle detection | **Keep / optionally upgrade n→s** | Already core; works |
| ByteTrack | **Keep** | Already integrated; adequate for fixed CCTV |
| Vehicle classification | **Improve lightly** | COCO classes OK; add autorickshaw later via fine-tune |
| Trajectory analysis | **Improve** | Already present; add ROI + calibration |
| Traffic density estimation | **Add (simple)** | Count vehicles in approach ROIs from existing tracks |
| Speed estimation | **Improve** | Needs homography / reference length |
| Stopped / breakdown | **Improve** (rules + track ML) | See Module 2 |
| Bus detection / ID | **Detect keep; ID add** | ID via GTFS/GPS or ReID later — OCR optional |
| Traffic flow prediction | **Defer** | Needs historical time series DB first |
| RL adaptive signals | **Defer (research)** | No simulator loop / reward infra; unsafe for demo claims |
| Time-series forecasting | **Optional later** | Use GPS aggregates after geo pipeline |
| Anomaly detection | **Narrow** | Prefer supervised track stoppage over generic video anomaly |

---

### Per-proposal scorecard (summary)

| Proposal | Worth implementing? | Priority |
|----------|---------------------|----------|
| Camera calibration + approach ROIs | Yes | Critical |
| Wire de-leaked Module 2 track classifier behind rules | Yes | High |
| GTFS/GPS map + bus priority context | Yes | High |
| Density from ROI counts | Yes | High |
| Real conflict gap for U-turn | Yes | High |
| YOLO fine-tune Delhi classes | Maybe | Medium |
| ONNX FP16 export of YOLO | Yes | Medium |
| Revive arena DL/trees as-is | **No** | — |
| RL traffic lights | Later research | Low now |
| Full accident/violation suite | Selective later | Low–Medium |

---

## 6. Model optimization recommendations (RTX 4050 6GB / 24GB RAM)

| Technique | Recommendation | Why |
|-----------|----------------|-----|
| **ONNX Runtime** | **Yes** — export YOLOv8n/s to ONNX | Stable speedup on Windows; simpler than TensorRT |
| **TensorRT** | Optional later | More ops pain on Windows; 6GB benefits but higher complexity |
| **FP16** | **Yes** on CUDA | Easy win with Ultralytics `half=True` / ONNX FP16 |
| **INT8** | Later, with calibration set | Risk to small-object bus recall |
| **Pruning** | Low priority | YOLO-n already small |
| **Batch inference** | Only if multi-camera | Single-clip demo: low value |
| **Frame skipping** | **Yes** (`stride=2` or adaptive) | Analyze already has `stride`; expose in API |
| **ROI-based inference** | **Yes** after polygons | Cuts pixels → FPS↑ |
| **GPU/CPU** | Explicit `device=0` when CUDA available | Avoid silent CPU fallback |
| **Async processing** | **Critical** | `run_in_executor` or RQ/Celery + job id + poll/WS |
| **Max frames** | Make configurable; default maybe 300–400 | Balance UX vs latency |
| **Single H.264 encode** | Prefer ffmpeg directly | Drop intermediate mp4v when possible |

**Realistic target on this hardware:** maintain interactive single-stream analysis; do not plan multi-4K streams without ROI + skip + ONNX.

---

## 7. Advanced feature recommendations

### A. Essential improvements

1. Honest API/UI labeling (rules vs models; batch vs live)  
2. Async analyze jobs + progress  
3. Config/env for API URLs; relative `/outputs/` URLs  
4. Approach ROIs + optional homography  
5. Persist analyses/alerts (SQLite is enough initially)  
6. Clickable sample videos (serve from backend)  
7. Structured logging + proper error responses  
8. Deduplicate samples; clean dead code (`M2_MODEL` or wire it)  

### B. High-impact AI/ML improvements

1. Track-feature breakdown classifier (de-leaked) behind rules  
2. Delhi GTFS/GPS geospatial layer + route 425/534 visualization  
3. Traffic density / queue length from ROI track counts  
4. Calibrated speed/ETA for bus priority  
5. Conflict-based U-turn gap estimation  
6. YOLO ONNX FP16 optimization  
7. Model performance panel fed by **video-labeled** eval, not arena leakage F1  

### C. Advanced research-level features

1. Multi-intersection coordination  
2. Congestion forecasting (time-series on stored densities)  
3. RL / MPC adaptive signal control in simulation (SUMO) then shadow mode  
4. Accident detection beyond stoppage (collision kinematics)  
5. Cross-camera ReID  

### D. Optional features

1. Emergency vehicle priority (siren AV + class)  
2. Violation detection (stop-line, wrong-way) — high annotation cost  
3. Intelligent city-wide routing — needs network graph + live speeds  
4. Autorickshaw / specialized Delhi classes  
5. V2I broadcast integration (beyond string channels)  

**Only recommend what fits:** Build A → B before C/D. Avoid stacking RL + forecasting + violations until perception is calibrated and data is persisted.

---

## 8. Frontend / UI improvements

### Current UI audit (verified)

- Single route; no React Router  
- Real data after `/api/video/analyze`  
- No charts, maps, or mock KPI arrays  
- Non-functional sample chips  
- Cosmetic LIVE badge  
- Duplicate M1/M2/M3 status (rail + cards)  
- Dead `App.css` / unused assets  
- Arena mentioned in footer text; not fetched  
- Missing loading states for health/samples  

### Target: professional traffic operations center (truthful)

| View | Purpose | Data source requirement |
|------|---------|-------------------------|
| Junction monitor (enhance current) | Upload/live clip, overlays, actions | Existing analyze + ROIs |
| Delhi network map | Junctions, GTFS routes 425/534, stops | GTFS/GPS APIs — **no fake congestion** |
| Alerts inbox | Persistent alerts, ack/resolve | DB-backed |
| Bus priority board | Active requests + schedule lateness | Tracks + GTFS |
| Breakdown desk | Incident cards with map pin | Module 2 + geo |
| Signal status | Explicitly labeled **simulated** until TSC exists | Rules engine |
| Analytics | Historical counts/densities | Only after persistence |
| Engineering / model lab (separate page) | Arena + eval metrics | Keep off primary ops wall |

### UI rules for this project

- Never show fabricated live KPIs.  
- If a metric is proxy/simulated, label it.  
- Sample chips must either run analysis or be removed.  
- Prefer one ops composition; add map as a second primary surface, not a card farm.  

---

## 9. Backend improvements

| Area | Current | Recommended |
|------|---------|-------------|
| Structure | 2 Python files | Packages: `api/`, `perception/`, `decision/`, `geo/`, `jobs/` — incremental |
| Async | Blocking analyze | Job queue + `GET /api/jobs/{id}` |
| Models lifecycle | Global YOLO; dead XGB | Explicit device; lazy load; remove or wire XGB |
| DB | None | SQLite → Postgres if needed |
| Config | Hardcoded paths/URLs | pydantic-settings + `.env` |
| Logging | None | structlog/logging + request ids |
| Security | Open CORS, no auth | Restrict origins; optional API key for demos |
| Testing | None | Perception unit tests + golden video fixtures |
| Monitoring | None | Timing metrics: detect ms, encode ms, total |
| API design | One fat analyze | Split perception summary vs decision; version `/api/v1` |
| Outputs | Unbounded | Retention/TTL cleanup |

**Do not rewrite working perception** for style. Extract modules when adding jobs/geo/DB.

---

## 10. Recommended future architecture

```
                    ┌─────────────────────────┐
                    │  Ops Console (React)    │
                    │  Monitor | Map | Alerts │
                    └───────────┬─────────────┘
                                │ REST (+ later WS)
                    ┌───────────▼─────────────┐
                    │  API Gateway (FastAPI)  │
                    │  auth/config/validate   │
                    └─┬─────────┬─────────┬───┘
                      │         │         │
           ┌──────────▼──┐ ┌────▼────┐ ┌──▼──────────┐
           │ Job Worker  │ │ Geo/GTFS│ │ Alert Store │
           │ YOLO+Track  │ │ service │ │ (SQLite)    │
           │ Decisions   │ │         │ │             │
           └──────┬──────┘ └────┬────┘ └─────────────┘
                  │             │
           ┌──────▼──────┐     │
           │ Artifact FS │◄────┘ map overlays use junction cfg
           │ outputs/    │
           └─────────────┘

Offline (separate): training notebooks → metrics/models lab
Ops never imports leaky arena as decision truth.
```

**Shared perception features → three decision heads** remains the right modular shape.

---

## 11. Implementation priorities (phased roadmap)

### Phase 1 — Fix existing problems

| ID | Task | Priority | Complexity | Dependencies | Likely files | Outcome | Acceptance criteria |
|----|------|----------|------------|--------------|--------------|---------|---------------------|
| P1.1 | Remove or clearly mark cosmetic `selected_model` as non-decisive; document in API | Critical | Low | None | `perception.py`, `main.py`, docs | Honest API | Response field renamed or `decision_engine: "rules_v1"` present; docs match |
| P1.2 | Fix annotated URL to relative `/outputs/...` + frontend `API` from env | Critical | Low | None | `main.py`, `App.jsx`, `.env.example` | Portable deploy | Works when host ≠ 127.0.0.1 |
| P1.3 | Run blocking inference in threadpool / background job | Critical | Medium | None | `main.py`, new `jobs.py` | API stays responsive | Health responds during analyze |
| P1.4 | Make sample chips load/run sample via backend static or analyze-by-name | High | Low–Med | P1.2 | `main.py`, `App.jsx` | One-click smoke tests | Click sample → analysis without manual file pick |
| P1.5 | Resolve dead `M2_MODEL` (remove load **or** gate behind flag until retrain) | High | Low | None | `main.py` | No false “model loaded” implication | Health flag accurate; no unused load **or** real predict path |
| P1.6 | Dedupe sample videos; document keep-list | Medium | Low | None | `data/sample_videos/` | Less confusion | Unique MD5s only |
| P1.7 | Add logging + catch perception errors → 4xx/5xx detail | High | Low | None | `main.py`, `perception.py` | Debuggable failures | Failed video returns JSON error, not opaque 500 |
| P1.8 | CORS/credentials hardening | Medium | Low | None | `main.py` | Safer defaults | Explicit allowed origins |
| P1.9 | Remove dead frontend CSS/assets or stop shipping them | Low | Low | None | `App.css`, assets | Cleaner repo | Build unaffected; unused files gone |
| P1.10 | Extract ffmpeg.zip → `bin/ffmpeg.exe` or document imageio path | Medium | Low | None | `bin/`, docs | Reliable encode | Browser video plays without manual DLL hunt |

### Phase 2 — Improve existing ML/DL models

| ID | Task | Priority | Complexity | Dependencies | Likely files | Outcome | Acceptance criteria |
|----|------|----------|------------|--------------|--------------|---------|---------------------|
| P2.1 | Define junction config: ROIs, approaches, optional homography | Critical | Medium | None | new `config/junctions/`, `perception.py` | Calibrated geometry | Distance/approach use ROI; config-driven |
| P2.2 | Export track-feature dataset from perception for M2 | High | Medium | P2.1 | new `scripts/export_tracks.py` | Train/serve parity | Parquet of track features from sample videos |
| P2.3 | Retrain M2 without leak features; GroupKFold by video | High | Medium | P2.2, raw or local labels | `notebooks/`, `models/module2/` | Honest M2 model | Rule baseline F1 reported; leak feature absent |
| P2.4 | Integrate M2 model behind rule gates | High | Medium | P2.3 | `perception.py`, `main.py` | Hybrid decision | Alert only if rules pass AND p≥threshold |
| P2.5 | Bus priority: calibrated ETA + keep auditable rules | High | Medium | P2.1 | `perception.py` | Better grants | Unit tests on synthetic trajectories |
| P2.6 | U-turn: ROI + conflict-gap from opposing tracks | High | Medium | P2.1 | `perception.py` | Real gap_sec | gap derived from tracks; no constant 9.5/4.2 |
| P2.7 | Optional YOLO n→s or ONNX FP16 | Medium | Medium | GPU drivers | `perception.py`, `models/` | Faster/more accurate detect | Benchmark FPS + bus recall on samples |
| P2.8 | Do **not** re-enable arena trees/DL for ops decisions | Critical | — | — | docs | Avoid regression | Ops path still rules(+optional M2) |

### Phase 3 — Integrate unused datasets and maps

| ID | Task | Priority | Complexity | Dependencies | Likely files | Outcome | Acceptance criteria |
|----|------|----------|------------|--------------|--------------|---------|---------------------|
| P3.1 | GTFS stop/route read API | High | Medium | None | new `geo/gtfs.py`, `main.py` | Serve Delhi stops/routes | `/api/geo/stops` returns real lat/lon |
| P3.2 | Map UI: junction + routes 425/534 + stops | High | Medium | P3.1 | frontend map page, Leaflet | Geospatial ops | Map shows real GTFS points; no fake congestion layer |
| P3.3 | Historical GPS hotspot aggregation (offline job) | Medium | Medium | P3.1 | scripts + parquet | Hotspot layer from real GPS | Grid/stop stats from `bus_gps_points` |
| P3.4 | Bind alerts to junction geo | Medium | Low | P2.1, P3.2 | decision + frontend | Breakdown on map | Alert pin at configured junction approach |
| P3.5 | Schedule lateness feature for bus module (when GPS time aligns) | Medium | High | P3.1 | decision M1 | Priority uses lateness | Lateness computed from GTFS stop_times join |
| P3.6 | Archive/quarantine leaky ML tables with README | Medium | Low | None | `datasets/README.md` | Prevent misuse | Clear “do not use for production metrics” |

**Note:** If separate Delhi map images/tiles exist outside this repo, add them under `data/maps/` and wire as basemap; **they are not present now.**

### Phase 4 — Advanced AI features

| ID | Task | Priority | Complexity | Dependencies | Likely files | Outcome | Acceptance criteria |
|----|------|----------|------------|--------------|--------------|---------|---------------------|
| P4.1 | Approach density / queue length metrics | High | Medium | P2.1 | perception + API + UI | Real density from counts | Density changes with video content |
| P4.2 | Persist analyses & alerts (SQLite) | High | Medium | P1.3 | new `db.py`, models | History | Refresh keeps past alerts |
| P4.3 | Optional live webcam/RTSP worker | Medium | High | P1.3, P2.7 | worker service | Near-real-time | Sustained FPS documented on 4050 |
| P4.4 | Eval harness vs labeled clips | High | Medium | P2.x | `tests/`, `metrics/ops_eval/` | Honest metrics | Confusion matrix for M1/M2/M3 events |
| P4.5 | Emergency vehicle class/priority | Low–Med | High | detect data | perception + rules | Extra module | Only with labeled data |
| P4.6 | SUMO shadow signal simulation | Low | High | P4.1 | sim/ | Research adaptive control | Sim metrics, not claimed field deployment |
| P4.7 | Congestion forecast | Low | High | P4.2 history | forecasting/ | Research | Backtested MAE on stored series |

### Phase 5 — Upgrade frontend and visualizations

| ID | Task | Priority | Complexity | Dependencies | Likely files | Outcome | Acceptance criteria |
|----|------|----------|------------|--------------|--------------|---------|---------------------|
| P5.1 | Split components; add routes (Monitor / Map / Alerts / Lab) | High | Medium | None | `src/` restructure | Scalable UI | Routes work; ops default = Monitor |
| P5.2 | Label simulated signal state explicitly | Critical | Low | None | `App.jsx` | No false TSC claim | UI text “Simulated phase” |
| P5.3 | Show tracks_preview table / timeline | Medium | Low | None | Monitor view | Use API data already returned | Track table visible |
| P5.4 | Alerts inbox with server ack | High | Medium | P4.2 | Alerts page | Authority workflow | Ack persists |
| P5.5 | Delhi map page (real GTFS only) | High | Medium | P3.2 | Map page | Geo monitoring | No invented traffic tiles |
| P5.6 | Engineering Lab page for arena (optional) | Low | Low | None | Lab page | Thesis without polluting ops | Arena not on main wall |
| P5.7 | Loading/error/retry UX for health & jobs | High | Low | P1.3 | frontend | Robust console | Failures visible + retry |
| P5.8 | Density / queue charts **from stored real series** | Medium | Medium | P4.1–P4.2 | charts | Analytics without fakes | Empty state until data exists |

### Phase 6 — Performance optimization and testing

| ID | Task | Priority | Complexity | Dependencies | Likely files | Outcome | Acceptance criteria |
|----|------|----------|------------|--------------|--------------|---------|---------------------|
| P6.1 | YOLO ONNX + FP16 benchmark | High | Medium | GPU | perception | Faster inference | Documented ms/frame on 4050 |
| P6.2 | Expose stride/max_frames/ROI in API | Medium | Low | P2.1 | API | Tunable latency | Query params affect runtime |
| P6.3 | Single-pass H.264 encode | Medium | Medium | ffmpeg | perception | Less IO | One encode step in happy path |
| P6.4 | Output retention policy | Low | Low | None | jobs/cleanup | Disk hygiene | Old annot_* pruned |
| P6.5 | Unit tests for decision rules | Critical | Medium | None | `tests/test_decisions.py` | Regressions caught | Synthetic tracks cover M1/M2/M3 + override |
| P6.6 | Golden video smoke CI (optional local) | Medium | Medium | samples | CI script | Demo stability | Exit 0 on expected module activation |
| P6.7 | Pin requirements versions | High | Low | None | `requirements.txt` | Reproducible env | joblib models load without `_loss` break |
| P6.8 | Memory profiling under 6GB VRAM | Medium | Low | P6.1 | docs | Fits hardware | Peak VRAM recorded &lt; 5.5GB for chosen model |

---

## 12. Dependencies and hardware requirements

### Current runtime (effective)

- Python: FastAPI, Ultralytics, OpenCV, joblib, (torch via ultralytics)  
- Node: React 19, Vite 8, axios  

### Recommended additions (incremental)

| Dependency | Phase | Purpose |
|------------|-------|---------|
| `pydantic-settings` | 1 | Config |
| `onnxruntime-gpu` or `onnxruntime` | 6 | YOLO accelerate |
| `sqlalchemy` or `aiosqlite` | 4 | Persistence |
| `leaflet` + `react-leaflet` | 3/5 | Delhi map |
| `pytest` | 6 | Tests |
| Optional: `redis` + worker | 4 | Multi-job queue |

### Hardware guidance (RTX 4050 6GB / 24GB RAM)

| Workload | Fit? |
|----------|------|
| YOLOv8n + ByteTrack single 1080p stream | Yes |
| YOLOv8s FP16 single stream | Yes (monitor VRAM) |
| YOLOv8m multi-stream | Risky |
| Train large Transformers locally | Prefer Colab; keep local for fine-tune small models |
| TensorRT | Optional; ONNX first |
| Leaflet + SQLite + API | Trivial on 24GB RAM |

---

## 13. Testing and validation strategy

1. **Rule unit tests** — synthetic tracks for bus grant/deny, breakdown FP suppress, U-turn schedule, incident override.  
2. **Perception smoke** — fixed sample videos; assert module activation expectations from `OPS_CONSOLE_GUIDE.md`.  
3. **Calibration tests** — after homography, known pixel lengths → meter error bound.  
4. **ML eval protocol** — video-level GroupKFold; always publish **rule-only baseline** beside model metrics; never use leaky features.  
5. **Ops acceptance** — UI shows only backend-backed numbers; simulated fields labeled.  
6. **Performance** — p50/p95 analyze latency; FPS; VRAM peak on 4050.  
7. **Regression** — golden annotated hashes optional; decision JSON snapshots.  

---

## 14. What NOT to do

- Do not present Colab arena F1 as live CCTV accuracy.  
- Do not replace working YOLO+ByteTrack with a heavier stack without benchmarks.  
- Do not add fake real-time dashboards.  
- Do not invent Delhi map files — use GTFS/GPS or add real OSM/geo assets deliberately.  
- Do not implement RL signal control as a “deployed” feature without simulation + safety case.  
- Do not wire current Module 2 XGBoost until de-leaked retraining is done.  
- Do not rewrite the whole backend for aesthetics; evolve by extraction.

---

## 15. Suggested approval sequence

If you approve this roadmap, recommended execution order for the next engineering sprint:

1. **Phase 1** honesty + async + samples + logging  
2. **Phase 2.1** junction ROI/calibration  
3. **Phase 3.1–3.2** GTFS Delhi map (unlocks unused data)  
4. **Phase 2.2–2.4** Module 2 hybrid ML  
5. **Phase 5** UI restructuring on real APIs  
6. **Phase 6** ONNX + tests  

---

## Appendix A — Arena metrics (documented, not production claims)

| Module | Best classical F1 (arena JSON) | Best DL F1 (approx) | Interpretation |
|--------|--------------------------------|---------------------|----------------|
| M1 Bus | RF/GB ≈ **0.999** | TCN ≈ **0.94** | Rule-label recovery |
| M2 Breakdown | XGBoost ≈ **0.95** | TCN ≈ **0.83** | Leak via video-level flag |
| M3 U-turn | RF/GB/XGB = **1.0** | Transformers ≈ **0.65–0.74** | Label≈feature / tiny set |

Source: `metrics/module*/arena_module*_11models.json`, `docs/ML_VS_DL_ACCURACY.md`.

---

## Appendix B — Sample video integrity

| File | MD5 | Note |
|------|-----|------|
| `test_module3_pexels.mp4` | `920501f8…` | Identical to intersection |
| `test_module3_intersection.mp4` | `920501f8…` | Duplicate |
| `test_module3_cars.mp4` | `e919de11…` | Identical to uturn |
| `test_module3_uturn.mp4` | `e919de11…` | Duplicate |

---

*End of audit roadmap. Awaiting approval before any implementation.*
