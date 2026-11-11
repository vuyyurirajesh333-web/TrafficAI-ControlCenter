from perception import (
    PERCEPTION_ENGINE,
    analyze_and_annotate,
)
from decision_pipeline import DECISION_ENGINE, run_decisions
from module_inference import ModelInferenceError
from model_registry import (
    health_flags,
    list_models,
    load_clean_arena,
    load_clean_arena_bundle,
    get_model_entry,
)
from fastapi.staticfiles import StaticFiles

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import asyncio
import tempfile
import os
import logging
import re

from junction_config import list_junctions, load_junction
from db import (
    acknowledge_alert,
    get_analysis,
    init_db,
    list_alerts,
    list_analyses,
    save_analysis,
)
import geo as geo_service

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("traffic_ai")

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_SAFE_SAMPLE_NAME = re.compile(r"^[\w.\- ]+$")


ROOT = Path(__file__).resolve().parents[2]

OUTPUTS = ROOT / "data" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

METRICS = ROOT / "metrics"
SAMPLES = ROOT / "data" / "sample_videos"
DEMO_TMP = Path(tempfile.gettempdir()) / "traffic_ai_demo_outputs"
DEMO_TMP.mkdir(parents=True, exist_ok=True)

SAMPLE_LABELS = {
    "test_module1_bus.mp4": "Bus Priority Demo",
    "test_module2_breakdown.mp4": "Breakdown Alert Demo",
    "test_module3_uturn.mp4": "U-Turn Demo",
}

_DEFAULT_CORS = "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:4173,http://localhost:4173"
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ALLOW_ORIGINS", _DEFAULT_CORS).split(",")
    if o.strip()
]


app = FastAPI(
    title="Traffic AI Control Center API",
    description="Government junction operations API — bus priority, incident alerts, U-turn demand",
    version="1.2.0",
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUTS)),
    name="outputs"
)
app.mount(
    "/demo-outputs",
    StaticFiles(directory=str(DEMO_TMP)),
    name="demo_outputs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    init_db()
    flags = health_flags()
    logger.info(
        "Ops SQLite ready; decision_engine=%s m1_loaded=%s m2_loaded=%s m3_loaded=%s",
        DECISION_ENGINE,
        flags.get("module1_model_loaded"),
        flags.get("module2_model_loaded"),
        flags.get("module3_model_loaded"),
    )


def load_arena(module: str):
    if module not in {"module1", "module2", "module3"}:
        raise HTTPException(400, "module must be module1|module2|module3")
    bundle = load_clean_arena_bundle(module)
    if not bundle.get("models"):
        raise HTTPException(
            404,
            f"missing clean/legacy arena JSON for {module}",
        )
    return bundle


@app.get("/api/health")
def health():
    samples = []
    if SAMPLES.exists():
        samples = sorted(
            p.name for p in SAMPLES.iterdir()
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            and p.stat().st_size > 10_000
        )
    flags = health_flags()
    return {
        "status": "ok",
        "service": "Traffic AI Junction Operations",
        "root": str(ROOT),
        "metrics_exists": METRICS.exists(),
        "decision_engine": DECISION_ENGINE,
        "perception_engine": PERCEPTION_ENGINE,
        "sample_videos": samples,
        "camera_assumption": "stationary_fixed_cctv",
        "cors_allow_origins": CORS_ALLOW_ORIGINS,
        "analyze_executor": "asyncio.to_thread",
        "geo": geo_service.geo_summary(),
        **flags,
    }


@app.get("/api/samples")
def list_samples():
    if not SAMPLES.exists():
        return {"samples": []}
    rows = []
    for p in sorted(SAMPLES.iterdir()):
        if p.suffix.lower() in VIDEO_SUFFIXES and p.stat().st_size > 10_000:
            hint = "general"
            name = p.name.lower()
            if "module1" in name or "bus" in name:
                hint = "module1_bus"
            elif "module2" in name or "break" in name:
                hint = "module2_breakdown"
            elif "module3" in name or "uturn" in name or "turn" in name:
                hint = "module3_uturn"
            rows.append({
                "filename": p.name,
                "label": SAMPLE_LABELS.get(p.name) or p.name.replace("test_", "").replace(".mp4", ""),
                "bytes": p.stat().st_size,
                "hint": hint,
                "demo_chip": p.name in SAMPLE_LABELS,
            })
    # Prefer friendly demo chips first
    rows.sort(key=lambda r: (0 if r.get("demo_chip") else 1, r["label"]))
    return {"samples": rows}


@app.get("/api/arena/{module}")
def arena(module: str):
    """Lab model comparison metrics (demo mode)."""
    bundle = load_arena(module)
    return {
        "module": module,
        "models": bundle.get("models") or [],
        "photo_lab": True,
        "source_path": bundle.get("source_path"),
        "deployment_winner": bundle.get("deployment_winner"),
        "selection_rule": bundle.get("selection_rule"),
        "label_meaning": bundle.get("label_meaning"),
        "missing_artifacts": bundle.get("missing_artifacts") or [],
        "source": "lab_comparison_metrics",
    }


@app.get("/api/models/{module}")
def api_models(module: str):
    if module not in {"module1", "module2", "module3"}:
        raise HTTPException(400, "module must be module1|module2|module3")
    return list_models(module)


@app.get("/api/models/{module}/{model_key}")
def api_model_one(module: str, model_key: str):
    if module not in {"module1", "module2", "module3"}:
        raise HTTPException(400, "module must be module1|module2|module3")
    entry = get_model_entry(module, model_key)
    if not entry.get("exists"):
        raise HTTPException(404, f"artifact not found for {module}/{model_key}: {entry.get('path')}")
    return entry


class SampleAnalyzeRequest(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    junction_id: str = "junction_01"
    max_frames: int = Field(400, ge=30, le=2000)
    stride: int = Field(1, ge=1, le=10)
    module_mode: str = "all"
    model_module1: str | None = None
    model_module2: str | None = None
    model_module3: str | None = None
    allow_rules_fallback: bool = False
    persist_outputs: bool = False


class AckRequest(BaseModel):
    alert_row_id: int


def resolve_sample_path(filename: str) -> Path:
    normalized = filename.replace("\\", "/")
    name = Path(normalized).name
    if name != normalized or "/" in normalized or ".." in normalized:
        raise HTTPException(400, "Invalid sample filename")
    if not _SAFE_SAMPLE_NAME.match(name):
        raise HTTPException(400, "Invalid sample filename characters")
    if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
        raise HTTPException(400, "Sample must be a video file")

    samples_root = SAMPLES.resolve()
    path = (SAMPLES / name).resolve()
    try:
        path.relative_to(samples_root)
    except ValueError as exc:
        raise HTTPException(400, "Sample path escapes sample_videos") from exc
    if not path.is_file() or path.stat().st_size <= 10_000:
        raise HTTPException(404, f"Sample not found: {name}")
    return path


def run_junction_analysis(
    video_path: str,
    display_name: str,
    analysis_mode: str,
    junction_id: str = "junction_01",
    max_frames: int = 400,
    stride: int = 1,
    module_mode: str = "all",
    model_module1: str | None = None,
    model_module2: str | None = None,
    model_module3: str | None = None,
    allow_rules_fallback: bool = False,
    persist_outputs: bool = False,
) -> dict:
    try:
        junction = load_junction(junction_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    out_name = f"annot_{Path(display_name).stem}.mp4"
    # Demo default: write annotated preview to temp only (no permanent outputs folder clutter)
    out_dir = OUTPUTS if persist_outputs else DEMO_TMP
    out_path = str(out_dir / out_name)
    media_prefix = "/outputs" if persist_outputs else "/demo-outputs"

    try:
        perception = analyze_and_annotate(
            video_path,
            out_path,
            max_frames=max_frames,
            stride=stride,
            junction_id=junction_id,
        )
    except Exception as exc:
        logger.exception("Perception failed for %s", display_name)
        raise HTTPException(500, f"Perception failed: {exc}") from exc

    arena1 = load_clean_arena("module1")
    arena2 = load_clean_arena("module2")
    arena3 = load_clean_arena("module3")

    try:
        decided = run_decisions(
            perception,
            junction=junction,
            arena1=arena1,
            arena2=arena2,
            arena3=arena3,
            module_mode=module_mode,
            model_module1=model_module1,
            model_module2=model_module2,
            model_module3=model_module3,
            allow_rules_fallback=allow_rules_fallback,
        )
    except ModelInferenceError as exc:
        raise HTTPException(
            409,
            detail={
                "error": "model_inference_unavailable",
                "message": str(exc),
                "hint": (
                    "Copy clean winners from Google Drive into local models/ "
                    "(module1/dl/TCN.pt, module2/baselines/LightGBM.joblib, "
                    "module3/baselines/LightGBM.joblib) or pass allow_rules_fallback=true."
                ),
                "allow_rules_fallback": allow_rules_fallback,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    m1 = decided["module1_bus"]
    m2 = decided["module2_breakdown"]
    m3 = decided["module3_uturn"]
    actions = decided["actions"]
    active = decided["active_modules"]
    alerts = decided["alerts"]

    web_name = Path(perception["annotated_video"]).name

    if m2.get("status") == "INCIDENT DETECTED":
        signal_phase = "HOLD — INCIDENT ON APPROACH"
        signal_mode = "incident_override"
    elif m1.get("priority_status") == "REQUESTED":
        approach = m1.get("approach", "NS")
        signal_phase = f"PRIORITY GREEN · {approach}"
        signal_mode = "bus_priority"
    elif m3.get("active") and str(m3.get("recommended_action", "")).startswith("SCHEDULE"):
        signal_phase = "PROTECTED U-TURN WINDOW"
        signal_mode = "uturn_adaptive"
    else:
        signal_phase = "PHASE_1 · NS THROUGH"
        signal_mode = "normal"

    result = {
        "filename": display_name,
        "n_tracks": perception["n_tracks"],
        "frames_processed": perception["frames_processed"],
        "fps": perception.get("fps"),
        "vehicle_counts": perception.get("vehicle_counts", {}),
        "approach_density": perception.get("approach_density", {}),
        "camera_mode": perception.get("camera_mode", "stationary_cctv"),
        "decision_engine": decided["decision_engine"],
        "perception_engine": PERCEPTION_ENGINE,
        "analysis_mode": analysis_mode,
        "max_frames": max_frames,
        "stride": stride,
        "annotated_video_url": f"{media_prefix}/{web_name}",
        "active_modules": active,
        "alerts": alerts,
        "alert_count": len(alerts),
        "signal_phase": signal_phase,
        "signal_mode": signal_mode,
        "signal_phase_simulated": True,
        "recommended_action": " + ".join(actions),
        "module1_bus": m1,
        "module2_breakdown": m2,
        "module3_uturn": m3,
        "results": decided["results"],
        "safety_shield": decided["safety_shield"],
        "fallback_warning": decided["fallback_warning"],
        "persist_outputs": persist_outputs,
        "processing_metadata": {
            "perception_engine": PERCEPTION_ENGINE,
            "decision_engine": decided["decision_engine"],
            "allow_rules_fallback": allow_rules_fallback,
            "persist_outputs": persist_outputs,
            "junction_id": junction.get("id"),
            "frames_processed": perception["frames_processed"],
            "module_mode": module_mode,
            "selected_models": {
                "module1": m1.get("selected_model"),
                "module2": m2.get("selected_model"),
                "module3": m3.get("selected_model"),
            },
            "inference_errors": decided.get("inference_errors") or [],
            "schema_versions": {
                "module1": "1.0.0",
                "module2": "1.0.0",
                "module3": "1.0.0",
            },
        },

        "junction": {
            "id": junction.get("id"),
            "name": junction.get("name"),
            "lat": junction.get("lat"),
            "lon": junction.get("lon"),
            "camera_id": junction.get("camera_id"),
            "meters_per_pixel": junction.get("meters_per_pixel"),
            "meters_per_pixel_note": junction.get("meters_per_pixel_note"),
            "source": "config/junctions",
        },
        "ops_summary": {
            "junction": junction.get("id"),
            "junction_source": "config/junctions",
            "camera": junction.get("camera_id"),
            "camera_source": "config/junctions",
            "tracks": perception["n_tracks"],
            "buses": perception.get("vehicle_counts", {}).get("bus", 0),
            "incidents": len(alerts),
            "density_total": perception.get("approach_density", {}).get("total"),
            "action": " + ".join(actions),
        },
        "tracks_preview": perception["tracks"][:20],
    }

    if persist_outputs:
        try:
            analysis_id = save_analysis(result, junction_id=junction.get("id"))
            result["analysis_id"] = analysis_id
        except Exception:
            logger.exception("Failed to persist analysis")
            result["analysis_id"] = None
    else:
        result["analysis_id"] = None

    return result


@app.post("/api/video/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    junction_id: str = Query("junction_01"),
    max_frames: int = Query(400, ge=30, le=2000),
    stride: int = Query(1, ge=1, le=10),
    module_mode: str = Query("all"),
    model_module1: str | None = Query(None),
    model_module2: str | None = Query(None),
    model_module3: str | None = Query(None),
    allow_rules_fallback: bool = Query(False),
    persist_outputs: bool = Query(False),
):
    if not file.filename or Path(file.filename).suffix.lower() not in VIDEO_SUFFIXES:
        raise HTTPException(400, "Upload a video file (.mp4/.avi/.mov/.mkv/.webm)")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        return await asyncio.to_thread(
            run_junction_analysis,
            tmp_path,
            file.filename,
            "batch_upload",
            junction_id,
            max_frames,
            stride,
            module_mode,
            model_module1,
            model_module2,
            model_module3,
            allow_rules_fallback,
            persist_outputs,
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/api/video/analyze-sample")
async def analyze_sample(body: SampleAnalyzeRequest):
    path = resolve_sample_path(body.filename)
    return await asyncio.to_thread(
        run_junction_analysis,
        str(path),
        path.name,
        "sample_on_disk",
        body.junction_id,
        body.max_frames,
        body.stride,
        body.module_mode,
        body.model_module1,
        body.model_module2,
        body.model_module3,
        body.allow_rules_fallback,
        body.persist_outputs,
    )


@app.get("/api/junctions")
def api_junctions():
    return {"junctions": list_junctions()}


@app.get("/api/junctions/{junction_id}")
def api_junction(junction_id: str):
    try:
        return load_junction(junction_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/geo/summary")
def api_geo_summary():
    return geo_service.geo_summary()


@app.get("/api/geo/stops")
def api_geo_stops(
    limit: int = Query(400, ge=1, le=5000),
    min_lat: float | None = None,
    min_lon: float | None = None,
    max_lat: float | None = None,
    max_lon: float | None = None,
):
    bbox = None
    if None not in (min_lat, min_lon, max_lat, max_lon):
        bbox = (min_lat, min_lon, max_lat, max_lon)
    return {"stops": geo_service.list_stops(limit=limit, bbox=bbox)}


@app.get("/api/geo/routes/{route_code}")
def api_geo_route(route_code: str):
    return geo_service.route_node_map(route_code)


@app.get("/api/geo/hotspots")
def api_geo_hotspots(limit_cells: int = Query(40, ge=5, le=200)):
    return geo_service.gps_hotspots(limit_cells=limit_cells)


@app.get("/api/history/analyses")
def api_history_analyses(limit: int = Query(20, ge=1, le=100)):
    return {"analyses": list_analyses(limit=limit)}


@app.get("/api/history/analyses/{analysis_id}")
def api_history_one(analysis_id: int):
    row = get_analysis(analysis_id)
    if not row:
        raise HTTPException(404, "analysis not found")
    return row


@app.get("/api/history/alerts")
def api_history_alerts(
    status: str | None = Query("ACTIVE"),
    limit: int = Query(50, ge=1, le=200),
):
    if status == "all":
        status = None
    return {"alerts": list_alerts(status=status, limit=limit)}


@app.post("/api/history/alerts/ack")
def api_ack_alert(body: AckRequest):
    ok = acknowledge_alert(body.alert_row_id)
    if not ok:
        raise HTTPException(404, "active alert not found")
    return {"ok": True, "alert_row_id": body.alert_row_id}
