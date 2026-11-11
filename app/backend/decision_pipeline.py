"""Compose perception tracks + selected models into module decisions."""

from __future__ import annotations

from typing import Any

from junction_config import estimate_conflict_gap_sec
from module_inference import ModelInferenceError, score_tracks_for_module
from model_registry import winner_config
from perception import build_module_outputs
from safety_shield import apply_safety_shield

DECISION_ENGINE = "model_inference_v1"
PERCEPTION_ENGINE = "yolov8n_bytetrack"

M1_THRESHOLD = 0.55
M2_THRESHOLD = 0.55
M3_THRESHOLD = 0.55


def _active_modules(module_mode: str) -> set[str]:
    mode = (module_mode or "all").lower()
    if mode in {"all", "auto"}:
        return {"module1", "module2", "module3"}
    if mode in {"module1", "module2", "module3"}:
        return {mode}
    raise ValueError("module_mode must be auto|module1|module2|module3|all")


def _rules_bundle(perception, arena1, arena2, arena3, junction):
    m1, m2, m3, actions, active, alerts = build_module_outputs(
        perception, arena1, arena2, arena3, junction=junction
    )
    for mod, key in ((m1, "module1"), (m2, "module2"), (m3, "module3")):
        mod["decision_source"] = "rules_fallback"
        mod["fallback_warning"] = True
        mod["arena_model_drives_decision"] = False
        mod["decision_engine"] = "rules_v1"
        mod.setdefault("model_score", mod.get("incident_probability") or 0.0)
        mod.setdefault("safety_status", "OK")
        if key == "module1":
            mod["decision"] = (
                "REQUEST_PRIORITY"
                if mod.get("priority_status") == "REQUESTED"
                else "NONE"
            )
        elif key == "module2":
            mod["decision"] = (
                "SUSPECTED_ABNORMAL_STOP"
                if mod.get("status") == "INCIDENT DETECTED"
                else "NORMAL_OR_EXPECTED_STOP"
            )
        else:
            mod["decision"] = (
                "UTURN_DEMAND_SIGNAL"
                if str(mod.get("recommended_action", "")).startswith("SCHEDULE")
                else ("UTURN" if mod.get("active") else "NOT_UTURN")
            )
    return m1, m2, m3, actions, active, alerts


def _infer_module1(perception, model_key: str) -> dict:
    buses = [t for t in perception.get("tracks", []) if t.get("is_bus")]
    scored = score_tracks_for_module("module1", model_key, perception, buses)
    track = scored.pop("track", None) or (buses[0] if buses else None)
    grant = bool(buses) and scored["model_score"] >= M1_THRESHOLD
    # ETA display from features if present
    feats = scored.get("features_used") or {}
    module1 = {
        "active": bool(buses),
        "reason": (
            "Model inference on bus track features"
            if buses
            else "No bus detected in this clip"
        ),
        "decision_engine": DECISION_ENGINE,
        "selected_model": scored["selected_model"],
        "model_version": scored["model_version"],
        "schema_id": scored["schema_id"],
        "model_score": scored["model_score"],
        "decision": "REQUEST_PRIORITY" if grant else "NONE",
        "decision_source": "model_inference",
        "features_used": scored["features_used"],
        "safety_status": "OK",
        "fallback_warning": False,
        "arena_model_drives_decision": True,
        "track_id": scored.get("track_id"),
        "bus_id": f"BUS-{track['id']}" if track else None,
        "approach": track.get("approach") if track else None,
        "distance_m": round(float(track.get("distance_proxy_m") or 0), 1) if track else None,
        "eta_sec": round(float(feats.get("eta_proxy_sec") or 0), 1) if track else None,
        "priority_status": "REQUESTED" if grant else ("MONITORING" if buses else "INACTIVE"),
        "recommended_action": (
            "EXTEND GREEN (+12s)" if grant else (
                "HOLD — bus not yet in priority window" if buses else "No bus priority request"
            )
        ),
        "confidence": round(float(track.get("conf") or scored["model_score"]), 3) if track else scored["model_score"],
        "signal_request": {
            "from_phase": "current",
            "to_phase": f"{track.get('approach')}-THROUGH GREEN" if track else None,
            "duration_sec": 12 if grant else 0,
            "accepted": bool(grant),
        } if buses else None,
    }
    return module1


def _infer_module2(perception, model_key: str) -> tuple[dict, list]:
    vehicles = [t for t in perception.get("tracks", []) if t.get("is_vehicle")]
    # Prefer stopped-after-moving candidates; else score all vehicles
    preferred = [
        t for t in vehicles
        if t.get("was_moving") and (t.get("currently_stopped") or t.get("sudden_stop_candidate"))
    ]
    candidates = preferred or vehicles
    scored = score_tracks_for_module("module2", model_key, perception, candidates)
    track = scored.pop("track", None)
    positive = scored["model_score"] >= M2_THRESHOLD and track is not None
    # Extra safety: require some stop evidence for alert
    if positive and track is not None:
        if not (track.get("currently_stopped") or track.get("sudden_stop_candidate")):
            if scored["model_score"] < 0.8:
                positive = False

    alerts = []
    if positive and track is not None:
        dur = float(track.get("stationary_duration_sec") or 0.0)
        severity = "HIGH" if scored["model_score"] >= 0.78 or dur >= 6 else "MEDIUM"
        alert = {
            "alert_id": f"INC-BRK-{track['id']}",
            "type": "UNEXPECTED_STOPPAGE",
            "severity": severity,
            "vehicle_id": f"VEH-{track['id']}",
            "vehicle_class": track.get("cls_name", "vehicle"),
            "location": f"{track.get('approach')} Approach",
            "stoppage_sec": round(dur, 1),
            "message": (
                f"{str(track.get('cls_name', 'Vehicle')).title()} VEH-{track['id']} "
                f"scored abnormal stop (model={scored['selected_model']}, "
                f"p={scored['model_score']:.3f}) on {track.get('approach')} approach."
            ),
            "channels": ["Control Room", "Patrol Dispatch", "V2I Slow-Down Broadcast"],
            "status": "ACTIVE",
            "recommended_response": (
                f"Dispatch patrol/tow to {track.get('approach')} approach. "
                "Broadcast slow-down advisory."
            ),
            "module": "module2",
        }
        alerts.append(alert)
        module2 = {
            "active": True,
            "reason": "Trained model inferred suspected abnormal stop",
            "decision_engine": DECISION_ENGINE,
            "selected_model": scored["selected_model"],
            "model_version": scored["model_version"],
            "schema_id": scored["schema_id"],
            "model_score": scored["model_score"],
            "decision": "SUSPECTED_ABNORMAL_STOP",
            "decision_source": "model_inference",
            "features_used": scored["features_used"],
            "safety_status": "REVIEW_REQUIRED",
            "fallback_warning": False,
            "arena_model_drives_decision": True,
            "track_id": scored.get("track_id"),
            "status": "INCIDENT DETECTED",
            "vehicle_id": f"VEH-{track['id']}",
            "vehicle_class": track.get("cls_name", "vehicle"),
            "location": f"{track.get('approach')} Approach",
            "stationary_duration_sec": round(dur, 1),
            "prior_max_speed": round(float(track.get("max_speed") or 0), 1),
            "incident_probability": scored["model_score"],
            "incident_probability_source": "model_inference",
            "action": "Traffic Incident Alert",
            "control_room": alert["recommended_response"],
            "alert": alert,
        }
    else:
        module2 = {
            "active": False,
            "reason": "Model did not confirm abnormal stop",
            "decision_engine": DECISION_ENGINE,
            "selected_model": scored["selected_model"],
            "model_version": scored["model_version"],
            "schema_id": scored["schema_id"],
            "model_score": scored["model_score"],
            "decision": "NORMAL_OR_EXPECTED_STOP",
            "decision_source": "model_inference",
            "features_used": scored["features_used"],
            "safety_status": "OK",
            "fallback_warning": False,
            "arena_model_drives_decision": True,
            "track_id": scored.get("track_id"),
            "status": "NORMAL",
            "action": "Monitoring — no alert",
            "incident_probability": scored["model_score"],
            "incident_probability_source": "model_inference",
            "alert": None,
        }
    return module2, alerts


def _infer_module3(perception, model_key: str, junction) -> dict:
    vehicles = [t for t in perception.get("tracks", []) if t.get("is_vehicle")]
    scored = score_tracks_for_module("module3", model_key, perception, vehicles)
    track = scored.pop("track", None)
    positive = scored["model_score"] >= M3_THRESHOLD and track is not None
    gap, gap_source = (0.0, "n/a")
    if track is not None:
        gap, gap_source = estimate_conflict_gap_sec(track, vehicles, junction or {})
    high = positive and scored["model_score"] >= 0.7
    schedule = positive and gap >= 3.0
    module3 = {
        "active": bool(positive),
        "reason": (
            "Trained model inferred U-turn / protected-turn demand"
            if positive
            else "Model did not confirm U-turn demand"
        ),
        "decision_engine": DECISION_ENGINE,
        "selected_model": scored["selected_model"],
        "model_version": scored["model_version"],
        "schema_id": scored["schema_id"],
        "model_score": scored["model_score"],
        "decision": (
            "UTURN_DEMAND_SIGNAL" if schedule else ("UTURN" if positive else "NOT_UTURN")
        ),
        "decision_source": "model_inference",
        "features_used": scored["features_used"],
        "safety_status": "OK" if not schedule else "OK",
        "fallback_warning": False,
        "arena_model_drives_decision": True,
        "track_id": scored.get("track_id"),
        "demand_level": "High" if high else ("Medium" if positive else "Low"),
        "uturn_count": 1 if positive else 0,
        "queue_estimate": 1 if positive else 0,
        "gap_sec": gap,
        "gap_sec_source": gap_source,
        "in_uturn_pocket": bool(track.get("in_uturn_pocket")) if track else False,
        "recommended_action": (
            "SCHEDULE PROTECTED U-TURN (+8s)" if schedule else (
                "HOLD — gap / demand not ready" if positive else "Maintain standard phase plan"
            )
        ),
        "turn_intensity": round(float((track or {}).get("uturn_score") or scored["model_score"]), 3),
        "net_heading_rad": round(float((track or {}).get("net_heading") or 0), 3),
    }
    return module3


def run_decisions(
    perception: dict,
    *,
    junction: dict | None,
    arena1,
    arena2,
    arena3,
    module_mode: str = "all",
    model_module1: str | None = None,
    model_module2: str | None = None,
    model_module3: str | None = None,
    allow_rules_fallback: bool = False,
) -> dict[str, Any]:
    wanted = _active_modules(module_mode)
    errors: list[str] = []

    m1 = m2 = m3 = None
    alerts: list = []

    try:
        if "module1" in wanted:
            m1 = _infer_module1(
                perception, model_module1 or winner_config("module1")["model_key"]
            )
    except ModelInferenceError as exc:
        errors.append(f"module1: {exc}")

    try:
        if "module2" in wanted:
            m2, alerts = _infer_module2(
                perception, model_module2 or winner_config("module2")["model_key"]
            )
    except ModelInferenceError as exc:
        errors.append(f"module2: {exc}")

    try:
        if "module3" in wanted:
            m3 = _infer_module3(
                perception,
                model_module3 or winner_config("module3")["model_key"],
                junction,
            )
    except ModelInferenceError as exc:
        errors.append(f"module3: {exc}")

    if errors:
        if not allow_rules_fallback:
            raise ModelInferenceError(" | ".join(errors))
        rm1, rm2, rm3, actions, active, alerts = _rules_bundle(
            perception, arena1, arena2, arena3, junction
        )
        # Fill only failed modules from rules
        if m1 is None:
            m1 = rm1
        if m2 is None:
            m2 = rm2
        if m3 is None:
            m3 = rm3
        fallback_warning = True
    else:
        fallback_warning = False
        # inactive modules placeholders when mode subset
        if m1 is None:
            m1 = {
                "active": False,
                "decision": "NONE",
                "decision_source": "skipped",
                "selected_model": model_module1,
                "model_score": 0.0,
                "fallback_warning": False,
                "safety_status": "OK",
                "priority_status": "INACTIVE",
                "recommended_action": "Module not selected",
            }
        if m2 is None:
            m2 = {
                "active": False,
                "decision": "NORMAL_OR_EXPECTED_STOP",
                "decision_source": "skipped",
                "selected_model": model_module2,
                "model_score": 0.0,
                "fallback_warning": False,
                "safety_status": "OK",
                "status": "NORMAL",
                "recommended_action": "Module not selected",
                "alert": None,
            }
        if m3 is None:
            m3 = {
                "active": False,
                "decision": "NOT_UTURN",
                "decision_source": "skipped",
                "selected_model": model_module3,
                "model_score": 0.0,
                "fallback_warning": False,
                "safety_status": "OK",
                "demand_level": "Low",
                "recommended_action": "Module not selected",
            }

    shielded, _ = apply_safety_shield(m1, m2, m3)
    m1, m2, m3 = shielded["module1"], shielded["module2"], shielded["module3"]
    safety_shield = shielded["safety_shield"]

    actions = []
    active = []
    if m2.get("status") == "INCIDENT DETECTED" or m2.get("decision") == "SUSPECTED_ABNORMAL_STOP":
        active.append("module2_breakdown")
        actions.append("BROADCAST BREAKDOWN ALERT")
        if m1.get("priority_status") in {"REQUESTED", "DEFERRED"}:
            actions.append("DEFER BUS PRIORITY (incident override)")
        else:
            actions.append("HOLD SIGNAL — clear incident approach")
    else:
        if m1.get("priority_status") == "REQUESTED":
            active.append("module1_bus")
            actions.append(f"EXTEND GREEN for {m1.get('bus_id') or 'BUS'}")
        if m3.get("active") and str(m3.get("recommended_action", "")).startswith("SCHEDULE"):
            active.append("module3_uturn")
            actions.append("SCHEDULE PROTECTED U-TURN")
        elif m1.get("active"):
            active.append("module1_bus")
        elif m3.get("active"):
            active.append("module3_uturn")

    if not actions:
        actions.append("MAINTAIN CURRENT SIGNAL PLAN")

    if fallback_warning:
        for mod in (m1, m2, m3):
            mod["fallback_warning"] = True

    results = {
        "module1": m1,
        "module2": m2,
        "module3": m3,
    }

    return {
        "module1_bus": m1,
        "module2_breakdown": m2,
        "module3_uturn": m3,
        "results": results,
        "actions": actions,
        "active_modules": active,
        "alerts": alerts,
        "safety_shield": safety_shield,
        "decision_engine": DECISION_ENGINE if not fallback_warning else "rules_fallback",
        "fallback_warning": fallback_warning,
        "inference_errors": errors,
        "allow_rules_fallback": allow_rules_fallback,
    }
