"""Policy layer: validate model recommendations before operator actions."""

from __future__ import annotations


def apply_safety_shield(module1: dict, module2: dict, module3: dict) -> tuple[dict, list[str]]:
    overrides: list[str] = []
    blocked: list[str] = []
    notes: list[str] = []

    m1 = dict(module1)
    m2 = dict(module2)
    m3 = dict(module3)

    # Low confidence → review
    for mod, name in ((m1, "module1"), (m2, "module2"), (m3, "module3")):
        score = float(mod.get("model_score") or 0.0)
        if mod.get("decision_source") == "model_inference" and 0.0 < score < 0.45 and mod.get("active"):
            mod["safety_status"] = "REVIEW_REQUIRED"
            notes.append(f"{name}: low model_score={score:.3f}")

    incident = m2.get("status") == "INCIDENT DETECTED" or m2.get("decision") == "SUSPECTED_ABNORMAL_STOP"
    if incident and m1.get("priority_status") == "REQUESTED":
        m1["priority_status"] = "DEFERRED"
        m1["recommended_action"] = "DEFER BUS PRIORITY (incident override)"
        m1["safety_status"] = "DEFERRED"
        overrides.append("defer_bus_priority_for_incident")

    if m3.get("active") and str(m3.get("recommended_action", "")).startswith("SCHEDULE"):
        gap = float(m3.get("gap_sec") or 0.0)
        if gap < 3.0:
            m3["recommended_action"] = "HOLD — gap / demand not ready"
            m3["safety_status"] = "CAUTION"
            blocked.append("uturn_protected_window_gap_lt_3s")

    # Defaults
    for mod in (m1, m2, m3):
        mod.setdefault("safety_status", "OK")

    shield = {
        "overrides": overrides,
        "blocked_actions": blocked,
        "notes": notes,
    }
    return {"module1": m1, "module2": m2, "module3": m3, "safety_shield": shield}, []
