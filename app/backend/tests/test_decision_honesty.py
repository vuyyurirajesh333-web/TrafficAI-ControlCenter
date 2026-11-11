"""P1.1 — decision engine honesty and rule behaviour (no YOLO required)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception import (
    DECISION_ENGINE,
    PERCEPTION_ENGINE,
    build_module_outputs,
    pick_best_model_name,
)


ARENA = [
    {"model": "Random_Forest", "f1": 0.99},
    {"model": "XGBoost", "f1": 0.95},
]


def _track(**kwargs):
    base = {
        "id": 1,
        "is_bus": False,
        "is_vehicle": True,
        "cls_name": "car",
        "conf": 0.9,
        "approach": "NS",
        "distance_proxy_m": 200.0,
        "last_speed": 40.0,
        "mean_recent_speed": 40.0,
        "sudden_stop_candidate": False,
        "currently_stopped": False,
        "was_moving": True,
        "stationary_ratio": 0.0,
        "stationary_duration_sec": 0.0,
        "max_speed": 50.0,
        "uturn_score": 0.0,
        "net_heading": 0.0,
        "frames": 30,
        "moving_frames": 20,
        "turn_amount": 0.0,
    }
    base.update(kwargs)
    return base


def test_pick_best_model_name_by_f1():
    name, meta = pick_best_model_name(ARENA)
    assert name == "Random_Forest"
    assert meta["f1"] == 0.99


def test_module_payloads_declare_rules_engine_not_arena():
    perception = {
        "tracks": [
            _track(id=7, is_bus=True, cls_name="bus", distance_proxy_m=50.0, last_speed=30.0),
        ]
    }
    m1, m2, m3, actions, active, alerts = build_module_outputs(
        perception, ARENA, ARENA, ARENA
    )

    for mod in (m1, m2, m3):
        assert mod["decision_engine"] == DECISION_ENGINE
        assert mod["arena_model_drives_decision"] is False
        assert "decision_note" in mod
        # Compat field still present but must not imply live inference
        assert mod["selected_model"] == "Random_Forest"

    assert m1["active"] is True
    assert m1["priority_status"] == "REQUESTED"
    assert "module1_bus" in active
    assert any("EXTEND GREEN" in a for a in actions)


def test_incident_overrides_bus_priority():
    perception = {
        "tracks": [
            _track(
                id=1,
                is_bus=True,
                cls_name="bus",
                distance_proxy_m=40.0,
                last_speed=25.0,
            ),
            _track(
                id=2,
                sudden_stop_candidate=True,
                currently_stopped=True,
                was_moving=True,
                stationary_ratio=0.8,
                stationary_duration_sec=5.0,
                max_speed=60.0,
                mean_recent_speed=2.0,
                last_speed=1.0,
            ),
            _track(
                id=3,
                currently_stopped=False,
                was_moving=True,
                last_speed=45.0,
                mean_recent_speed=45.0,
                max_speed=50.0,
            ),
        ]
    }
    m1, m2, m3, actions, active, alerts = build_module_outputs(
        perception, ARENA, ARENA, ARENA
    )

    assert m2["status"] == "INCIDENT DETECTED"
    assert m2["incident_probability_source"] == "heuristic_formula"
    assert m1["priority_status"] == "REQUESTED"
    assert any("DEFER BUS PRIORITY" in a for a in actions)
    assert len(alerts) == 1


def test_uturn_gap_from_conflict_estimator():
    perception = {
        "tracks": [
            _track(
                id=9,
                uturn_score=0.7,
                net_heading=2.8,
                frames=40,
                approach="North",
            ),
            _track(
                id=10,
                approach="South",
                currently_stopped=False,
                last_speed=40.0,
                mean_recent_speed=40.0,
                distance_proxy_m=80.0,
            ),
        ]
    }
    _m1, _m2, m3, actions, active, _alerts = build_module_outputs(
        perception, ARENA, ARENA, ARENA
    )
    assert m3["active"] is True
    assert m3["gap_sec_source"] in {
        "conflict_track_eta",
        "conflict_tracks_empty_default",
    }
    assert m3["gap_sec"] > 0
    assert m3["recommended_action"].startswith("SCHEDULE")
    assert "module3_uturn" in active
    assert any("U-TURN" in a for a in actions)


def test_engine_constants():
    assert DECISION_ENGINE == "rules_v1"
    assert PERCEPTION_ENGINE == "yolov8n_bytetrack"
