"""Model registry + decision pipeline wiring (no YOLO required)."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_registry import get_model_entry, health_flags, list_models, winner_config
from module_inference import ModelInferenceError
from decision_pipeline import run_decisions


def test_winners_point_to_clean_artifact_names():
    assert winner_config("module1")["model_key"] == "TCN"
    assert winner_config("module1")["artifact_relpath"].endswith("TCN.pt")
    assert winner_config("module2")["model_key"] == "LightGBM"
    assert winner_config("module3")["model_key"] == "LightGBM"


def test_list_models_includes_winner():
    info = list_models("module2")
    assert info["winner"]["model_key"] == "LightGBM"
    assert any(m["model_key"].lower() == "lightgbm" for m in info["models"])


def test_health_flags_declare_rules_fallback_off():
    flags = health_flags()
    assert flags["rules_fallback_default"] is False


def test_missing_artifact_raises_without_fallback():
    perception = {"tracks": [], "approach_density": {"total": 0}}
    with pytest.raises(ModelInferenceError):
        run_decisions(
            perception,
            junction={"approaches": {}, "meters_per_pixel": 0.05},
            arena1=[],
            arena2=[],
            arena3=[],
            allow_rules_fallback=False,
            model_module2="LightGBM",
        )


def test_rules_fallback_sets_warning_when_enabled():
    # Without artifacts this should fall back only if allowed
    entry = get_model_entry("module2", "LightGBM")
    if entry.get("exists"):
        pytest.skip("clean LightGBM present — fallback path not exercised")
    perception = {
        "tracks": [
            {
                "id": 1,
                "is_bus": False,
                "is_vehicle": True,
                "cls_name": "car",
                "conf": 0.9,
                "approach": "North",
                "distance_proxy_m": 40.0,
                "last_speed": 1.0,
                "mean_recent_speed": 1.0,
                "sudden_stop_candidate": True,
                "currently_stopped": True,
                "was_moving": True,
                "stationary_ratio": 0.8,
                "stationary_duration_sec": 5.0,
                "max_speed": 40.0,
                "uturn_score": 0.0,
                "net_heading": 0.0,
                "frames": 40,
                "moving_frames": 10,
                "turn_amount": 0.0,
            }
        ],
        "approach_density": {"total": 1, "North": 1},
    }
    out = run_decisions(
        perception,
        junction={"approaches": {}, "meters_per_pixel": 0.05},
        arena1=[{"model": "x", "f1": 0.1}],
        arena2=[{"model": "x", "f1": 0.1}],
        arena3=[{"model": "x", "f1": 0.1}],
        allow_rules_fallback=True,
    )
    assert out["fallback_warning"] is True
    assert out["module2_breakdown"]["decision_source"] == "rules_fallback"
