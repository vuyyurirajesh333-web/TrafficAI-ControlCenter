"""Demo-mode registry checks."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_registry import (
    ALLOWED_MODEL_KEYS,
    get_model_entry,
    health_flags,
    list_models,
    load_lab_photo_metrics,
    winner_config,
)
from module_inference import ModelInferenceError
from decision_pipeline import run_decisions


def test_allowed_set_is_six():
    assert ALLOWED_MODEL_KEYS[0] == "GAT_GRU"
    assert "LightGBM" not in ALLOWED_MODEL_KEYS
    assert "XGBoost" not in ALLOWED_MODEL_KEYS
    assert len(ALLOWED_MODEL_KEYS) == 6


def test_winners_are_allowed_and_prefer_gat_when_present():
    w = winner_config("module2")
    assert w["model_key"] in ALLOWED_MODEL_KEYS
    entry = get_model_entry("module2", None)
    if Path(entry["path"]).is_file() and "GAT_GRU" not in list_models("module2")["missing_artifacts"]:
        assert w["model_key"] == "GAT_GRU"


def test_list_models_only_allowed():
    info = list_models("module1")
    keys = [m["model_key"] for m in info["models"]]
    assert keys == ALLOWED_MODEL_KEYS or set(keys) == set(ALLOWED_MODEL_KEYS)
    assert all(m["model_key"] in ALLOWED_MODEL_KEYS for m in info["models"])


def test_photo_lab_metrics_transcribed():
    photo = load_lab_photo_metrics()
    assert len(photo.get("models") or []) == 11
    gat = next(m for m in photo["models"] if m["model"] == "GAT_GRU")
    assert abs(gat["f1"] - 0.9641) < 1e-6


def test_health_flags_declare_rules_fallback_off():
    flags = health_flags()
    assert flags["rules_fallback_default"] is False


def test_missing_autoencoder_listed():
    missing = list_models("module2")["missing_artifacts"]
    assert "Autoencoder_IsolationForest" in missing
