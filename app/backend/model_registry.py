"""Demo-mode model registry: only viva-allowed architectures."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger("traffic_ai.registry")

ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = ROOT / "models"
METRICS_ROOT = ROOT / "metrics"

MODULE_SCHEMA = {
    "module1": "module1_bus_priority_v1",
    "module2": "module2_breakdown_v1",
    "module3": "module3_uturn_v1",
}

# Exact demo-allowed architecture set (UI + registry)
ALLOWED_MODEL_KEYS = [
    "GAT_GRU",
    "Transformer_Encoder_ClassWeighted",
    "Transformer_Feature_Attention",
    "TCN_Attention",
    "BiLSTM_Attention",
    "Autoencoder_IsolationForest",
]

# Deterministic priority when clean val metrics are not used for selection
PRIORITY_ORDER = list(ALLOWED_MODEL_KEYS)

ARTIFACT_RELPATH = {
    "GAT_GRU": "dl/GAT_GRU.pt",
    "Transformer_Encoder_ClassWeighted": "dl/Transformer_Encoder_ClassWeighted.pt",
    "Transformer_Feature_Attention": "dl/Transformer_Feature_Attention.pt",
    "TCN_Attention": "dl/TCN_Attention.pt",
    "BiLSTM_Attention": "dl/BiLSTM_Attention.pt",
    "Autoencoder_IsolationForest": "dl/Autoencoder_IsolationForest.joblib",
}


def _module_dir(module: str) -> Path:
    return MODELS_ROOT / module


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact_path(module: str, relpath: str) -> Path:
    return (_module_dir(module) / relpath).resolve()


def _photo_f1_map() -> dict[str, float]:
    photo = _read_json(METRICS_ROOT / "lab_notebook_photo_metrics.json") or {}
    out = {}
    for row in photo.get("models") or []:
        if row.get("model") and row.get("f1") is not None:
            out[str(row["model"])] = float(row["f1"])
    return out


def pick_default_winner(module: str) -> dict:
    """Best REAL available allowed model: photo F1 among existing files, else priority order."""
    photo_f1 = _photo_f1_map()
    existing = []
    for key in ALLOWED_MODEL_KEYS:
        rel = ARTIFACT_RELPATH[key]
        path = resolve_artifact_path(module, rel)
        if path.is_file():
            existing.append((key, rel, photo_f1.get(key)))
    if not existing:
        # fall back to first priority even if missing (UI shows missing)
        key = PRIORITY_ORDER[0]
        return {
            "module": module,
            "model_key": key,
            "artifact_relpath": ARTIFACT_RELPATH[key],
            "selection_rule": "demo_allowed_priority",
            "selected_on": "none_available",
            "schema_id": MODULE_SCHEMA[module],
        }

    scored = [e for e in existing if e[2] is not None]
    if scored:
        best = max(scored, key=lambda e: e[2])
    else:
        # priority order among existing
        order = {k: i for i, k in enumerate(PRIORITY_ORDER)}
        best = sorted(existing, key=lambda e: order.get(e[0], 999))[0]
    return {
        "module": module,
        "model_key": best[0],
        "artifact_relpath": best[1],
        "selection_rule": "demo_photo_f1_then_priority",
        "selected_on": "available_allowed_artifacts",
        "schema_id": MODULE_SCHEMA[module],
        "photo_f1": best[2],
    }


def winner_config(module: str) -> dict:
    path = _module_dir(module) / "deployment" / "winner.json"
    data = _read_json(path)
    if isinstance(data, dict) and data.get("model_key") in ALLOWED_MODEL_KEYS:
        rel = data.get("artifact_relpath") or ARTIFACT_RELPATH[data["model_key"]]
        if resolve_artifact_path(module, rel).is_file():
            return {
                **data,
                "artifact_relpath": rel,
                "schema_id": MODULE_SCHEMA[module],
            }
    return pick_default_winner(module)


def list_models(module: str) -> dict:
    if module not in MODULE_SCHEMA:
        raise KeyError(module)
    winner = winner_config(module)
    models = []
    for key in ALLOWED_MODEL_KEYS:
        rel = ARTIFACT_RELPATH[key]
        path = resolve_artifact_path(module, rel)
        models.append({
            "model_key": key,
            "kind": "dl",
            "path": str(path),
            "relpath": rel,
            "meta": {},
            "exists": path.is_file(),
            "allowed": True,
        })
    # winner first
    models.sort(key=lambda m: (0 if m["model_key"] == winner["model_key"] else 1, ALLOWED_MODEL_KEYS.index(m["model_key"])))
    missing = [m["model_key"] for m in models if not m["exists"]]
    return {
        "module": module,
        "schema_id": MODULE_SCHEMA[module],
        "winner": winner,
        "models": models,
        "allowed_model_keys": ALLOWED_MODEL_KEYS,
        "missing_artifacts": missing,
        "demo_mode": True,
    }


def get_model_entry(module: str, model_key: str | None) -> dict:
    info = list_models(module)
    winner = info["winner"]
    key = (model_key or winner["model_key"]).strip()
    if key not in ALLOWED_MODEL_KEYS:
        raise KeyError(f"{key} not in demo-allowed set: {ALLOWED_MODEL_KEYS}")
    for row in info["models"]:
        if row["model_key"].lower() == key.lower():
            return {**row, "winner_config": winner, "schema_id": MODULE_SCHEMA[module]}
    rel = ARTIFACT_RELPATH[key]
    ap = resolve_artifact_path(module, rel)
    return {
        "model_key": key,
        "kind": "dl",
        "path": str(ap),
        "relpath": rel,
        "meta": {},
        "exists": ap.is_file(),
        "winner_config": winner,
        "schema_id": MODULE_SCHEMA[module],
    }


def load_joblib_artifact(path: Path) -> dict[str, Any]:
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return {
            "estimator": obj["model"],
            "feature_names": list(obj.get("feature_names") or obj.get("features") or obj.get("feat_cols") or []),
            "label_name": obj.get("label_name"),
            "raw": obj,
        }
    return {"estimator": obj, "feature_names": [], "label_name": None, "raw": obj}


def load_torch_artifact(path: Path, meta: dict | None = None) -> dict[str, Any]:
    import torch

    meta = meta or {}
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return {"blob": blob, "meta": meta, "path": path}


@lru_cache(maxsize=64)
def load_cached(module: str, model_key: str) -> dict[str, Any]:
    entry = get_model_entry(module, model_key)
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing artifact for {module}/{model_key}: {path}. "
            f"Allowed set requires one of: {ALLOWED_MODEL_KEYS}"
        )
    if path.suffix.lower() == ".joblib":
        payload = load_joblib_artifact(path)
        return {
            "module": module,
            "model_key": entry["model_key"],
            "kind": "ml",
            "path": str(path),
            "schema_id": entry["schema_id"],
            **payload,
        }
    if path.suffix.lower() == ".pt":
        payload = load_torch_artifact(path, entry.get("meta") or {})
        blob = payload.get("blob")
        feature_names = []
        mean = std = None
        seq_len = None
        if isinstance(blob, dict):
            feature_names = list(
                blob.get("feat_cols")
                or blob.get("seq_feats")
                or blob.get("features")
                or blob.get("feature_names")
                or []
            )
            mean = blob.get("mean")
            std = blob.get("std")
            seq_len = blob.get("seq_len")
        # Side-car norm stats written by Colab export
        dl_dir = path.parent
        mean_p, std_p = dl_dir / "seq_mean.npy", dl_dir / "seq_std.npy"
        if mean is None and mean_p.is_file() and std_p.is_file():
            import numpy as np

            mean = np.load(mean_p)
            std = np.load(std_p)
        return {
            "module": module,
            "model_key": entry["model_key"],
            "kind": "dl",
            "path": str(path),
            "schema_id": entry["schema_id"],
            "feature_names": feature_names,
            "mean": mean,
            "std": std,
            "seq_len": seq_len,
            **payload,
        }
    raise ValueError(f"Unsupported artifact type: {path}")


def clear_cache() -> None:
    load_cached.cache_clear()


def health_flags() -> dict:
    out = {
        "rules_fallback_default": False,
        "models_root": str(MODELS_ROOT),
        "demo_mode": True,
        "allowed_model_keys": ALLOWED_MODEL_KEYS,
    }
    for module in MODULE_SCHEMA:
        entry = get_model_entry(module, None)
        path = Path(entry["path"])
        loaded = False
        if path.is_file():
            try:
                load_cached(module, entry["model_key"])
                loaded = True
            except Exception as exc:
                logger.warning("Could not load %s winner: %s", module, exc)
        out[f"{module}_winner"] = entry["model_key"]
        out[f"{module}_artifact"] = str(path)
        out[f"{module}_model_loaded"] = loaded
        out[f"{module}_model_in_decision_path"] = loaded
        out[f"{module}_missing"] = list_models(module).get("missing_artifacts") or []
    return out


def load_lab_photo_metrics() -> dict:
    data = _read_json(METRICS_ROOT / "lab_notebook_photo_metrics.json")
    if not data:
        return {"models": [], "source": None}
    return data


def load_clean_arena(module: str) -> list:
    """Lab uses notebook photo metrics in demo mode (not leaky arena JSON)."""
    photo = load_lab_photo_metrics()
    return list(photo.get("models") or [])


def load_clean_arena_bundle(module: str) -> dict:
    photo = load_lab_photo_metrics()
    winner = winner_config(module)
    models = list(photo.get("models") or [])
    return {
        "module": module,
        "source_path": str(METRICS_ROOT / "lab_notebook_photo_metrics.json"),
        "clean": False,
        "photo_lab": True,
        "label_meaning": photo.get("note"),
        "selection_rule": winner.get("selection_rule"),
        "deployment_winner": winner.get("model_key"),
        "deployment_winner_val_f1": None,
        "deployment_winner_test_f1": None,
        "n_models": len(models),
        "models": models,
        "missing_artifacts": list_models(module).get("missing_artifacts") or [],
    }
