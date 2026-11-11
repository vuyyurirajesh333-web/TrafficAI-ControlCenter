"""Discover and load clean module artifacts for live inference."""

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

DEFAULT_WINNERS = {
    "module1": {"model_key": "TCN", "artifact_relpath": "dl/TCN.pt"},
    "module2": {"model_key": "LightGBM", "artifact_relpath": "baselines/LightGBM.joblib"},
    "module3": {"model_key": "LightGBM", "artifact_relpath": "baselines/LightGBM.joblib"},
}


def _module_dir(module: str) -> Path:
    return MODELS_ROOT / module


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def winner_config(module: str) -> dict:
    path = _module_dir(module) / "deployment" / "winner.json"
    data = _read_json(path)
    if isinstance(data, dict) and data.get("model_key") and data.get("artifact_relpath"):
        return data
    fallback = DEFAULT_WINNERS[module]
    return {
        "module": module,
        "model_key": fallback["model_key"],
        "artifact_relpath": fallback["artifact_relpath"],
        "selection_rule": "best_validation_f1_then_pr_auc_then_lower_latency",
        "selected_on": "validation_only",
        "schema_id": MODULE_SCHEMA[module],
        "source": "default_winner",
    }


def resolve_artifact_path(module: str, relpath: str) -> Path:
    return (_module_dir(module) / relpath).resolve()


def _scan_artifacts(module: str) -> list[dict]:
    base = _module_dir(module)
    rows: list[dict] = []
    for sub, kind in (("baselines", "ml"), ("dl", "dl")):
        folder = base / sub
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".joblib", ".pt"}:
                continue
            # Skip quarantined legacy names only if under quarantine — here we list live folder
            key = path.stem
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            if not meta_path.exists():
                meta_path = path.with_name(path.stem + ".meta.json")
            meta = _read_json(meta_path) if meta_path.exists() else {}
            rows.append({
                "model_key": key,
                "kind": kind,
                "path": str(path),
                "relpath": f"{sub}/{path.name}",
                "meta": meta or {},
                "exists": True,
            })
    return rows


def list_models(module: str) -> dict:
    if module not in MODULE_SCHEMA:
        raise KeyError(module)
    winner = winner_config(module)
    models = _scan_artifacts(module)
    # Ensure winner appears even if file missing (UI can show unavailable)
    keys = {m["model_key"].lower() for m in models}
    if winner["model_key"].lower() not in keys:
        ap = resolve_artifact_path(module, winner["artifact_relpath"])
        models.insert(0, {
            "model_key": winner["model_key"],
            "kind": "dl" if ap.suffix.lower() == ".pt" else "ml",
            "path": str(ap),
            "relpath": winner["artifact_relpath"],
            "meta": {},
            "exists": ap.is_file(),
        })
    return {
        "module": module,
        "schema_id": MODULE_SCHEMA[module],
        "winner": winner,
        "models": models,
    }


def get_model_entry(module: str, model_key: str | None) -> dict:
    info = list_models(module)
    winner = info["winner"]
    key = (model_key or winner["model_key"]).strip()
    for row in info["models"]:
        if row["model_key"].lower() == key.lower():
            return {**row, "winner_config": winner, "schema_id": MODULE_SCHEMA[module]}
    # Fall back to winner path even if not scanned
    ap = resolve_artifact_path(module, winner["artifact_relpath"])
    return {
        "model_key": winner["model_key"] if not model_key else key,
        "kind": "dl" if ap.suffix.lower() == ".pt" else "ml",
        "path": str(ap),
        "relpath": winner["artifact_relpath"],
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
            "feature_names": list(obj.get("feature_names") or obj.get("features") or []),
            "label_name": obj.get("label_name"),
            "raw": obj,
        }
    # Raw estimator
    feature_names = []
    if hasattr(obj, "feature_name_"):
        try:
            feature_names = list(obj.feature_name_())
        except Exception:
            feature_names = []
    if not feature_names and hasattr(obj, "feature_names_in_"):
        feature_names = list(obj.feature_names_in_)
    return {"estimator": obj, "feature_names": feature_names, "label_name": None, "raw": obj}


def load_torch_artifact(path: Path, meta: dict | None = None) -> dict[str, Any]:
    import torch

    meta = meta or {}
    meta_path = path.with_name(path.stem + ".meta.json")
    if meta_path.exists():
        file_meta = _read_json(meta_path) or {}
        meta = {**file_meta, **meta}

    blob = torch.load(path, map_location="cpu", weights_only=False)
    return {"blob": blob, "meta": meta, "path": path}


@lru_cache(maxsize=32)
def load_cached(module: str, model_key: str) -> dict[str, Any]:
    entry = get_model_entry(module, model_key)
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing artifact for {module}/{model_key}: {path}. "
            "Copy clean models from Google Drive into the local models/ tree."
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
        meta = payload.get("meta") or {}
        feature_names = list(meta.get("feature_names") or [])
        mean = std = None
        if isinstance(blob, dict):
            feature_names = list(blob.get("features") or feature_names or [])
            mean = blob.get("mean")
            std = blob.get("std")
        return {
            "module": module,
            "model_key": entry["model_key"],
            "kind": "dl",
            "path": str(path),
            "schema_id": entry["schema_id"],
            "feature_names": feature_names,
            "mean": mean,
            "std": std,
            **payload,
        }
    raise ValueError(f"Unsupported artifact type: {path}")


def clear_cache() -> None:
    load_cached.cache_clear()


def health_flags() -> dict:
    out = {
        "rules_fallback_default": False,
        "models_root": str(MODELS_ROOT),
    }
    for module, winner in DEFAULT_WINNERS.items():
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
    return out


def load_clean_arena(module: str) -> list:
    """Prefer clean arena JSON; fall back to legacy for Lab only."""
    candidates = [
        METRICS_ROOT / f"arena_{module}_11models_clean.json",
        METRICS_ROOT / module / f"arena_{module}_11models_clean.json",
        METRICS_ROOT / module / f"arena_{module}_11models.json",
    ]
    for p in candidates:
        data = _read_json(p)
        if data is not None:
            return data if isinstance(data, list) else data.get("models", [])
    return []
