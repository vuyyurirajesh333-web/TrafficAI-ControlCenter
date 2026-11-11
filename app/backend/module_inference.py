"""Run selected ML/DL module models on track-derived features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from feature_builder import (
    MODULE1_FEATURES,
    MODULE2_FEATURES,
    MODULE3_FEATURES,
    build_module1_features,
    build_module2_features,
    build_module3_features,
    vectorize,
)
from model_registry import load_cached

logger = logging.getLogger("traffic_ai.inference")

DEFAULT_FEATURES = {
    "module1": MODULE1_FEATURES,
    "module2": MODULE2_FEATURES,
    "module3": MODULE3_FEATURES,
}

BUILDERS = {
    "module1": build_module1_features,
    "module2": build_module2_features,
    "module3": build_module3_features,
}


class ModelInferenceError(Exception):
    """Raised when a selected model cannot run."""


def _predict_proba_positive(estimator, X: np.ndarray) -> float:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, -1])
        return float(proba.ravel()[0])
    if hasattr(estimator, "decision_function"):
        score = float(np.asarray(estimator.decision_function(X)).ravel()[0])
        # map to (0,1) via logistic
        return float(1.0 / (1.0 + np.exp(-score)))
    pred = estimator.predict(X)
    return float(np.asarray(pred).ravel()[0])


def _build_tabular_tcn():
    import torch.nn as nn

    class TabularTCN(nn.Module):
        """Matches Colab export: Conv1d(1→32→32,k=3,p=1) + mean pool + Linear(32→1)."""

        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            # x: (B, F)
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.mean(dim=-1)
            return self.head(x)

    return TabularTCN


def _predict_torch(artifact: dict, values: list[float]) -> float:
    import torch
    import torch.nn as nn

    blob = artifact["blob"]
    meta = artifact.get("meta") or {}
    arr = np.asarray(values, dtype=np.float32).reshape(1, -1)

    mean = artifact.get("mean")
    std = artifact.get("std")
    if mean is None and isinstance(blob, dict):
        mean = blob.get("mean")
        std = blob.get("std")
    if mean is not None and std is not None:
        mean_a = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        std_a = np.asarray(std, dtype=np.float32).reshape(1, -1)
        if mean_a.shape[1] == arr.shape[1]:
            arr = (arr - mean_a) / np.clip(std_a, 1e-6, None)

    x = torch.tensor(arr, dtype=torch.float32)

    # Full nn.Module saved
    if isinstance(blob, nn.Module):
        model = blob
        model.eval()
        with torch.no_grad():
            out = model(x)
    elif isinstance(blob, dict):
        state = blob.get("state_dict") or blob.get("model_state_dict")
        model_name = str(blob.get("model_name") or meta.get("model_name") or "").lower()
        if isinstance(blob.get("model"), nn.Module):
            model = blob["model"]
        elif state is not None and (
            model_name == "tcn" or any(k.startswith("conv.") for k in state.keys())
        ):
            TabularTCN = _build_tabular_tcn()
            model = TabularTCN()
            try:
                model.load_state_dict(state, strict=True)
            except Exception as exc:
                raise ModelInferenceError(
                    f"Could not load TCN state_dict from {artifact.get('path')}: {exc}"
                ) from exc
        else:
            n_features = int(meta.get("n_features") or len(values))
            hidden = int(meta.get("hidden") or 64)
            model = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, max(2, int(meta.get("n_classes", 2)))),
            )
            if state is not None:
                try:
                    model.load_state_dict(state, strict=False)
                except Exception as exc:
                    raise ModelInferenceError(
                        f"Could not reconstruct DL model from {artifact.get('path')}: {exc}"
                    ) from exc
        model.eval()
        with torch.no_grad():
            try:
                out = model(x)
            except Exception:
                out = model(x.unsqueeze(1))
    else:
        raise ModelInferenceError(f"Unsupported torch payload in {artifact.get('path')}")

    out_t = torch.as_tensor(out).detach().cpu().float().view(-1)
    if out_t.numel() == 1:
        val = float(out_t.item())
        if 0.0 <= val <= 1.0:
            return val
        return float(torch.sigmoid(out_t).item())
    probs = torch.softmax(out_t, dim=0)
    return float(probs[-1].item())


def score_feature_row(module: str, model_key: str, features: dict) -> dict[str, Any]:
    try:
        artifact = load_cached(module, model_key)
    except FileNotFoundError as exc:
        raise ModelInferenceError(str(exc)) from exc
    except Exception as exc:
        raise ModelInferenceError(f"Failed loading {module}/{model_key}: {exc}") from exc

    names, values = vectorize(
        features,
        artifact.get("feature_names"),
        DEFAULT_FEATURES[module],
    )
    X = np.asarray([values], dtype=np.float32)

    try:
        if artifact["kind"] == "ml":
            score = _predict_proba_positive(artifact["estimator"], X)
        else:
            score = _predict_torch(artifact, values)
    except ModelInferenceError:
        raise
    except Exception as exc:
        raise ModelInferenceError(f"Inference failed for {module}/{model_key}: {exc}") from exc

    score = float(max(0.0, min(1.0, score)))
    return {
        "selected_model": artifact["model_key"],
        "model_version": str((artifact.get("meta") or {}).get("artifact_version") or "1.0"),
        "schema_id": artifact.get("schema_id"),
        "model_score": round(score, 6),
        "decision_source": "model_inference",
        "features_used": {n: float(v) for n, v in zip(names, values)},
        "artifact_path": artifact.get("path"),
    }


def score_tracks_for_module(
    module: str,
    model_key: str,
    perception: dict,
    candidate_tracks: list[dict],
) -> dict[str, Any]:
    if not candidate_tracks:
        builder = BUILDERS[module]
        # empty features → score 0
        feats = builder(
            {
                "id": None,
                "is_bus": 0,
                "is_vehicle": 0,
                "approach": "unknown",
                "frames": 0,
                "stationary_ratio": 0,
                "max_speed": 0,
                "last_speed": 0,
                "mean_recent_speed": 0,
                "stationary_duration_sec": 0,
                "moving_frames": 0,
                "was_moving": 0,
                "currently_stopped": 0,
                "distance_proxy_m": 0,
                "duration_sec": 0,
                "tortuosity": 1,
                "net_heading": 0,
                "peak_heading": 0,
                "in_uturn_pocket": 0,
            },
            perception,
        )
        out = score_feature_row(module, model_key, feats)
        out["track_id"] = None
        return out

    builder = BUILDERS[module]
    best = None
    for track in candidate_tracks:
        feats = builder(track, perception)
        scored = score_feature_row(module, model_key, feats)
        scored["track_id"] = track.get("id")
        scored["track"] = track
        if best is None or scored["model_score"] > best["model_score"]:
            best = scored
    assert best is not None
    return best
