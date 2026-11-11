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
from model_registry import ALLOWED_MODEL_KEYS, load_cached, winner_config
from demo_models import build_demo_sequence_model, sanitize_norm_stats, track_to_demo_feat_row

logger = logging.getLogger("traffic_ai.inference")

DEMO_SEQ_KEYS = {
    "gat_gru",
    "bilstm_attention",
    "tcn_attention",
    "transformer_feature_attention",
    "transformer_encoder_classweighted",
}

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


def _reconstruct_dl(blob: dict, n_features: int, model_key: str | None = None):
    """Build a module whose shapes match the checkpoint (module1/2 tabular vs module3 seq)."""
    import torch.nn as nn

    state = blob.get("state_dict") or blob.get("model_state_dict")
    if state is None:
        raise ModelInferenceError("DL artifact missing state_dict")
    name = str(blob.get("model_name") or model_key or "").lower().replace("-", "_")
    keys = list(state.keys())

    # Demo-allowed Attention / GAT family (seq checkpoints with feat_cols or seq_feats)
    demo_keys_hit = any(
        k.startswith(("lstm.", "gru.", "c1.", "conv1.", "in_proj.", "proj.", "feat."))
        or k.startswith(("enc.layers.", "encoder.layers."))
        for k in keys
    )
    if name in DEMO_SEQ_KEYS or (
        demo_keys_hit and ("feat_cols" in blob or "seq_feats" in blob or "seq_len" in blob)
    ):
        try:
            key = model_key or blob.get("model_name") or name
            return build_demo_sequence_model(str(key), state)
        except Exception as exc:
            raise ModelInferenceError(f"Demo architecture rebuild failed for {model_key}: {exc}") from exc

    # ---- MLP variants (weights live under self.net.*) ----
    if name == "mlp" or any(k.startswith("net.") for k in keys):
        if "net.0.weight" in state:
            in0 = state["net.0.weight"].shape[1]
            h0 = state["net.0.weight"].shape[0]
            h1 = state["net.3.weight"].shape[0]

            class MLPNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in0, h0),
                        nn.ReLU(),
                        nn.Dropout(0.0),
                        nn.Linear(h0, h1),
                        nn.ReLU(),
                        nn.Linear(h1, 1),
                    )
                    self.expected = in0

                def forward(self, x):
                    if x.shape[-1] != self.expected:
                        if x.shape[-1] < self.expected:
                            reps = (self.expected + x.shape[-1] - 1) // x.shape[-1]
                            x = x.repeat(1, reps)[:, : self.expected]
                        else:
                            x = x[:, : self.expected]
                    return self.net(x)

            model = MLPNet()
            model.load_state_dict(state, strict=True)
            return model

        if "net.1.weight" in state:
            in1 = state["net.1.weight"].shape[1]
            h1 = state["net.1.weight"].shape[0]
            h2 = state["net.4.weight"].shape[0]
            feat_dim = 7 if in1 % 7 == 0 else n_features

            class MLPFlat(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(in1, h1),
                        nn.ReLU(),
                        nn.Dropout(0.0),
                        nn.Linear(h1, h2),
                        nn.ReLU(),
                        nn.Linear(h2, 1),
                    )
                    self.expected = in1
                    self.feat_dim = feat_dim

                def forward(self, x):
                    feats = x[:, : self.feat_dim] if x.shape[-1] >= self.feat_dim else x
                    seq_len = max(1, self.expected // max(feats.shape[-1], 1))
                    tiled = feats.unsqueeze(1).repeat(1, seq_len, 1).reshape(x.shape[0], -1)
                    if tiled.shape[-1] < self.expected:
                        tiled = nn.functional.pad(tiled, (0, self.expected - tiled.shape[-1]))
                    else:
                        tiled = tiled[:, : self.expected]
                    return self.net(tiled)

            model = MLPFlat()
            model.load_state_dict(state, strict=True)
            return model

    # ---- Conv / TCN / TemporalCNN ----
    if any(k.startswith("conv.") for k in keys):
        c0 = state["conv.0.weight"]
        out0, in0, k0 = c0.shape[0], c0.shape[1], c0.shape[2]
        c2 = state["conv.2.weight"]
        out2, _, k2 = c2.shape[0], c2.shape[1], c2.shape[2]
        has_c4 = "conv.4.weight" in state
        head_is_seq = "head.0.weight" in state

        layers = [
            nn.Conv1d(in0, out0, kernel_size=k0, padding=k0 // 2),
            nn.ReLU(),
            nn.Conv1d(out0, out2, kernel_size=k2, padding=k2 // 2),
            nn.ReLU(),
        ]
        if has_c4:
            c4 = state["conv.4.weight"]
            out4, _, k4 = c4.shape[0], c4.shape[1], c4.shape[2]
            layers += [
                nn.Conv1d(out2, out4, kernel_size=k4, padding=k4 // 2),
                nn.ReLU(),
            ]
            conv_out = out4
        else:
            conv_out = out2
        conv = nn.Sequential(*layers)
        if head_is_seq:
            h0 = state["head.0.weight"].shape[0]
            head = nn.Sequential(
                nn.Linear(conv_out, h0),
                nn.ReLU(),
                nn.Linear(h0, 1),
            )
        else:
            head = nn.Linear(conv_out, 1)

        class ConvTab(nn.Module):
            def __init__(self, conv, head, in_ch):
                super().__init__()
                self.conv = conv
                self.head = head
                self.in_ch = in_ch

            def forward(self, x):
                # Support (B,F) with in_ch=1 -> unsqueeze channel
                # or in_ch=F treating length as pseudo-time with multi-channel needs reshape
                if self.in_ch == 1:
                    z = x.unsqueeze(1)
                else:
                    # (B, F) -> (B, C=in_ch, T) with T=1 by using features as channels
                    z = x.unsqueeze(-1)
                    if z.shape[1] != self.in_ch:
                        # pad/truncate channels
                        if z.shape[1] < self.in_ch:
                            z = nn.functional.pad(z, (0, 0, 0, self.in_ch - z.shape[1]))
                        else:
                            z = z[:, : self.in_ch, :]
                z = self.conv(z)
                z = z.mean(dim=-1)
                return self.head(z)

        model = ConvTab(conv, head, in0)
        model.load_state_dict(state, strict=True)
        return model

    # ---- RNN family ----
    if any(k.startswith("rnn.") for k in keys):
        w = state["rnn.weight_ih_l0"]
        # LSTM: 4*H, GRU: 3*H
        gate = 4 if "lstm" in name or w.shape[0] % 4 == 0 else 3
        if "gru" in name:
            gate = 3
        if "lstm" in name or "bilstm" in name:
            gate = 4
        hidden = w.shape[0] // gate
        input_size = w.shape[1]
        bidirectional = any(k.endswith("_reverse") for k in keys)
        n_layers = 1 + (1 if any(".weight_ih_l1" in k for k in keys) else 0)
        # detect more layers
        max_l = 0
        for k in keys:
            if "weight_ih_l" in k and not k.endswith("_reverse"):
                max_l = max(max_l, int(k.split("weight_ih_l")[-1].split("_")[0]))
        n_layers = max_l + 1
        rnn_cls = nn.GRU if gate == 3 else nn.LSTM
        rnn = rnn_cls(
            input_size,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        if "head.0.weight" in state:
            h0 = state["head.0.weight"].shape[0]
            head = nn.Sequential(
                nn.Linear(out_dim, h0),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Linear(h0, 1),
            )
        else:
            head = nn.Linear(out_dim, 1)

        class RNNWrap(nn.Module):
            def __init__(self, rnn, head, input_size):
                super().__init__()
                self.rnn = rnn
                self.head = head
                self.input_size = input_size

            def forward(self, x):
                # x (B,F)
                if self.input_size == 1:
                    z = x.unsqueeze(-1)  # (B,F,1)
                else:
                    feats = x
                    if feats.shape[-1] < self.input_size:
                        feats = nn.functional.pad(feats, (0, self.input_size - feats.shape[-1]))
                    else:
                        feats = feats[:, : self.input_size]
                    # repeat to short sequence
                    z = feats.unsqueeze(1).repeat(1, 16, 1)
                out, _ = self.rnn(z)
                return self.head(out[:, -1, :])

        model = RNNWrap(rnn, head, input_size)
        model.load_state_dict(state, strict=True)
        return model

    # ---- Transformer ----
    if any(k.startswith("emb.") for k in keys):
        d_model = state["emb.weight"].shape[0]
        in_dim = state["emb.weight"].shape[1]
        n_layers = 1 + max(
            int(k.split("enc.layers.")[1].split(".")[0])
            for k in keys
            if k.startswith("enc.layers.")
        )
        # nhead from in_proj 3*d
        nhead = 4 if d_model % 4 == 0 else (8 if d_model % 8 == 0 else 2)
        ff = state.get("enc.layers.0.linear1.weight")
        dim_ff = ff.shape[0] if ff is not None else d_model * 2
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True
        )
        enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        if "head.0.weight" in state:
            h0 = state["head.0.weight"].shape[0]
            head = nn.Sequential(nn.Linear(d_model, h0), nn.ReLU(), nn.Linear(h0, 1))
        else:
            head = nn.Linear(d_model, 1)

        class TrWrap(nn.Module):
            def __init__(self, emb, enc, head, in_dim):
                super().__init__()
                self.emb = emb
                self.enc = enc
                self.head = head
                self.in_dim = in_dim

            def forward(self, x):
                if self.in_dim == 1:
                    z = self.emb(x.unsqueeze(-1))
                else:
                    feats = x
                    if feats.shape[-1] < self.in_dim:
                        feats = nn.functional.pad(feats, (0, self.in_dim - feats.shape[-1]))
                    else:
                        feats = feats[:, : self.in_dim]
                    z = self.emb(feats).unsqueeze(1)  # (B,1,d)
                    z = z.repeat(1, 8, 1)
                z = self.enc(z)
                return self.head(z.mean(dim=1))

        emb = nn.Linear(in_dim, d_model)
        model = TrWrap(emb, enc, head, in_dim)
        model.load_state_dict(state, strict=True)
        return model

    raise ModelInferenceError(f"Unsupported DL architecture: {name or keys[:8]}")


def _predict_torch(artifact: dict, values: list[float]) -> float:
    import torch
    import torch.nn as nn

    blob = artifact["blob"]
    arr = np.asarray(values, dtype=np.float32).reshape(1, -1)
    model_key = str(artifact.get("model_key") or "")
    seq_len = int(artifact.get("seq_len") or (blob.get("seq_len") if isinstance(blob, dict) else 0) or 0)

    mean = artifact.get("mean")
    std = artifact.get("std")
    if mean is None and isinstance(blob, dict):
        mean = blob.get("mean")
        std = blob.get("std")
    mean, std = sanitize_norm_stats(mean, std)
    if mean is not None and std is not None:
        mean_a = np.asarray(mean, dtype=np.float32).reshape(-1)
        std_a = np.asarray(std, dtype=np.float32).reshape(-1)
        if mean_a.size == arr.shape[1]:
            arr = (arr - mean_a.reshape(1, -1)) / np.clip(std_a.reshape(1, -1), 1e-6, None)

    # Demo Attention/GAT models expect (B, T, F)
    is_demo_seq = model_key.lower() in DEMO_SEQ_KEYS or (
        isinstance(blob, dict)
        and seq_len
        and (blob.get("feat_cols") or blob.get("seq_feats"))
    )
    if is_demo_seq:
        t = int(seq_len) if seq_len else 12
        seq = np.repeat(arr[:, None, :], t, axis=1)
        x = torch.tensor(seq, dtype=torch.float32)
    else:
        x = torch.tensor(arr, dtype=torch.float32)

    if isinstance(blob, nn.Module):
        model = blob
    elif isinstance(blob, dict):
        if isinstance(blob.get("model"), nn.Module):
            model = blob["model"]
        else:
            try:
                model = _reconstruct_dl(blob, n_features=arr.shape[1], model_key=model_key)
            except Exception as exc:
                raise ModelInferenceError(
                    f"Could not reconstruct DL model from {artifact.get('path')}: {exc}"
                ) from exc
    else:
        raise ModelInferenceError(f"Unsupported torch payload in {artifact.get('path')}")

    model.eval()
    with torch.no_grad():
        out = model(x)

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


def _features_for_track(module: str, model_key: str, track: dict, perception: dict) -> dict:
    builder = BUILDERS[module]
    feats = builder(track, perception)
    # Merge demo legacy feature aliases for Attention/GAT checkpoints
    if model_key in ALLOWED_MODEL_KEYS:
        feats = {**feats, **track_to_demo_feat_row(track, perception)}
    return feats


def score_tracks_for_module(
    module: str,
    model_key: str,
    perception: dict,
    candidate_tracks: list[dict],
) -> dict[str, Any]:
    if not model_key:
        model_key = winner_config(module)["model_key"]
    if not candidate_tracks:
        empty = {
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
        }
        feats = _features_for_track(module, model_key, empty, perception)
        out = score_feature_row(module, model_key, feats)
        out["track_id"] = None
        return out

    best = None
    for track in candidate_tracks:
        feats = _features_for_track(module, model_key, track, perception)
        scored = score_feature_row(module, model_key, feats)
        scored["track_id"] = track.get("id")
        scored["track"] = track
        if best is None or scored["model_score"] > best["model_score"]:
            best = scored
    assert best is not None
    return best
