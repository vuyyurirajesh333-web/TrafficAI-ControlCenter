"""Reconstruct notebook-era Attention / GAT_GRU checkpoints for demo inference.

Shapes and naming differ by module:
  module1: 8 feats, 3-class head, often `encoder.*` / `conv1.*`
  module2: 12 feats, binary head, `enc.*` / `c1.*`
  module3: 7 feats, binary head, `enc.*` / `c1.*`
"""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, h):
        a = torch.softmax(self.w(h), dim=1)
        return (a * h).sum(dim=1)


class BiLSTMAttention(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, n_out: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True, bidirectional=True)
        self.attn = AttnPool(hidden * 2)
        self.fc = nn.Linear(hidden * 2, n_out)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(self.attn(h))


class TCNAttention(nn.Module):
    def __init__(self, n_feat: int, channels: int = 64, n_out: int = 1):
        super().__init__()
        self.c1 = nn.Conv1d(n_feat, channels, 3, padding=1)
        self.c2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.c3 = nn.Conv1d(channels, channels, 3, padding=1)
        self.attn = AttnPool(channels)
        self.fc = nn.Linear(channels, n_out)

    def forward(self, x):
        z = x.transpose(1, 2)
        z = F.relu(self.c1(z))
        z = F.relu(self.c2(z))
        z = F.relu(self.c3(z))
        z = z.transpose(1, 2)
        return self.fc(self.attn(z))


class TransformerFeatureAttention(nn.Module):
    def __init__(self, n_feat: int, d_model: int = 64, n_layers: int = 2, n_out: int = 1):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(n_feat, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_feat),
        )
        self.proj = nn.Linear(n_feat, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.attn = AttnPool(d_model)
        self.fc = nn.Linear(d_model, n_out)

    def forward(self, x):
        x = x + self.feat(x)
        z = self.proj(x)
        z = self.enc(z)
        return self.fc(self.attn(z))


class TransformerEncoderClassWeighted(nn.Module):
    def __init__(self, n_feat: int, d_model: int = 64, n_layers: int = 3, n_out: int = 1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_out)

    def forward(self, x):
        z = self.proj(x)
        z = self.enc(z)
        return self.fc(z.mean(dim=1))


class GATGRU(nn.Module):
    def __init__(self, n_feat: int, d_model: int = 32, hidden: int = 64, n_out: int = 1):
        super().__init__()
        self.in_proj = nn.Linear(n_feat, d_model)
        self.attn_l = nn.Linear(d_model * 2, 1)
        self.gru = nn.GRU(d_model, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_out)

    def forward(self, x):
        h = self.in_proj(x)
        mean = h.mean(dim=1, keepdim=True).expand_as(h)
        cat = torch.cat([h, mean], dim=-1)
        a = torch.softmax(self.attn_l(cat), dim=1)
        h = h * a
        out, _ = self.gru(h)
        return self.fc(out[:, -1, :])


def _normalize_state_keys(state: dict) -> dict:
    """Map module1 naming (encoder.*, convN.*) onto canonical demo names."""
    out = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("encoder."):
            nk = "enc." + nk[len("encoder.") :]
        nk = re.sub(r"^conv(\d)\.", r"c\1.", nk)
        out[nk] = v
    return out


def _infer_dims(state: dict) -> dict[str, int]:
    n_feat = None
    n_out = 1
    d_model = 64
    hidden = 64
    n_layers = 2
    channels = 64

    if "in_proj.weight" in state:
        n_feat = int(state["in_proj.weight"].shape[1])
        d_model = int(state["in_proj.weight"].shape[0])
    if "lstm.weight_ih_l0" in state:
        n_feat = int(state["lstm.weight_ih_l0"].shape[1])
        # bidirectional LSTM: 4 * hidden
        hidden = int(state["lstm.weight_ih_l0"].shape[0] // 4)
    if "c1.weight" in state:
        channels = int(state["c1.weight"].shape[0])
        n_feat = int(state["c1.weight"].shape[1])
    if "feat.0.weight" in state:
        n_feat = int(state["feat.0.weight"].shape[1])
        d_model = int(state["feat.0.weight"].shape[0])
    if "proj.weight" in state:
        n_feat = int(state["proj.weight"].shape[1])
        d_model = int(state["proj.weight"].shape[0])
    if "fc.weight" in state:
        n_out = int(state["fc.weight"].shape[0])
        if "gru.weight_hh_l0" in state:
            hidden = int(state["gru.weight_hh_l0"].shape[1])
        elif "lstm.weight_ih_l0" not in state:
            hidden = int(state["fc.weight"].shape[1])

    layer_ids = []
    for k in state:
        m = re.match(r"enc\.layers\.(\d+)\.", k)
        if m:
            layer_ids.append(int(m.group(1)))
    if layer_ids:
        n_layers = max(layer_ids) + 1

    if n_feat is None:
        raise ValueError("Could not infer n_feat from checkpoint state_dict")
    return {
        "n_feat": n_feat,
        "n_out": n_out,
        "d_model": d_model,
        "hidden": hidden,
        "n_layers": n_layers,
        "channels": channels,
    }


def build_demo_sequence_model(model_key: str, state: dict) -> nn.Module:
    state = _normalize_state_keys(dict(state))
    dims = _infer_dims(state)
    key = model_key.lower().replace("-", "_")

    if key == "bilstm_attention":
        model = BiLSTMAttention(dims["n_feat"], hidden=dims["hidden"], n_out=dims["n_out"])
    elif key == "tcn_attention":
        model = TCNAttention(dims["n_feat"], channels=dims["channels"], n_out=dims["n_out"])
    elif key == "transformer_feature_attention":
        model = TransformerFeatureAttention(
            dims["n_feat"], d_model=dims["d_model"], n_layers=dims["n_layers"], n_out=dims["n_out"]
        )
    elif key == "transformer_encoder_classweighted":
        model = TransformerEncoderClassWeighted(
            dims["n_feat"], d_model=dims["d_model"], n_layers=dims["n_layers"], n_out=dims["n_out"]
        )
    elif key == "gat_gru":
        model = GATGRU(
            dims["n_feat"], d_model=dims["d_model"], hidden=dims["hidden"], n_out=dims["n_out"]
        )
    else:
        raise ValueError(f"Unsupported demo model_key: {model_key}")

    model.load_state_dict(state, strict=True)
    return model


def track_to_demo_feat_row(track: dict, perception: dict | None = None) -> dict:
    """Map live track kinematics into checkpoint feature names (no leakage flags)."""
    perception = perception or {}
    fps = float(perception.get("fps") or 30.0)
    duration = float(track.get("duration_sec") or 0.0)
    mean_sp = float(track.get("mean_recent_speed") or 0.0)
    max_sp = float(track.get("max_speed") or 0.0)
    last_sp = float(track.get("last_speed") or 0.0)
    stationary_ratio = float(track.get("stationary_ratio") or 0.0)
    dist = float(track.get("distance_proxy_m") or 0.0)
    mpp = float(track.get("distance_scale_mpp") or 0.05)
    speed_mps = mean_sp * mpp
    eta = dist / max(speed_mps, 0.15)
    approach = str(track.get("approach") or "unknown")
    approach_code = {"North": 0, "South": 1, "East": 2, "West": 3, "NS": 0}.get(approach, 4)
    direction_up = 1.0 if approach in {"North", "East", "NS"} else 0.0
    xy = track.get("last_xy") or [0.0, 0.0]
    x = float(xy[0] if isinstance(xy, (list, tuple)) else 0.0)
    y = float(xy[1] if isinstance(xy, (list, tuple)) else 0.0)
    heading = float(track.get("net_heading") or 0.0)
    heading_delta = float(track.get("peak_heading") or heading)
    # crude dx/dy from speed + heading
    dx = float(last_sp * max(duration, 1.0) / max(float(track.get("frames") or 1), 1.0))
    dy = float(mean_sp - last_sp)
    route_code = float(hash(str(track.get("id") or 0)) % 1000)

    return {
        # module2 legacy 12-d
        "t_sec": float(track.get("last_t") or duration),
        "duration_sec": duration,
        "rel_time": 0.5,
        "fps": fps,
        "motion_mean": mean_sp,
        "motion_std": abs(max_sp - mean_sp),
        "motion_p95": max_sp,
        "stationary_score": stationary_ratio,
        "motion_mean_z": 0.0,
        "low_motion_flag": float(1 if track.get("currently_stopped") else 0),
        "rule_stop_hint": float(1 if track.get("sudden_stop_candidate") else 0),
        "has_any_anomaly_in_video": 0.0,
        # module1 bus 8-d
        "dist_to_junction_m": dist,
        "speed_mps": speed_mps,
        "eta_sec": float(eta),
        "lateness_sec": float(max(0.0, eta - 30.0)),
        "direction_up": direction_up,
        "node": float(approach_code),
        "route_code": route_code,
        # module3 seq 7-d
        "x": x,
        "y": y,
        "speed": last_sp if last_sp else mean_sp,
        "dx": dx,
        "dy": dy,
        "heading": heading,
        "heading_delta": heading_delta,
    }


def sanitize_norm_stats(mean: Any, std: Any) -> tuple[Any, Any]:
    """Replace NaNs in checkpoint norm arrays."""
    import numpy as np

    if mean is None or std is None:
        return mean, std
    mean_a = np.asarray(mean, dtype=np.float32)
    std_a = np.asarray(std, dtype=np.float32)
    mean_a = np.nan_to_num(mean_a, nan=0.0)
    std_a = np.nan_to_num(std_a, nan=1.0)
    std_a = np.where(std_a < 1e-6, 1.0, std_a)
    return mean_a, std_a
