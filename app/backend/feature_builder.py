"""Build module feature dicts from YOLO+ByteTrack track summaries."""

from __future__ import annotations

from typing import Any, Callable

# Schema defaults (fallback if artifact has no feature list)
MODULE1_FEATURES = [
    "dist_to_junction_m",
    "speed_mps",
    "node",
    "direction_up",
]

MODULE2_FEATURES = [
    "frames",
    "stationary_frames",
    "mean_speed",
    "turn_amount",
    "distance_proxy_m",
    "is_bus",
    "cls",
]

MODULE3_FEATURES = [
    "frames",
    "mean_speed",
    "std_speed",
    "mean_acc",
    "std_acc",
    "path_len",
    "displacement",
    "lane_changes",
    "start_lane",
    "end_lane",
]

APPROACH_CODE = {"North": 0, "South": 1, "East": 2, "West": 3, "unknown": 4, "NS": 0}


def _approach_code(name: Any) -> int:
    if name is None:
        return 4
    return APPROACH_CODE.get(str(name), 4)


def _mpp(track: dict) -> float:
    return float(track.get("distance_scale_mpp") or 0.05)


def _speed_mps(track: dict) -> float:
    px = float(track.get("mean_recent_speed") or track.get("last_speed") or 0.0)
    return px * _mpp(track)


def _density_for(track: dict, perception: dict) -> float:
    dens = perception.get("approach_density") or {}
    approach = track.get("approach")
    if approach in dens:
        return float(dens[approach])
    return float(dens.get("total") or 0)


def _peer_stats(track: dict, tracks: list[dict]) -> tuple[int, float]:
    others = [t for t in tracks if t.get("is_vehicle") and t.get("id") != track.get("id")]
    if not others:
        return 0, 0.0
    movers = 0
    for t in others:
        if (
            float(t.get("last_speed") or 0) >= 8.0
            or float(t.get("mean_recent_speed") or 0) >= 6.0
            or (float(t.get("moving_frames") or 0) / max(float(t.get("frames") or 1), 1.0)) > 0.35
        ):
            movers += 1
    return movers, movers / max(len(others), 1)


def build_module1_features(track: dict, perception: dict, signal_phase_code: int = 0) -> dict:
    dist = float(track.get("distance_proxy_m") or 0.0)
    speed = _speed_mps(track)
    mean_recent = float(track.get("mean_recent_speed") or 0.0)
    last_speed = float(track.get("last_speed") or 0.0)
    accel_proxy = (mean_recent - last_speed) * _mpp(track)
    eta = dist / max(speed, 0.15)
    node = _approach_code(track.get("approach"))
    direction_up = 1 if str(track.get("approach")) in {"North", "East", "NS"} else 0
    duration = float(track.get("duration_sec") or 0.0)
    return {
        # Clean / demo Attention artifact names (8-d GAT/BiLSTM/TCN/Transformer)
        "dist_to_junction_m": dist,
        "speed_mps": speed,
        "node": float(node),
        "direction_up": float(direction_up),
        "eta_sec": float(eta),
        "lateness_sec": float(max(0.0, eta - 30.0)),
        "t_sec": float(track.get("last_t") or duration),
        "route_code": float(hash(str(track.get("id") or 0)) % 1000),
        # Schema / compatibility aliases
        "is_bus": int(bool(track.get("is_bus"))),
        "distance_proxy_m": dist,
        "speed_mps_proxy": speed,
        "accel_proxy": float(accel_proxy),
        "eta_proxy_sec": float(eta),
        "approach_id": node,
        "approach_density": _density_for(track, perception),
        "signal_phase_code": int(signal_phase_code or 0),
        "track_age_sec": duration,
        "mean_recent_speed": mean_recent,
    }


def build_module2_features(track: dict, perception: dict) -> dict:
    tracks = perception.get("tracks") or []
    peer_count, peer_ratio = _peer_stats(track, tracks)
    frames = int(track.get("frames") or 0)
    stationary_frames = int(track.get("stationary_frames") or 0)
    mean_speed = float(track.get("mean_recent_speed") or 0.0)
    return {
        # Clean LightGBM artifact names
        "frames": frames,
        "stationary_frames": stationary_frames,
        "mean_speed": mean_speed,
        "turn_amount": float(track.get("turn_amount") or track.get("uturn_score") or 0.0),
        "distance_proxy_m": float(track.get("distance_proxy_m") or 0.0),
        "is_bus": int(bool(track.get("is_bus"))),
        "cls": int(track.get("cls") or 0),
        # Extra schema fields (harmless if unused)
        "stationary_ratio": float(track.get("stationary_ratio") or 0.0),
        "max_speed": float(track.get("max_speed") or 0.0),
        "last_speed": float(track.get("last_speed") or 0.0),
        "mean_recent_speed": mean_speed,
        "stationary_duration_sec": float(track.get("stationary_duration_sec") or 0.0),
        "moving_frames": int(track.get("moving_frames") or 0),
        "was_moving": int(bool(track.get("was_moving"))),
        "currently_stopped": int(bool(track.get("currently_stopped"))),
        "peer_moving_count": int(peer_count),
        "peer_moving_ratio": float(peer_ratio),
        "approach_density": _density_for(track, perception),
        "in_travel_lane_proxy": 1,
    }


def build_module3_features(track: dict, perception: dict) -> dict:
    tracks = perception.get("tracks") or []
    waiting = sum(
        1
        for t in tracks
        if t.get("is_vehicle")
        and t.get("currently_stopped")
        and t.get("approach") == track.get("approach")
    )
    duration = float(track.get("duration_sec") or 0.0)
    mean_speed = float(track.get("mean_recent_speed") or 0.0)
    max_speed = float(track.get("max_speed") or 0.0)
    last_speed = float(track.get("last_speed") or 0.0)
    std_speed = float(track.get("std_speed") or abs(max_speed - mean_speed))
    mean_acc = mean_speed - last_speed
    std_acc = abs(mean_acc)
    path_len = float(track.get("path_length") or max_speed * max(duration, 1.0))
    displacement = float(track.get("displacement") or track.get("distance_proxy_m") or 0.0)
    approach = _approach_code(track.get("approach"))
    xy = track.get("last_xy") or [0.0, 0.0]
    x = float(xy[0] if isinstance(xy, (list, tuple)) else 0.0)
    y = float(xy[1] if isinstance(xy, (list, tuple)) else 0.0)
    heading = float(track.get("net_heading") or 0.0)
    heading_delta = float(track.get("peak_heading") or heading)
    frames = int(track.get("frames") or 0)
    dx = float(last_speed * max(duration, 1.0) / max(frames, 1))
    dy = float(mean_speed - last_speed)
    return {
        # Clean LightGBM artifact names
        "frames": frames,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "mean_acc": float(mean_acc),
        "std_acc": float(std_acc),
        "path_len": path_len,
        "path_length": path_len,
        "displacement": displacement,
        "lane_changes": float(track.get("lane_changes") or 0.0),
        "start_lane": float(track.get("start_lane") or approach),
        "end_lane": float(track.get("end_lane") or approach),
        # Demo seq_feats (7-d Attention/GAT for module3)
        "x": x,
        "y": y,
        "speed": last_speed if last_speed else mean_speed,
        "dx": dx,
        "dy": dy,
        "heading": heading,
        "heading_delta": heading_delta,
        # Schema / UI extras
        "duration_sec": duration,
        "max_speed": max_speed,
        "tortuosity": float(track.get("tortuosity") or 1.0),
        "net_heading_change": heading,
        "peak_heading_change": heading_delta,
        "heading_reversal_ratio": float(min(1.0, abs(heading) / 3.1416)),
        "in_uturn_pocket": int(bool(track.get("in_uturn_pocket"))),
        "approach_id_start": approach,
        "approach_id_end": approach,
        "waiting_vehicle_count": int(waiting),
        "approach_density": _density_for(track, perception),
    }


def vectorize(features: dict, feature_names: list[str] | None, default_names: list[str]) -> tuple[list[str], list[float]]:
    names = list(feature_names) if feature_names else list(default_names)
    alias = {
        "speed_mps_proxy": "speed_mps",
        "eta_proxy_sec": "eta_sec",
        "approach": "approach_id",
        "path_length": "path_len",
    }
    values = []
    used = []
    for name in names:
        if name in features:
            values.append(float(features[name]))
        else:
            src = alias.get(name)
            if src and src in features:
                values.append(float(features[src]))
            elif name in alias.values():
                # reverse lookup
                rev = next((k for k, v in alias.items() if v == name and k in features), None)
                values.append(float(features[rev]) if rev else 0.0)
            else:
                values.append(0.0)
        used.append(name)
    return used, values
