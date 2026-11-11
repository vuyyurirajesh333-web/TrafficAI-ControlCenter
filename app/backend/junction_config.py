"""Load junction geometry / calibration used by perception + geo APIs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
JUNCTION_DIR = ROOT / "config" / "junctions"
DEFAULT_JUNCTION_ID = "junction_01"


def _point_in_poly(x: float, y: float, poly: list[list[float]]) -> bool:
    """Ray casting; poly vertices are [x, y] in same units as point."""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


@lru_cache(maxsize=8)
def load_junction(junction_id: str = DEFAULT_JUNCTION_ID) -> dict[str, Any]:
    path = JUNCTION_DIR / f"{junction_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing junction config: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def list_junctions() -> list[dict[str, Any]]:
    rows = []
    if not JUNCTION_DIR.exists():
        return rows
    for p in sorted(JUNCTION_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            rows.append({
                "id": data.get("id", p.stem),
                "name": data.get("name"),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
                "camera_id": data.get("camera_id"),
                "gtfs_route_codes": data.get("gtfs_route_codes", []),
            })
        except Exception:
            continue
    return rows


def assign_approach(x: float, y: float, w: int, h: int, junction: dict[str, Any]) -> str:
    """Prefer configured ROI polygons (normalized 0-1); fall back to screen quadrants."""
    if w <= 0 or h <= 0:
        return "Unknown"
    nx, ny = x / w, y / h
    approaches = junction.get("approaches") or {}
    for name, cfg in approaches.items():
        poly = cfg.get("polygon_norm") or []
        if poly and _point_in_poly(nx, ny, poly):
            return name

    # Legacy quadrant fallback
    approach = "North" if y < h * 0.5 else "South"
    if x > w * 0.66:
        approach = "East"
    elif x < w * 0.33:
        approach = "West"
    return approach


def in_uturn_pocket(x: float, y: float, w: int, h: int, junction: dict[str, Any]) -> bool:
    pocket = (junction.get("uturn_pocket") or {}).get("polygon_norm") or []
    if not pocket or w <= 0 or h <= 0:
        return False
    return _point_in_poly(x / w, y / h, pocket)


def meters_per_pixel(junction: dict[str, Any]) -> float:
    try:
        return float(junction.get("meters_per_pixel") or 0.05)
    except (TypeError, ValueError):
        return 0.05


def approach_density(tracks: list[dict], approaches: list[str] | None = None) -> dict[str, int]:
    names = approaches or ["North", "South", "East", "West"]
    dens = {a: 0 for a in names}
    for t in tracks:
        if not t.get("is_vehicle"):
            continue
        a = t.get("approach")
        if a in dens:
            dens[a] += 1
    dens["total"] = sum(v for k, v in dens.items() if k != "total")
    return dens


def estimate_conflict_gap_sec(
    uturn_track: dict,
    tracks: list[dict],
    junction: dict[str, Any],
) -> tuple[float, str]:
    """Estimate seconds until a conflicting mover reaches the junction pocket.

    Uses distance_proxy_m / speed_proxy on conflicting approaches.
    Returns (gap_sec, source_label).
    """
    approach = uturn_track.get("approach")
    cfg = (junction.get("approaches") or {}).get(approach) or {}
    conflicts = set(cfg.get("conflicts_with") or [])
    if not conflicts:
        conflicts = {"North", "South", "East", "West"} - {approach}

    mpp = meters_per_pixel(junction)
    gaps: list[float] = []
    for t in tracks:
        if t.get("id") == uturn_track.get("id"):
            continue
        if not t.get("is_vehicle"):
            continue
        if t.get("approach") not in conflicts:
            continue
        if t.get("currently_stopped"):
            continue
        # px/s → m/s via same scale used for distance
        speed_mps = max(
            float(t.get("last_speed") or 0) * mpp,
            float(t.get("mean_recent_speed") or 0) * mpp,
            0.5,
        )
        dist_m = float(t.get("distance_proxy_m") or 0)
        if dist_m <= 0:
            continue
        gaps.append(dist_m / speed_mps)

    if not gaps:
        # No conflicting movers observed — safer long gap from empty approach
        return 12.0, "conflict_tracks_empty_default"

    gap = float(min(gaps))
    return round(gap, 2), "conflict_track_eta"
