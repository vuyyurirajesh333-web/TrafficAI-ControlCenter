"""Delhi GTFS / GPS geo helpers using datasets/processed (Phase 3)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
M1 = ROOT / "datasets" / "processed" / "module1_bus"


@lru_cache(maxsize=1)
def _stops_df() -> pd.DataFrame:
    path = M1 / "gtfs_stops.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def _routes_df() -> pd.DataFrame:
    path = M1 / "gtfs_routes.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def list_stops(limit: int = 500, bbox: tuple[float, float, float, float] | None = None) -> list[dict]:
    df = _stops_df()
    if df.empty:
        return []
    out = df
    if bbox:
        min_lat, min_lon, max_lat, max_lon = bbox
        out = out[
            (out["stop_lat"] >= min_lat)
            & (out["stop_lat"] <= max_lat)
            & (out["stop_lon"] >= min_lon)
            & (out["stop_lon"] <= max_lon)
        ]
    out = out.head(limit)
    return [
        {
            "stop_id": int(r.stop_id) if pd.notna(r.stop_id) else None,
            "stop_code": None if pd.isna(r.stop_code) else str(r.stop_code),
            "stop_name": None if pd.isna(r.stop_name) else str(r.stop_name),
            "lat": float(r.stop_lat),
            "lon": float(r.stop_lon),
        }
        for r in out.itertuples()
    ]


def route_node_map(route_code: str) -> dict[str, Any]:
    path = M1 / f"gtfs_node_map_route_{route_code}.csv"
    if not path.exists():
        return {"route_code": route_code, "stops": [], "error": "node map not found"}
    df = pd.read_csv(path)
    stops = []
    for r in df.itertuples():
        stops.append({
            "stop_sequence": int(getattr(r, "stop_sequence", 0)),
            "stop_id": int(r.stop_id) if pd.notna(r.stop_id) else None,
            "stop_name": str(r.stop_name) if pd.notna(r.stop_name) else None,
            "lat": float(r.stop_lat),
            "lon": float(r.stop_lon),
        })
    return {
        "route_code": str(route_code),
        "stops": stops,
        "source": str(path.relative_to(ROOT)),
    }


def gps_hotspots(limit_cells: int = 40) -> dict[str, Any]:
    """Aggregate bus_gps_points into coarse lat/lon grid (historical, not live)."""
    path = M1 / "bus_gps_points.parquet"
    if not path.exists():
        return {"cells": [], "note": "bus_gps_points.parquet missing"}

    # Sample for responsiveness on large table
    df = pd.read_parquet(path, columns=["stop_lat", "stop_lon", "route_code"])
    df = df.dropna(subset=["stop_lat", "stop_lon"])
    if len(df) > 200_000:
        df = df.sample(n=200_000, random_state=42)

    # ~0.005 deg ≈ 500m cells
    df = df.copy()
    df["lat_bin"] = (df["stop_lat"] / 0.005).round().astype(int)
    df["lon_bin"] = (df["stop_lon"] / 0.005).round().astype(int)
    grouped = (
        df.groupby(["lat_bin", "lon_bin"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(limit_cells)
    )
    cells = []
    for r in grouped.itertuples():
        cells.append({
            "lat": float(r.lat_bin * 0.005),
            "lon": float(r.lon_bin * 0.005),
            "count": int(r.count),
        })
    return {
        "cells": cells,
        "grid_deg": 0.005,
        "note": "Historical GPS density from collected bus logs — not live congestion",
        "source": "datasets/processed/module1_bus/bus_gps_points.parquet",
    }


def geo_summary() -> dict[str, Any]:
    stops = _stops_df()
    routes = _routes_df()
    return {
        "stops_available": int(len(stops)),
        "routes_available": int(len(routes)),
        "dataset_root": str(M1),
        "node_maps": [
            p.stem.replace("gtfs_node_map_route_", "")
            for p in sorted(M1.glob("gtfs_node_map_route_*.csv"))
        ],
    }
