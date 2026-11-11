"""Phase 2/3 — junction ROI + geo dataset wiring."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from junction_config import assign_approach, load_junction, estimate_conflict_gap_sec
import geo as geo_service


def test_load_default_junction():
    j = load_junction("junction_01")
    assert j["lat"] == 28.5411
    assert "North" in j["approaches"]


def test_roi_approach_assignment():
    j = load_junction("junction_01")
    # Normalized north polygon center-ish
    assert assign_approach(320, 100, 640, 480, j) == "North"
    assert assign_approach(320, 400, 640, 480, j) == "South"


def test_conflict_gap_uses_tracks():
    j = load_junction("junction_01")
    uturn = {
        "id": 1,
        "approach": "North",
        "is_vehicle": True,
    }
    peers = [
        {
            "id": 2,
            "approach": "South",
            "is_vehicle": True,
            "currently_stopped": False,
            "last_speed": 40.0,
            "mean_recent_speed": 40.0,
            "distance_proxy_m": 100.0,
        }
    ]
    gap, source = estimate_conflict_gap_sec(uturn, peers, j)
    assert source == "conflict_track_eta"
    assert gap > 0


def test_gtfs_stops_available():
    summary = geo_service.geo_summary()
    assert summary["stops_available"] > 1000
    stops = geo_service.list_stops(limit=5)
    assert len(stops) == 5
    assert "lat" in stops[0]


def test_route_425_node_map():
    route = geo_service.route_node_map("425")
    assert len(route["stops"]) >= 10
    assert route["stops"][0]["lat"]
