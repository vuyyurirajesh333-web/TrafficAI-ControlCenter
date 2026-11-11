"""
Export ByteTrack-style track feature rows from sample videos for Module-2 retrain.

Usage (from repo root, backend venv active):
  python scripts/export_track_features.py

Writes: data/exports/track_features_module2.csv
Does NOT invent labels — sudden_stop_candidate is the rule label for baseline comparison.
Independent human labels should replace y before claiming production metrics.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

from perception import analyze_and_annotate  # noqa: E402

SAMPLES = ROOT / "data" / "sample_videos"
OUT = ROOT / "data" / "exports"
OUT.mkdir(parents=True, exist_ok=True)
TMP = OUT / "_tmp_annot"
TMP.mkdir(exist_ok=True)


FEATURE_COLS = [
    "video",
    "track_id",
    "cls_name",
    "frames",
    "stationary_ratio",
    "max_speed",
    "last_speed",
    "mean_recent_speed",
    "stationary_duration_sec",
    "moving_frames",
    "was_moving",
    "currently_stopped",
    "sudden_stop_candidate",
    "approach",
    "distance_proxy_m",
    "uturn_score",
]


def main():
    rows = []
    videos = sorted(SAMPLES.glob("*.mp4"))
    if not videos:
        raise SystemExit("No sample videos found")

    for video in videos:
        if video.stat().st_size < 10_000:
            continue
        out_path = str(TMP / f"annot_{video.stem}.mp4")
        print(f"Processing {video.name} …")
        result = analyze_and_annotate(
            str(video),
            out_path,
            max_frames=200,
            stride=2,
            junction_id="junction_01",
        )
        for t in result["tracks"]:
            rows.append({
                "video": video.name,
                "track_id": t["id"],
                "cls_name": t.get("cls_name"),
                "frames": t.get("frames"),
                "stationary_ratio": t.get("stationary_ratio"),
                "max_speed": t.get("max_speed"),
                "last_speed": t.get("last_speed"),
                "mean_recent_speed": t.get("mean_recent_speed"),
                "stationary_duration_sec": t.get("stationary_duration_sec"),
                "moving_frames": t.get("moving_frames"),
                "was_moving": int(bool(t.get("was_moving"))),
                "currently_stopped": int(bool(t.get("currently_stopped"))),
                "sudden_stop_candidate": int(bool(t.get("sudden_stop_candidate"))),
                "approach": t.get("approach"),
                "distance_proxy_m": t.get("distance_proxy_m"),
                "uturn_score": t.get("uturn_score"),
            })

    out_csv = OUT / "track_features_module2.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {out_csv}")


if __name__ == "__main__":
    main()
