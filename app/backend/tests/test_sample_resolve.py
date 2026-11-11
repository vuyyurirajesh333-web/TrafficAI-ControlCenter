"""P1.4 — sample path resolution (no YOLO)."""

from pathlib import Path
import sys

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as api


def test_resolve_known_sample():
    samples = api.list_samples()["samples"]
    assert samples, "expected files in data/sample_videos/"
    path = api.resolve_sample_path(samples[0]["filename"])
    assert path.is_file()
    assert path.parent == api.SAMPLES.resolve()


def test_reject_path_traversal():
    with pytest.raises(HTTPException) as exc:
        api.resolve_sample_path("../README.md")
    assert exc.value.status_code == 400


def test_reject_missing_sample():
    with pytest.raises(HTTPException) as exc:
        api.resolve_sample_path("does_not_exist_xyz.mp4")
    assert exc.value.status_code == 404


def test_reject_non_video():
    with pytest.raises(HTTPException) as exc:
        api.resolve_sample_path("README.md")
    assert exc.value.status_code == 400
