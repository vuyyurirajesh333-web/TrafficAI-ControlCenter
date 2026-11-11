"""P1.3 — analyze routes offload blocking work via asyncio.to_thread."""

from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as api


@pytest.mark.asyncio
async def test_analyze_sample_uses_thread_offload():
    samples = api.list_samples()["samples"]
    assert samples

    fake = {
        "filename": samples[0]["filename"],
        "decision_engine": "rules_v1",
        "analysis_mode": "sample_on_disk",
        "n_tracks": 0,
        "frames_processed": 0,
        "vehicle_counts": {},
        "annotated_video_url": "/outputs/x.mp4",
        "active_modules": [],
        "alerts": [],
        "alert_count": 0,
        "signal_phase": "PHASE_1 · NS THROUGH",
        "signal_mode": "normal",
        "signal_phase_simulated": True,
        "recommended_action": "MAINTAIN CURRENT SIGNAL PLAN",
        "module1_bus": {},
        "module2_breakdown": {},
        "module3_uturn": {},
        "ops_summary": {},
        "tracks_preview": [],
        "camera_mode": "stationary_cctv",
        "perception_engine": "yolov8n_bytetrack",
        "fps": 25,
    }

    with patch("main.run_junction_analysis", return_value=fake) as mocked:
        transport = ASGITransport(app=api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(
                "/api/video/analyze-sample",
                json={"filename": samples[0]["filename"]},
            )
        assert r.status_code == 200
        assert r.json()["analysis_mode"] == "sample_on_disk"
        mocked.assert_called_once()


@pytest.mark.asyncio
async def test_health_reports_thread_executor():
    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["analyze_executor"] == "asyncio.to_thread"
    assert "http://127.0.0.1:5173" in body["cors_allow_origins"]
