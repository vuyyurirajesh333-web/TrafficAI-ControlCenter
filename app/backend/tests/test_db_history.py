"""Phase 4 — SQLite persistence."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import acknowledge_alert, init_db, list_alerts, list_analyses, save_analysis


def test_save_and_ack_alert(tmp_path, monkeypatch):
    import db as dbmod

    monkeypatch.setattr(dbmod, "DB_PATH", tmp_path / "test.sqlite3")
    init_db()
    result = {
        "filename": "x.mp4",
        "analysis_mode": "test",
        "decision_engine": "rules_v1",
        "n_tracks": 1,
        "alert_count": 1,
        "recommended_action": "BROADCAST",
        "alerts": [
            {
                "alert_id": "INC-BRK-1",
                "type": "UNEXPECTED_STOPPAGE",
                "severity": "HIGH",
                "message": "test",
                "status": "ACTIVE",
            }
        ],
        "junction": {"lat": 28.54, "lon": 77.24},
    }
    aid = save_analysis(result, junction_id="junction_01")
    assert aid >= 1
    analyses = list_analyses(limit=5)
    assert analyses[0]["id"] == aid
    alerts = list_alerts(status="ACTIVE")
    assert len(alerts) == 1
    assert acknowledge_alert(alerts[0]["id"]) is True
    assert list_alerts(status="ACTIVE") == []
