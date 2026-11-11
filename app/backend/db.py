"""SQLite persistence for analysis runs and alerts (Phase 4)."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "ops" / "ops.sqlite3"

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _lock:
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    filename TEXT,
                    analysis_mode TEXT,
                    decision_engine TEXT,
                    n_tracks INTEGER,
                    alert_count INTEGER,
                    recommended_action TEXT,
                    junction_id TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    analysis_id INTEGER,
                    alert_id TEXT NOT NULL,
                    type TEXT,
                    severity TEXT,
                    message TEXT,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    acknowledged_at TEXT,
                    lat REAL,
                    lon REAL,
                    payload_json TEXT,
                    UNIQUE(alert_id, analysis_id),
                    FOREIGN KEY(analysis_id) REFERENCES analyses(id)
                );
                """
            )
            conn.commit()
        finally:
            conn.close()


def save_analysis(result: dict[str, Any], junction_id: str | None = None) -> int:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO analyses (
                    created_at, filename, analysis_mode, decision_engine,
                    n_tracks, alert_count, recommended_action, junction_id, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    result.get("filename"),
                    result.get("analysis_mode"),
                    result.get("decision_engine"),
                    result.get("n_tracks"),
                    result.get("alert_count"),
                    result.get("recommended_action"),
                    junction_id,
                    json.dumps(result),
                ),
            )
            analysis_id = int(cur.lastrowid)
            lat = (result.get("junction") or {}).get("lat")
            lon = (result.get("junction") or {}).get("lon")
            for alert in result.get("alerts") or []:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO alerts (
                        created_at, analysis_id, alert_id, type, severity,
                        message, status, lat, lon, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now,
                        analysis_id,
                        alert.get("alert_id"),
                        alert.get("type"),
                        alert.get("severity"),
                        alert.get("message"),
                        alert.get("status") or "ACTIVE",
                        lat,
                        lon,
                        json.dumps(alert),
                    ),
                )
            conn.commit()
            return analysis_id
        finally:
            conn.close()


def list_analyses(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT id, created_at, filename, analysis_mode, decision_engine,
                       n_tracks, alert_count, recommended_action, junction_id
                FROM analyses
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_analysis(analysis_id: int) -> dict[str, Any] | None:
    init_db()
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            data["payload"] = json.loads(data.pop("payload_json"))
            return data
        finally:
            conn.close()


def list_alerts(status: str | None = "ACTIVE", limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with _lock:
        conn = _connect()
        try:
            if status:
                rows = conn.execute(
                    """
                    SELECT id, created_at, analysis_id, alert_id, type, severity,
                           message, status, acknowledged_at, lat, lon
                    FROM alerts
                    WHERE status = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, created_at, analysis_id, alert_id, type, severity,
                           message, status, acknowledged_at, lat, lon
                    FROM alerts
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def acknowledge_alert(alert_row_id: int) -> bool:
    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                """
                UPDATE alerts
                SET status = 'ACKNOWLEDGED', acknowledged_at = ?
                WHERE id = ? AND status = 'ACTIVE'
                """,
                (now, alert_row_id),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
