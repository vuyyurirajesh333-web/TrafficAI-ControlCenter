import { useEffect, useState } from "react";
import { client } from "../api";

export default function History() {
  const [rows, setRows] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    client
      .get("/api/history/analyses?limit=30")
      .then((r) => {
        setRows(r.data.analyses || []);
        setError("");
      })
      .catch((e) => setError(e?.response?.data?.detail || e.message || "Failed to load history"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="ops-page">
      <div className="ops-stage__bar">
        <h2>Analysis history</h2>
        <span className="ops-pill">SQLite · data/ops/ops.sqlite3</span>
      </div>
      {loading && <div className="ops-loading">Loading…</div>}
      {error && <div className="ops-error">{error}</div>}
      {!loading && !rows.length && !error && (
        <div className="ops-empty">No saved analyses yet.</div>
      )}
      <table className="ops-table">
        <thead>
          <tr>
            <th>ID</th>
            <th>When (UTC)</th>
            <th>File</th>
            <th>Tracks</th>
            <th>Alerts</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id}>
              <td>{r.id}</td>
              <td>{r.created_at}</td>
              <td>{r.filename}</td>
              <td>{r.n_tracks}</td>
              <td>{r.alert_count}</td>
              <td>{r.recommended_action}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
