import { useCallback, useEffect, useState } from "react";
import { client } from "../api";

export default function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("ACTIVE");

  const load = useCallback(() => {
    setLoading(true);
    client
      .get(`/api/history/alerts`, { params: { status, limit: 50 } })
      .then((r) => {
        setAlerts(r.data.alerts || []);
        setError("");
      })
      .catch((e) => setError(e?.response?.data?.detail || e.message || "Failed to load alerts"))
      .finally(() => setLoading(false));
  }, [status]);

  useEffect(() => {
    load();
  }, [load]);

  async function ack(id) {
    try {
      await client.post("/api/history/alerts/ack", { alert_row_id: id });
      load();
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Ack failed");
    }
  }

  return (
    <div className="ops-page">
      <div className="ops-stage__bar">
        <h2>Alert inbox</h2>
        <div className="ops-alert-filters">
          <button type="button" className={status === "ACTIVE" ? "ops-chip ops-chip--active" : "ops-chip"} onClick={() => setStatus("ACTIVE")}>
            Active
          </button>
          <button type="button" className={status === "all" ? "ops-chip ops-chip--active" : "ops-chip"} onClick={() => setStatus("all")}>
            All
          </button>
          <button type="button" className="ops-btn ops-btn--ghost" onClick={load}>
            Refresh
          </button>
        </div>
      </div>
      {loading && <div className="ops-loading">Loading persisted alerts…</div>}
      {error && <div className="ops-error">{error}</div>}
      {!loading && !alerts.length && !error && (
        <div className="ops-empty">No alerts stored yet. Run a junction analysis that triggers Module 2.</div>
      )}
      <div className="ops-alert-list">
        {alerts.map((a) => (
          <article key={a.id} className="ops-alert-card">
            <header>
              <strong>
                {a.severity} · {a.type}
              </strong>
              <span>{a.status}</span>
            </header>
            <p>{a.message}</p>
            <small>
              #{a.id} · analysis {a.analysis_id} · {a.created_at}
              {a.lat != null ? ` · ${a.lat}, ${a.lon}` : ""}
            </small>
            {a.status === "ACTIVE" && (
              <button type="button" className="ops-btn ops-btn--primary" onClick={() => ack(a.id)}>
                Acknowledge (persisted)
              </button>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}
