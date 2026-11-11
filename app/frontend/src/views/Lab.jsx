import { useEffect, useState } from "react";
import { client } from "../api";

function fmt(n) {
  if (n === null || n === undefined) return "—";
  if (typeof n === "number") return n.toFixed(4);
  return String(n);
}

export default function Lab() {
  const [module, setModule] = useState("module1");
  const [models, setModels] = useState([]);
  const [meta, setMeta] = useState({});
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    client
      .get(`/api/arena/${module}`)
      .then((r) => {
        setModels(r.data.models || []);
        setMeta(r.data || {});
        setError("");
      })
      .catch((e) => setError(e?.response?.data?.detail || e.message || "Lab load failed"))
      .finally(() => setLoading(false));
  }, [module]);

  return (
    <div className="ops-page">
      <div className="ops-stage__bar">
        <h2>Model Lab · Comparison table</h2>
        <span className="ops-pill">validation metrics</span>
      </div>
      <p className="ops-map-meta">
        Model comparison on Accuracy, Precision, Recall, F1, and ROC-AUC. Live Monitor shows selected
        model name, decision, and <code>decision_source</code> only.
      </p>
      {meta.deployment_winner && (
        <p className="ops-map-meta">
          Default live model: <strong>{meta.deployment_winner}</strong>
        </p>
      )}
      <div className="ops-samples">
        {["module1", "module2", "module3"].map((m) => (
          <button
            key={m}
            type="button"
            className={`ops-chip ${module === m ? "ops-chip--active" : ""}`}
            onClick={() => setModule(m)}
          >
            {m}
          </button>
        ))}
      </div>
      {loading && <div className="ops-loading">Loading lab table…</div>}
      {error && <div className="ops-error">{error}</div>}
      <table className="ops-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>ROC-AUC</th>
            <th>Live UI</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m, idx) => (
            <tr key={m.model} className={m.highlighted ? "ops-row-hot" : undefined}>
              <td>{idx + 1}</td>
              <td>
                {m.model}
                {m.highlighted ? " ★" : ""}
              </td>
              <td>{fmt(m.accuracy)}</td>
              <td>{fmt(m.precision)}</td>
              <td>{fmt(m.recall)}</td>
              <td>{fmt(m.f1)}</td>
              <td>{fmt(m.roc_auc)}</td>
              <td>
                {m.selectable
                  ? m.artifact_status === "missing_on_disk"
                    ? "listed · artifact missing"
                    : "selectable"
                  : "hidden from selectors"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
