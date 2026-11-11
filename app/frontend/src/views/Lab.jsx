import { useEffect, useState } from "react";
import { client } from "../api";

export default function Lab() {
  const [module, setModule] = useState("module1");
  const [models, setModels] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    client
      .get(`/api/arena/${module}`)
      .then((r) => {
        setModels(r.data.models || []);
        setError("");
      })
      .catch((e) => setError(e?.response?.data?.detail || e.message || "Arena load failed"))
      .finally(() => setLoading(false));
  }, [module]);

  return (
    <div className="ops-page">
      <div className="ops-stage__bar">
        <h2>Engineering lab · Colab arena</h2>
        <span className="ops-pill">Not used for live ops decisions</span>
      </div>
      <p className="ops-map-meta">
        These F1 scores come from offline feature tables with known leakage / rule-derived labels.
        Live junction decisions use <code>rules_v1</code> on YOLO+ByteTrack tracks. See{" "}
        <code>docs/ML_VS_DL_ACCURACY.md</code>.
      </p>
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
      {loading && <div className="ops-loading">Loading arena…</div>}
      {error && <div className="ops-error">{error}</div>}
      <table className="ops-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Category</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.model}>
              <td>{m.model}</td>
              <td>{m.category}</td>
              <td>{m.accuracy?.toFixed?.(4) ?? m.accuracy}</td>
              <td>{m.precision?.toFixed?.(4) ?? m.precision}</td>
              <td>{m.recall?.toFixed?.(4) ?? m.recall}</td>
              <td>{m.f1?.toFixed?.(4) ?? m.f1}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
