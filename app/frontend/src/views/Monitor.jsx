import { useEffect, useMemo, useState } from "react";
import { API, client, mediaUrl } from "../api";

function fmt(n) {
  if (n === null || n === undefined) return "—";
  if (typeof n === "number") return Number.isInteger(n) ? String(n) : n.toFixed(2);
  return String(n);
}

function ModuleCard({ title, code, active, statusLabel, rows, accent }) {
  return (
    <section className={`ops-card ${active ? `ops-card--${accent}` : "ops-card--idle"}`}>
      <header className="ops-card__head">
        <div>
          <p className="ops-card__code">{code}</p>
          <h3>{title}</h3>
        </div>
        <span className={`ops-badge ${active ? "ops-badge--on" : "ops-badge--off"}`}>
          {statusLabel}
        </span>
      </header>
      <dl className="ops-kv">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v ?? "—"}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export default function Monitor() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [samples, setSamples] = useState([]);
  const [samplesError, setSamplesError] = useState("");
  const [activeSample, setActiveSample] = useState(null);
  const [ackAlerts, setAckAlerts] = useState(() => new Set());
  const [stride, setStride] = useState(1);
  const [maxFrames, setMaxFrames] = useState(400);
  const [modelModule1, setModelModule1] = useState("");
  const [modelModule2, setModelModule2] = useState("");
  const [modelModule3, setModelModule3] = useState("");
  const [modelOptions, setModelOptions] = useState({ module1: [], module2: [], module3: [] });
  const [allowRulesFallback, setAllowRulesFallback] = useState(false);

  useEffect(() => {
    client
      .get("/api/samples")
      .then((r) => {
        setSamples(r.data.samples || []);
        setSamplesError("");
      })
      .catch(() => {
        setSamples([]);
        setSamplesError("Could not load sample list from API");
      });

    Promise.all(
      ["module1", "module2", "module3"].map((m) =>
        client.get(`/api/models/${m}`).then((r) => [m, r.data]).catch(() => [m, null])
      )
    ).then((rows) => {
      const next = { module1: [], module2: [], module3: [] };
      const winners = { module1: "", module2: "", module3: "" };
      for (const [m, data] of rows) {
        if (!data) continue;
        const winnerKey = data.winner?.model_key || "";
        winners[m] = winnerKey;
        const models = [...(data.models || [])].sort((a, b) => {
          if (a.model_key === winnerKey) return -1;
          if (b.model_key === winnerKey) return 1;
          return String(a.model_key).localeCompare(String(b.model_key));
        });
        next[m] = models;
      }
      setModelOptions(next);
      setModelModule1((v) => v || winners.module1);
      setModelModule2((v) => v || winners.module2);
      setModelModule3((v) => v || winners.module3);
    });
  }, []);

  const alerts = result?.alerts || [];
  const openAlerts = alerts.filter((a) => !ackAlerts.has(a.alert_id));
  const m1 = result?.module1_bus;
  const m2 = result?.module2_breakdown;
  const m3 = result?.module3_uturn;
  const counts = result?.vehicle_counts || {};
  const density = result?.approach_density || {};

  const shieldChecks = useMemo(() => {
    const incident = m2?.status === "INCIDENT DETECTED";
    const busReq = m1?.priority_status === "REQUESTED";
    return [
      {
        name: "Incident override",
        ok: !(incident && busReq && !String(result?.recommended_action || "").includes("DEFER")),
        note: "Breakdown outranks bus green extension",
      },
      {
        name: "U-turn gap gate",
        ok: !(m3?.recommended_action?.startsWith("SCHEDULE") && (m3?.gap_sec || 0) < 3),
        note: "Protected U-turn blocked if gap < 3s",
      },
      {
        name: "Fixed-camera assumption",
        ok: result?.camera_mode === "stationary_cctv" || !result,
        note: "Alerts target vehicles, never the CCTV mount",
      },
    ];
  }, [m1, m2, m3, result]);

  async function analyze(selected = file) {
    if (!selected) return;
    setLoading(true);
    setError("");
    setActiveSample(null);
    setAckAlerts(new Set());
    try {
      const form = new FormData();
      form.append("file", selected);
      const qs = new URLSearchParams({
        max_frames: String(maxFrames),
        stride: String(stride),
        module_mode: "all",
        allow_rules_fallback: String(allowRulesFallback),
      });
      if (modelModule1) qs.set("model_module1", modelModule1);
      if (modelModule2) qs.set("model_module2", modelModule2);
      if (modelModule3) qs.set("model_module3", modelModule3);
      const { data } = await client.post(`/api/video/analyze?${qs.toString()}`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(data);
    } catch (e) {
      const detail = e?.response?.data?.detail;
      setError(
        typeof detail === "object"
          ? detail.message || JSON.stringify(detail)
          : detail || e.message || "Analyze failed"
      );
    } finally {
      setLoading(false);
    }
  }

  async function analyzeSample(filename) {
    if (!filename || loading) return;
    setLoading(true);
    setError("");
    setFile(null);
    setActiveSample(filename);
    setAckAlerts(new Set());
    try {
      const { data } = await client.post("/api/video/analyze-sample", {
        filename,
        max_frames: maxFrames,
        stride,
        module_mode: "all",
        model_module1: modelModule1 || null,
        model_module2: modelModule2 || null,
        model_module3: modelModule3 || null,
        allow_rules_fallback: allowRulesFallback,
      });
      setResult(data);
    } catch (e) {
      const detail = e?.response?.data?.detail;
      setError(
        typeof detail === "object"
          ? detail.message || JSON.stringify(detail)
          : detail || e.message || "Sample analyze failed"
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="ops-page">
      {openAlerts.map((a) => (
        <div className="ops-alert" key={a.alert_id} role="alert">
          <div className="ops-alert__main">
            <p className="ops-alert__type">
              {a.severity} · {a.type.replaceAll("_", " ")} · {a.alert_id}
            </p>
            <p className="ops-alert__msg">{a.message}</p>
          </div>
          <button type="button" className="ops-btn ops-btn--ghost" onClick={() => setAckAlerts((p) => new Set([...p, a.alert_id]))}>
            Acknowledge (session)
          </button>
        </div>
      ))}

      {error && <div className="ops-error">{error}</div>}
      {(result?.fallback_warning || allowRulesFallback) && (
        <div className="ops-alert" role="status">
          <div className="ops-alert__main">
            <p className="ops-alert__type">WARNING · Rules fallback</p>
            <p className="ops-alert__msg">
              {result?.fallback_warning
                ? "At least one module used rules_fallback instead of trained-model inference."
                : "Rules fallback is enabled. Turn it off for model-only decisions."}
            </p>
          </div>
        </div>
      )}

      <div className="ops-grid">
        <section className="ops-stage">
          <div className="ops-stage__bar">
            <h2>Junction CCTV · Annotated feed</h2>
            <span className="ops-pill">
              {result?.perception_engine || "yolov8n_bytetrack"} · {result?.decision_engine || "model_inference_v1"}
            </span>
          </div>

          <div className="ops-stage__body">
            <div className="ops-upload">
              <label className="ops-file">
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => {
                    setFile(e.target.files?.[0] || null);
                    setActiveSample(null);
                  }}
                />
                <span>{file ? file.name : "Select corridor CCTV clip"}</span>
              </label>
              <label className="ops-tune">
                max frames
                <input type="number" min={30} max={2000} value={maxFrames} onChange={(e) => setMaxFrames(Number(e.target.value) || 400)} />
              </label>
              <label className="ops-tune">
                stride
                <input type="number" min={1} max={10} value={stride} onChange={(e) => setStride(Number(e.target.value) || 1)} />
              </label>
              <label className="ops-tune">
                M1 model
                <select value={modelModule1} onChange={(e) => setModelModule1(e.target.value)}>
                  {(modelOptions.module1.length ? modelOptions.module1 : [{ model_key: "GAT_GRU" }]).map((m) => (
                    <option key={m.model_key} value={m.model_key} disabled={m.exists === false}>
                      {m.model_key}{m.model_key === "GAT_GRU" ? " ★" : ""}{m.exists === false ? " (missing)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label className="ops-tune">
                M2 model
                <select value={modelModule2} onChange={(e) => setModelModule2(e.target.value)}>
                  {(modelOptions.module2.length ? modelOptions.module2 : [{ model_key: "GAT_GRU" }]).map((m) => (
                    <option key={m.model_key} value={m.model_key} disabled={m.exists === false}>
                      {m.model_key}{m.model_key === "GAT_GRU" ? " ★" : ""}{m.exists === false ? " (missing)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label className="ops-tune">
                M3 model
                <select value={modelModule3} onChange={(e) => setModelModule3(e.target.value)}>
                  {(modelOptions.module3.length ? modelOptions.module3 : [{ model_key: "GAT_GRU" }]).map((m) => (
                    <option key={m.model_key} value={m.model_key} disabled={m.exists === false}>
                      {m.model_key}{m.model_key === "GAT_GRU" ? " ★" : ""}{m.exists === false ? " (missing)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label className="ops-tune">
                <input
                  type="checkbox"
                  checked={allowRulesFallback}
                  onChange={(e) => setAllowRulesFallback(e.target.checked)}
                />
                allow rules fallback
              </label>
              <button type="button" className="ops-btn ops-btn--primary" disabled={!file || loading} onClick={() => analyze()}>
                {loading ? "Processing perception…" : "Run junction analysis"}
              </button>
            </div>

            {(!!samples.length || samplesError) && (
              <div className="ops-samples">
                <span>Demo clips · click to analyze</span>
                {samplesError && <em className="ops-samples__err">{samplesError}</em>}
                {samples
                  .filter((s) => s.demo_chip)
                  .filter((s, i, arr) => arr.findIndex((x) => x.label === s.label) === i)
                  .map((s) => (
                  <button
                    key={s.filename}
                    type="button"
                    className={`ops-chip ${activeSample === s.filename ? "ops-chip--active" : ""}`}
                    disabled={loading}
                    onClick={() => analyzeSample(s.filename)}
                  >
                    {s.label || s.filename}
                  </button>
                ))}
              </div>
            )}

            <div className="ops-video">
              {result?.annotated_video_url ? (
                <video key={result.annotated_video_url} src={mediaUrl(result.annotated_video_url)} controls autoPlay muted />
              ) : (
                <div className="ops-video__empty">
                  <strong>Awaiting CCTV upload</strong>
                  <p>Upload or click a sample. Boxes and track IDs render here after analysis.</p>
                </div>
              )}
            </div>

            <div className="ops-stats">
              <div><em>Tracks</em><b>{result?.n_tracks ?? 0}</b></div>
              <div><em>Frames</em><b>{result?.frames_processed ?? "—"}</b></div>
              <div><em>Cars</em><b>{counts.car ?? 0}</b></div>
              <div><em>Buses</em><b>{counts.bus ?? 0}</b></div>
              <div><em>Density</em><b>{density.total ?? 0}</b></div>
              <div><em>Alerts</em><b className={result?.alert_count ? "hot" : ""}>{result?.alert_count ?? 0}</b></div>
              <div><em>Analysis #</em><b>{result?.analysis_id ?? "—"}</b></div>
            </div>

            {result?.tracks_preview?.length > 0 && (
              <div className="ops-tracks">
                <h3>Tracks preview</h3>
                <table>
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Class</th>
                      <th>Approach</th>
                      <th>Dist m</th>
                      <th>U-turn</th>
                      <th>Stop?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.tracks_preview.slice(0, 12).map((t) => (
                      <tr key={t.id}>
                        <td>{t.id}</td>
                        <td>{t.cls_name}</td>
                        <td>{t.approach}</td>
                        <td>{fmt(t.distance_proxy_m)}</td>
                        <td>{fmt(t.uturn_score)}</td>
                        <td>{t.sudden_stop_candidate ? "yes" : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>

        <aside className="ops-rail">
          <div className="ops-stage__bar">
            <h2>Control decision</h2>
          </div>
          <div className="ops-action">
            <span>Recommended · {result?.decision_engine || "model_inference_v1"}</span>
            <strong>{result?.recommended_action || "Waiting for video"}</strong>
          </div>
          <div className="ops-signal">
            <span>Signal state · simulated (no TSC link)</span>
            <strong>{result?.signal_phase || "—"}</strong>
            <em>{result?.signal_mode || "idle"}</em>
          </div>
          <div className="ops-density">
            <h3>Approach density</h3>
            {["North", "South", "East", "West"].map((a) => (
              <div key={a} className="ops-density__row">
                <span>{a}</span>
                <b>{density[a] ?? 0}</b>
              </div>
            ))}
          </div>
          <ul className="ops-module-list">
            <li className={m1?.active ? "is-bus" : ""}>
              <span>M1 Bus priority</span>
              <b>{m1?.priority_status || "INACTIVE"}</b>
              <small>{m1?.recommended_action || "No bus in view"}</small>
            </li>
            <li className={m2?.status === "INCIDENT DETECTED" ? "is-alert" : ""}>
              <span>M2 Incident alert</span>
              <b>{m2?.status || "NORMAL"}</b>
              <small>{m2?.action || "Monitoring stoppages"}</small>
            </li>
            <li className={m3?.active ? "is-turn" : ""}>
              <span>M3 U-turn demand</span>
              <b>{m3?.demand_level || "Low"}</b>
              <small>{m3?.recommended_action || "Standard phase plan"}</small>
            </li>
          </ul>
          <div className="ops-shield">
            <h3>Safety gates</h3>
            {shieldChecks.map((c) => (
              <div key={c.name} className="ops-shield__row">
                <div>
                  <strong>{c.name}</strong>
                  <p>{c.note}</p>
                </div>
                <span className={c.ok ? "pass" : "fail"}>{c.ok ? "PASS" : "BLOCK"}</span>
              </div>
            ))}
          </div>
          {result?.junction && (
            <div className="ops-junction">
              <h3>Junction config</h3>
              <p>{result.junction.name}</p>
              <small>
                {result.junction.lat}, {result.junction.lon} · mpp={result.junction.meters_per_pixel}
              </small>
            </div>
          )}
        </aside>
      </div>

      <div className="ops-modules">
        <ModuleCard
          title="Public transport priority"
          code="Module 1"
          accent="bus"
          active={!!m1?.active}
          statusLabel={m1?.priority_status || "INACTIVE"}
          rows={[
            ["Selected model", m1?.selected_model],
            ["Decision", m1?.decision || m1?.priority_status],
            ["Source", m1?.decision_source],
            ["Alert / action", m1?.recommended_action],
          ]}
        />
        <ModuleCard
          title="Breakdown / stoppage alert"
          code="Module 2"
          accent="alert"
          active={m2?.status === "INCIDENT DETECTED"}
          statusLabel={m2?.status || "NORMAL"}
          rows={[
            ["Selected model", m2?.selected_model],
            ["Decision", m2?.decision || m2?.status],
            ["Source", m2?.decision_source],
            ["Alert / action", m2?.action || m2?.control_room],
          ]}
        />
        <ModuleCard
          title="U-turn / turn demand"
          code="Module 3"
          accent="turn"
          active={!!m3?.active}
          statusLabel={m3?.demand_level || "Low"}
          rows={[
            ["Selected model", m3?.selected_model],
            ["Decision", m3?.decision],
            ["Source", m3?.decision_source],
            ["Alert / action", m3?.recommended_action],
          ]}
        />
      </div>
      <p className="ops-api-hint">API base: {API}</p>
    </div>
  );
}
