import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { client } from "./api";
import Monitor from "./views/Monitor";
import MapView from "./views/MapView";
import Alerts from "./views/Alerts";
import History from "./views/History";
import Lab from "./views/Lab";
import "./index.css";

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState("");
  const [clock, setClock] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    client
      .get("/api/health")
      .then((r) => {
        setHealth(r.data);
        setHealthError("");
      })
      .catch((e) => {
        setHealth(null);
        setHealthError(e.message || "API offline");
      });
  }, []);

  const clockStr = clock.toLocaleString("en-IN", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div className="ops">
      <header className="ops-top">
        <div className="ops-brand">
          <div className="ops-crest" aria-hidden />
          <div>
            <p className="ops-kicker">Municipal Traffic Management · Junction Operations</p>
            <h1>Traffic AI Control Center</h1>
          </div>
        </div>
        <div className="ops-top-meta">
          <div className="ops-meta-block">
            <span>Local time</span>
            <strong>{clockStr}</strong>
          </div>
          <div className="ops-meta-block">
            <span>API</span>
            <strong className={health?.status === "ok" ? "ok" : "bad"}>
              {health?.status === "ok" ? "ONLINE" : healthError ? "OFFLINE" : "…"}
            </strong>
          </div>
          <div className="ops-meta-block">
            <span>Engine</span>
            <strong>{health?.decision_engine || "model_inference_v1"}</strong>
          </div>
        </div>
      </header>

      <nav className="ops-nav">
        <NavLink to="/" end>
          Monitor
        </NavLink>
        <NavLink to="/map">Delhi map</NavLink>
        <NavLink to="/alerts">Alerts</NavLink>
        <NavLink to="/history">History</NavLink>
        <NavLink to="/lab">Lab</NavLink>
      </nav>

      <Routes>
        <Route path="/" element={<Monitor />} />
        <Route path="/map" element={<MapView />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/history" element={<History />} />
        <Route path="/lab" element={<Lab />} />
      </Routes>

      <footer className="ops-foot">
        <p>
          Live decisions: <code>{health?.decision_engine || "model_inference_v1"}</code> on{" "}
          <code>{health?.perception_engine || "yolov8n_bytetrack"}</code>. Lab shows model comparison metrics.
        </p>
        <p>GTFS stops available: {health?.geo?.stops_available ?? "—"} · Camera: fixed CCTV</p>
      </footer>
    </div>
  );
}
