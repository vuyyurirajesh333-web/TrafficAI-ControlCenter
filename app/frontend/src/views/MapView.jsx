import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { client } from "../api";

export default function MapView() {
  const mapRef = useRef(null);
  const mapEl = useRef(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [meta, setMeta] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      setLoading(true);
      setError("");
      try {
        const [juncRes, route425, route534, hotspots] = await Promise.all([
          client.get("/api/junctions/junction_01"),
          client.get("/api/geo/routes/425"),
          client.get("/api/geo/routes/534"),
          client.get("/api/geo/hotspots?limit_cells=30"),
        ]);
        if (cancelled) return;

        const j = juncRes.data;
        setMeta({
          junction: j.name,
          stops425: route425.data.stops?.length || 0,
          stops534: route534.data.stops?.length || 0,
          hotspotNote: hotspots.data.note,
        });

        if (!mapEl.current) return;
        if (mapRef.current) {
          mapRef.current.remove();
          mapRef.current = null;
        }

        const map = L.map(mapEl.current).setView([j.lat, j.lon], 13);
        mapRef.current = map;
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
          attribution: "&copy; OpenStreetMap",
          maxZoom: 19,
        }).addTo(map);

        L.circleMarker([j.lat, j.lon], {
          radius: 10,
          color: "#b42318",
          fillColor: "#ef4444",
          fillOpacity: 0.9,
          weight: 2,
        })
          .addTo(map)
          .bindPopup(`<strong>${j.name}</strong><br/>${j.id}<br/>Config junction (real lat/lon)`);

        const addRoute = (payload, color) => {
          const pts = (payload.stops || [])
            .filter((s) => s.lat && s.lon)
            .map((s) => [s.lat, s.lon]);
          if (pts.length < 2) return;
          L.polyline(pts, { color, weight: 4, opacity: 0.85 }).addTo(map);
          pts.forEach((p, i) => {
            L.circleMarker(p, { radius: 4, color, fillOpacity: 0.8 })
              .addTo(map)
              .bindPopup(`${payload.route_code} · ${payload.stops[i]?.stop_name || ""}`);
          });
        };
        addRoute(route425.data, "#0b6e4f");
        addRoute(route534.data, "#1d4ed8");

        (hotspots.data.cells || []).forEach((c) => {
          const r = Math.min(18, 4 + Math.log10(c.count + 1) * 4);
          L.circleMarker([c.lat, c.lon], {
            radius: r,
            color: "#b45309",
            fillColor: "#f59e0b",
            fillOpacity: 0.35,
            weight: 1,
          })
            .addTo(map)
            .bindPopup(`Historical GPS density · count=${c.count}`);
        });
      } catch (e) {
        if (!cancelled) setError(e?.response?.data?.detail || e.message || "Map load failed");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    boot();
    return () => {
      cancelled = true;
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, []);

  return (
    <div className="ops-page">
      <div className="ops-stage__bar">
        <h2>Delhi network map</h2>
        <span className="ops-pill">GTFS routes 425 / 534 · historical GPS only</span>
      </div>
      {loading && <div className="ops-loading">Loading map layers from API…</div>}
      {error && <div className="ops-error">{error}</div>}
      {meta && (
        <p className="ops-map-meta">
          Junction: {meta.junction} · Route 425 stops: {meta.stops425} · Route 534 stops: {meta.stops534}
          <br />
          {meta.hotspotNote}
        </p>
      )}
      <div ref={mapEl} className="ops-map" />
    </div>
  );
}
