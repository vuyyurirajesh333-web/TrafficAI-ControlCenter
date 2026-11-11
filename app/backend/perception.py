from pathlib import Path
import math
import shutil
import subprocess

import cv2
from ultralytics import YOLO


BUS, CAR, TRUCK, MOTORCYCLE = 5, 2, 7, 3
VEHICLE_CLASSES = {BUS, CAR, TRUCK, MOTORCYCLE}
_model = None

# Perception thresholds tuned for fixed CCTV (camera itself is always stationary).
# Alerts fire only on VEHICLE behaviour — never on “scene looks still”.
STATIONARY_SPEED_PX = 12.0         # px/sec below this counts toward stop (allows tracker jitter)
STOPPED_RECENT_PX = 18.0           # mean recent speed to call "currently stopped"
MOVING_SPEED_PX = 28.0             # must have moved at this speed earlier
MIN_TRACK_FRAMES = 20
MIN_STOP_FRAMES = 25               # ~1s+ at 25fps
MIN_STOP_RATIO = 0.40
MIN_PRIOR_MOTION_FRAMES = 5
SCENE_MOVING_PEER_MIN = 1          # at least one other vehicle still moving
# U-turn: use net heading change (~π) + path tortuosity, not raw jitter sum
UTURN_NET_MED = 1.8                # ~103°
UTURN_NET_HIGH = 2.6               # ~149°
UTURN_MIN_PATH = 40.0              # px of travel before judging a turn


# --------------------------------------------------
# YOLO MODEL
# --------------------------------------------------

def get_model():
    global _model

    if _model is None:
        _model = YOLO("yolov8n.pt")

    return _model


# --------------------------------------------------
# H.264 BROWSER-COMPATIBLE VIDEO CONVERSION
# --------------------------------------------------

def _resolve_ffmpeg() -> str | None:
    """Prefer PATH / local bin ffmpeg; then imageio-ffmpeg's bundled binary."""
    found = shutil.which("ffmpeg")
    if found:
        return found

    local = Path(__file__).resolve().parent / "bin" / "ffmpeg.exe"
    if local.exists():
        return str(local)

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _make_browser_mp4_ffmpeg(src: Path, dst: Path, ffmpeg: str) -> None:
    cmd = [
        ffmpeg,
        "-y",
        "-i", str(src),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        err = (result.stderr or result.stdout or "").strip()[-800:]
        raise RuntimeError(f"ffmpeg H.264 conversion failed: {err or 'unknown error'}")


def _ensure_openh264() -> None:
    """Best-effort fetch of Cisco OpenH264 DLL expected by OpenCV on Windows."""
    import bz2
    import urllib.request

    dll_name = "openh264-2.5.0-win64.dll"
    search_dirs = [
        Path(cv2.__file__).resolve().parent,
        Path(__file__).resolve().parent / "bin",
        Path(__file__).resolve().parent,
    ]
    if any((d / dll_name).exists() for d in search_dirs):
        return

    url = f"http://ciscobinary.openh264.org/{dll_name}.bz2"
    target_dir = search_dirs[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    bz_path = target_dir / f"{dll_name}.bz2"
    try:
        urllib.request.urlretrieve(url, bz_path)
        (target_dir / dll_name).write_bytes(bz2.decompress(bz_path.read_bytes()))
    finally:
        if bz_path.exists():
            bz_path.unlink(missing_ok=True)


def _make_browser_mp4_opencv(src: Path, dst: Path) -> None:
    """Re-encode with OpenH264 (avc1) when system ffmpeg is unavailable."""
    _ensure_openh264()
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open intermediate video: {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    width -= width % 2
    height -= height % 2
    if width < 2 or height < 2:
        cap.release()
        raise RuntimeError("Invalid video dimensions for H.264 encode")

    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not open H.264 VideoWriter (avc1). "
            "Install ffmpeg or place openh264-2.5.0-win64.dll next to OpenCV."
        )

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        frames += 1

    writer.release()
    cap.release()

    if frames == 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError("OpenCV H.264 conversion produced no output")


def make_browser_mp4(src_path: str) -> str:
    """Re-encode OpenCV mp4v output to browser-playable H.264 MP4."""
    src = Path(src_path)
    dst = src.with_name(src.stem + "_web.mp4")

    ffmpeg = _resolve_ffmpeg()
    if ffmpeg:
        _make_browser_mp4_ffmpeg(src, dst, ffmpeg)
    else:
        _make_browser_mp4_opencv(src, dst)

    return str(dst)


# --------------------------------------------------
# SPEED / MOTION HELPERS
# --------------------------------------------------

def _speed(positions):
    if len(positions) < 2:
        return 0.0

    (x0, y0, t0), (x1, y1, t1) = positions[-2], positions[-1]
    dt = max(t1 - t0, 1e-3)
    return math.hypot(x1 - x0, y1 - y0) / dt


def _mean_speed(positions, last_n=8):
    if len(positions) < 2:
        return 0.0
    pts = positions[-last_n:] if len(positions) >= last_n else positions
    speeds = []
    for i in range(1, len(pts)):
        x0, y0, t0 = pts[i - 1]
        x1, y1, t1 = pts[i]
        dt = max(t1 - t0, 1e-3)
        speeds.append(math.hypot(x1 - x0, y1 - y0) / dt)
    return sum(speeds) / max(len(speeds), 1)


def _class_label(cid, names):
    if isinstance(names, dict):
        return names.get(cid, str(cid))
    try:
        return names[cid]
    except Exception:
        return str(cid)


# --------------------------------------------------
# TRACK SUMMARIZATION
# --------------------------------------------------

def _heading_metrics(positions):
    """Net / peak heading change and path stats — robust to tracker jitter."""
    steps = []
    for i in range(1, len(positions)):
        x0, y0, _ = positions[i - 1]
        x1, y1, _ = positions[i]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 2.0:
            continue
        steps.append((math.atan2(dy, dx), dist))

    if len(steps) < 3:
        return {
            "turn_amount": 0.0,
            "net_heading": 0.0,
            "peak_heading": 0.0,
            "path_length": 0.0,
            "displacement": 0.0,
            "tortuosity": 1.0,
            "uturn_score": 0.0,
        }

    headings = [h for h, _ in steps]
    path_length = sum(d for _, d in steps)
    x0, y0, _ = positions[0]
    x1, y1, _ = positions[-1]
    displacement = math.hypot(x1 - x0, y1 - y0)

    # unwrap cumulative signed heading change
    net = 0.0
    peak = 0.0
    for i in range(1, len(headings)):
        d = (headings[i] - headings[i - 1] + math.pi) % (2 * math.pi) - math.pi
        # ignore tiny jitter flips
        if abs(d) < 0.12:
            continue
        net += d
        peak = max(peak, abs(net))

    tortuosity = path_length / max(displacement, 1.0)
    abs_net = abs(net)
    # U-turn-like: need real heading reverse — tortuosity alone is not enough (tracker jitter)
    uturn_score = 0.0
    if path_length >= UTURN_MIN_PATH and (abs_net >= 1.0 or peak >= 1.3):
        reverse_term = min(abs_net, math.pi) / math.pi
        curl_term = min(peak, math.pi) / math.pi
        tort_term = min(max(tortuosity - 1.5, 0) / 2.5, 1.0)
        return_term = 1.0 - min(displacement / max(path_length, 1.0), 1.0)
        uturn_score = 0.50 * reverse_term + 0.30 * curl_term + 0.10 * tort_term + 0.10 * return_term
        if abs_net < 1.4 and peak < 1.6:
            uturn_score *= 0.55

    return {
        "turn_amount": float(abs_net),
        "net_heading": float(net),
        "peak_heading": float(peak),
        "path_length": float(path_length),
        "displacement": float(displacement),
        "tortuosity": float(tortuosity),
        "uturn_score": float(uturn_score),
    }


def _summarize(tracks, w, h, fps, junction=None):
    from junction_config import (
        assign_approach,
        in_uturn_pocket,
        load_junction,
        meters_per_pixel,
    )

    if junction is None:
        try:
            junction = load_junction()
        except Exception:
            junction = {"meters_per_pixel": 0.05, "approaches": {}}

    mpp = meters_per_pixel(junction)
    summaries = []

    for tid, tr in tracks.items():
        pos = tr["positions"]

        if len(pos) < 3:
            continue

        hm = _heading_metrics(pos)

        x, y = tr.get("last_xy", (0, 0))
        approach = assign_approach(x, y, w, h, junction)
        pocket = in_uturn_pocket(x, y, w, h, junction)

        frames = tr["frames"]
        stationary_frames = tr["stationary_frames"]
        moving_frames = tr["moving_frames"]
        max_speed = float(tr.get("max_speed", 0.0))
        last_speed = float(tr.get("last_speed", 0.0))
        mean_recent = _mean_speed(pos, last_n=12)

        # Suddenly-stopped = had prior motion, then stayed slow
        was_moving = moving_frames >= MIN_PRIOR_MOTION_FRAMES or max_speed >= MOVING_SPEED_PX
        currently_stopped = (
            mean_recent < STOPPED_RECENT_PX
            or (mean_recent < STOPPED_RECENT_PX * 1.25 and last_speed < STOPPED_RECENT_PX)
        )
        long_enough = (
            frames >= MIN_TRACK_FRAMES
            and stationary_frames >= MIN_STOP_FRAMES
            and stationary_frames / max(frames, 1) >= MIN_STOP_RATIO
        )

        summaries.append({
            "id": tid,
            "cls": tr["cls"],
            "cls_name": tr.get("cls_name", "vehicle"),
            "is_bus": tr["cls"] == BUS,
            "is_vehicle": tr["cls"] in VEHICLE_CLASSES,
            "conf": tr["conf"],
            "frames": frames,
            "stationary_frames": stationary_frames,
            "moving_frames": moving_frames,
            "stationary_ratio": stationary_frames / max(frames, 1),
            "max_speed": max_speed,
            "last_speed": last_speed,
            "mean_recent_speed": float(mean_recent),
            "was_moving": bool(was_moving),
            "currently_stopped": bool(currently_stopped),
            "sudden_stop_candidate": bool(was_moving and currently_stopped and long_enough),
            "turn_amount": hm["turn_amount"],
            "net_heading": hm["net_heading"],
            "peak_heading": hm["peak_heading"],
            "uturn_score": hm["uturn_score"],
            "tortuosity": hm["tortuosity"],
            "approach": approach,
            "in_uturn_pocket": bool(pocket),
            "last_xy": [float(x), float(y)],
            "last_t": float(tr.get("last_t", 0.0)),
            "distance_proxy_m": float(max(h - y, 0) * mpp),
            "distance_scale_mpp": float(mpp),
            "duration_sec": float(frames / max(fps, 1.0)),
            "stationary_duration_sec": float(stationary_frames / max(fps, 1.0)),
        })

    return summaries


def _scene_has_moving_peers(summaries, exclude_id=None):
    """True if other vehicles are still moving — distinguishes red-light stop vs breakdown."""
    movers = [
        t for t in summaries
        if t["is_vehicle"]
        and t["id"] != exclude_id
        and (
            t["last_speed"] >= MOVING_SPEED_PX * 0.6
            or t["mean_recent_speed"] >= MOVING_SPEED_PX * 0.5
            or (t["moving_frames"] / max(t["frames"], 1)) > 0.35
        )
    ]
    return len(movers) >= SCENE_MOVING_PEER_MIN, len(movers)


# --------------------------------------------------
# VIDEO ANALYSIS + ANNOTATION
# --------------------------------------------------

def analyze_and_annotate(
    video_path: str,
    out_path: str,
    max_frames: int = 400,
    stride: int = 1,
    junction_id: str | None = None,
):
    from junction_config import load_junction

    try:
        junction = load_junction(junction_id or "junction_01")
    except Exception:
        junction = {"id": "fallback", "meters_per_pixel": 0.05, "approaches": {}}

    model = get_model()
    names = model.names

    tracks = {}
    frame_idx = 0
    alert_ids = set()

    w = h = None
    fps = 25.0
    writer = None

    # Probe FPS from file
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        probed = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if probed > 1:
            fps = probed
        cap.release()

    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        conf=0.35,
        imgsz=640,
        verbose=False,
        classes=list(VEHICLE_CLASSES),
    )

    for r in results:
        frame_idx += 1

        if frame_idx % stride != 0:
            continue

        if frame_idx > max_frames * stride:
            break

        frame = r.orig_img.copy()
        h, w = frame.shape[:2]

        if writer is None:
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError("Could not create annotated video.")

        t = frame_idx / fps

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()

            ids = (
                r.boxes.id.cpu().numpy().astype(int)
                if r.boxes.id is not None
                else [None] * len(xyxy)
            )

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                cid = int(cls[i])
                tid = ids[i]
                class_name = _class_label(cid, names)

                label = f"{class_name} {conf[i]:.2f}"
                color = (40, 180, 90)

                if cid == BUS:
                    color = (0, 140, 255)
                    label = "BUS " + label

                if tid is not None:
                    cxy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    tr = tracks.setdefault(
                        int(tid),
                        {
                            "id": int(tid),
                            "cls": cid,
                            "cls_name": class_name,
                            "conf": float(conf[i]),
                            "positions": [],
                            "frames": 0,
                            "stationary_frames": 0,
                            "moving_frames": 0,
                            "max_speed": 0.0,
                        }
                    )

                    tr["cls"] = cid
                    tr["cls_name"] = class_name
                    tr["conf"] = max(tr["conf"], float(conf[i]))
                    tr["positions"].append((cxy[0], cxy[1], t))
                    tr["frames"] += 1

                    sp = _speed(tr["positions"])
                    tr["last_speed"] = sp
                    tr["max_speed"] = max(tr["max_speed"], sp)
                    tr["last_xy"] = cxy
                    tr["last_t"] = t

                    if sp < STATIONARY_SPEED_PX:
                        tr["stationary_frames"] += 1
                    if sp >= MOVING_SPEED_PX:
                        tr["moving_frames"] += 1

                    label = f"ID:{int(tid)} {label}"

                    # Live overlay only after sudden-stop evidence (prior motion + stop)
                    frames_n = tr["frames"]
                    stop_n = tr["stationary_frames"]
                    recent = _mean_speed(tr["positions"], last_n=12)
                    if (
                        tr["moving_frames"] >= MIN_PRIOR_MOTION_FRAMES
                        and frames_n >= MIN_TRACK_FRAMES
                        and stop_n >= MIN_STOP_FRAMES
                        and stop_n / max(frames_n, 1) >= MIN_STOP_RATIO
                        and recent < STOPPED_RECENT_PX
                    ):
                        color = (0, 0, 255)
                        label = "STOP? " + label
                        alert_ids.add(int(tid))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(22, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        status = (
            f"TrafficAI Ops | Fixed CCTV | tracks={len(tracks)} | "
            f"stop-watch={len(alert_ids)}"
        )
        cv2.putText(
            frame,
            status,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "ALERT = vehicle moved then stopped (not camera / red-light)",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 220, 255),
            1,
        )

        writer.write(frame)

    if writer is not None:
        writer.release()

    if writer is None:
        raise RuntimeError(
            "No video frames were processed. Cannot create annotated video."
        )

    summaries = _summarize(tracks, w or 1, h or 1, fps, junction=junction)
    web_path = make_browser_mp4(out_path)

    from junction_config import approach_density

    vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0, "other": 0}
    for t in summaries:
        name = str(t.get("cls_name", "")).lower()
        if "bus" in name:
            vehicle_counts["bus"] += 1
        elif "truck" in name:
            vehicle_counts["truck"] += 1
        elif "motor" in name or "bike" in name:
            vehicle_counts["motorcycle"] += 1
        elif "car" in name:
            vehicle_counts["car"] += 1
        else:
            vehicle_counts["other"] += 1

    density = approach_density(summaries)

    return {
        "tracks": summaries,
        "n_tracks": len(summaries),
        "frames_processed": frame_idx,
        "fps": fps,
        "annotated_video": web_path,
        "vehicle_counts": vehicle_counts,
        "approach_density": density,
        "junction_id": junction.get("id"),
        "camera_mode": "stationary_cctv",
        "frame_size": {"w": w, "h": h},
    }


# --------------------------------------------------
# DECISION / ARENA METADATA
# Live ops use track heuristics (rules_v1). Colab arena names are
# engineering reference only and never drive decisions.
# --------------------------------------------------

DECISION_ENGINE = "rules_v1"
PERCEPTION_ENGINE = "yolov8n_bytetrack"


def pick_best_model_name(arena_models):
    if not arena_models:
        return "N/A", None

    best = max(
        arena_models,
        key=lambda m: (
            m.get("f1") is not None,
            m.get("f1", -1)
        )
    )
    return best.get("model", "N/A"), best


def _arena_reference_fields(best_name, meta):
    """Compat + honesty fields attached to every module payload."""
    return {
        "decision_engine": DECISION_ENGINE,
        # retained for API compatibility — Colab leaderboard winner name only
        "selected_model": best_name,
        "selected_model_f1": None if not meta else meta.get("f1"),
        "arena_model_drives_decision": False,
        "decision_note": (
            "Ops decision uses track heuristics (rules_v1). "
            "selected_model is an offline Colab arena reference, not the live decision model."
        ),
    }


# --------------------------------------------------
# BUILD MODULE OUTPUTS
# --------------------------------------------------

def build_module_outputs(perception, arena1, arena2, arena3, junction=None):
    from junction_config import estimate_conflict_gap_sec, load_junction

    tracks = perception["tracks"]

    buses = [t for t in tracks if t["is_bus"]]
    vehicles = [t for t in tracks if t["is_vehicle"]]

    if junction is None:
        try:
            junction = load_junction(perception.get("junction_id") or "junction_01")
        except Exception:
            junction = {"approaches": {}, "meters_per_pixel": 0.05}

    best1, meta1 = pick_best_model_name(arena1)
    best2, meta2 = pick_best_model_name(arena2)
    best3, meta3 = pick_best_model_name(arena3)

    active = []
    alerts = []

    # ---------- MODULE 1: BUS PRIORITY ----------
    if buses:
        active.append("module1_bus")
        b = sorted(buses, key=lambda x: x["distance_proxy_m"])[0]
        # Prefer real motion; fall back to distance-only when bus is slow in frame
        speed_proxy = max(b["last_speed"] * 0.05, b.get("mean_recent_speed", 0) * 0.05, 0.15)
        eta = b["distance_proxy_m"] / speed_proxy
        grant = b["distance_proxy_m"] < 140 and eta < 60

        module1 = {
            "active": True,
            "reason": "Public transport bus detected and tracked approaching junction",
            **_arena_reference_fields(best1, meta1),
            "bus_id": f"BUS-{b['id']}",
            "approach": b["approach"],
            "distance_m": round(b["distance_proxy_m"], 1),
            "eta_sec": round(eta, 1),
            "priority_status": "REQUESTED" if grant else "MONITORING",
            "recommended_action": "EXTEND GREEN (+12s)" if grant else "HOLD — bus not yet in priority window",
            "confidence": round(float(b["conf"]), 3),
            "signal_request": {
                "from_phase": "current",
                "to_phase": f"{b['approach']}-THROUGH GREEN",
                "duration_sec": 12 if grant else 0,
                "accepted": bool(grant),
            },
        }
    else:
        module1 = {
            "active": False,
            "reason": "No bus detected in this clip",
            **_arena_reference_fields(best1, meta1),
            "priority_status": "INACTIVE",
            "recommended_action": "No bus priority request",
            "signal_request": None,
        }

    # ---------- MODULE 2: BREAKDOWN + ALERT ----------
    # Fixed CCTV is always stationary — never treat “low scene motion” as an incident.
    # Require: prior motion → stop, enough duration, and peers still moving.
    sudden = [t for t in vehicles if t.get("sudden_stop_candidate")]

    incident_track = None
    peer_movers = 0
    for cand in sorted(sudden, key=lambda x: x["stationary_ratio"], reverse=True):
        has_peers, peer_movers = _scene_has_moving_peers(vehicles, exclude_id=cand["id"])
        # If almost every vehicle is stopped, treat as red light / congestion — suppress
        stopped_ratio = sum(1 for v in vehicles if v["currently_stopped"]) / max(len(vehicles), 1)
        if stopped_ratio >= 0.75 and not has_peers:
            continue
        # Require a meaningful stop duration before paging control room
        if cand["stationary_duration_sec"] < 2.5 and not (
            has_peers and cand["stationary_duration_sec"] >= 2.0
        ):
            continue
        if has_peers or cand["stationary_duration_sec"] >= 4.0:
            incident_track = cand
            break

    if incident_track is not None:
        active.append("module2_breakdown")
        c = incident_track
        # Severity from stop duration + isolation from moving peers
        dur = c["stationary_duration_sec"]
        base = 0.55 + min(0.35, dur / 20.0)
        if peer_movers >= 2:
            base += 0.08
        prob = min(0.97, base)
        severity = "HIGH" if prob >= 0.78 or dur >= 6 else "MEDIUM"
        incident = True

        alert = {
            "alert_id": f"INC-BRK-{c['id']}",
            "type": "UNEXPECTED_STOPPAGE",
            "severity": severity,
            "vehicle_id": f"VEH-{c['id']}",
            "vehicle_class": c.get("cls_name", "vehicle"),
            "location": f"{c['approach']} Approach",
            "stoppage_sec": round(dur, 1),
            "message": (
                f"{c.get('cls_name', 'Vehicle').title()} VEH-{c['id']} moved then stopped "
                f"unexpectedly on {c['approach']} approach "
                f"({dur:.1f}s stationary while traffic still flowing)."
            ),
            "channels": ["Control Room", "Patrol Dispatch", "V2I Slow-Down Broadcast"],
            "status": "ACTIVE",
            "recommended_response": (
                f"Dispatch patrol/tow to {c['approach']} approach. "
                "Broadcast slow-down advisory. Reassess signal plan for blocked approach."
            ),
        }
        alerts.append(alert)

        module2 = {
            "active": True,
            "reason": "Sudden stop after prior motion with surrounding traffic still moving",
            **_arena_reference_fields(best2, meta2),
            "status": "INCIDENT DETECTED",
            "vehicle_id": f"VEH-{c['id']}",
            "vehicle_class": c.get("cls_name", "vehicle"),
            "location": f"{c['approach']} Approach",
            "stationary_duration_sec": round(dur, 1),
            "prior_max_speed": round(c["max_speed"], 1),
            "moving_peers": peer_movers,
            "incident_probability": round(prob, 4),
            "incident_probability_source": "heuristic_formula",
            "action": "Traffic Incident Alert",
            "control_room": alert["recommended_response"],
            "alert": alert,
            "suppressed_false_positives": {
                "camera_stationary": "ignored (fixed CCTV assumed)",
                "all_stopped_as_red_light": True,
            },
        }
    else:
        # Soft monitoring candidates (shown but no alert)
        soft = [t for t in vehicles if t.get("currently_stopped") and t.get("was_moving")]
        module2 = {
            "active": False,
            "reason": (
                "No confirmed unexpected stoppage "
                "(need prior motion → stop, duration, and/or moving peers)"
            ),
            **_arena_reference_fields(best2, meta2),
            "status": "NORMAL",
            "action": "Monitoring — no alert",
            "incident_probability": 0.0,
            "incident_probability_source": "heuristic_formula",
            "watchlist_count": len(soft),
            "alert": None,
            "suppressed_false_positives": {
                "camera_stationary": "ignored (fixed CCTV assumed)",
                "all_stopped_as_red_light": True,
            },
        }

    # ---------- MODULE 3: SMART U-TURN ----------
    # Score blends net reverse heading + tortuosity (not raw angle jitter).
    scored = sorted(
        [t for t in vehicles if t.get("uturn_score", 0) >= 0.42],
        key=lambda x: x.get("uturn_score", 0),
        reverse=True,
    )
    # Also accept large net heading if path was long enough
    for t in vehicles:
        if (
            abs(t.get("net_heading", 0)) >= UTURN_NET_MED
            and t.get("frames", 0) >= 20
            and t not in scored
        ):
            scored.append(t)
    scored = sorted(scored, key=lambda x: x.get("uturn_score", 0), reverse=True)

    if scored:
        active.append("module3_uturn")
        u = scored[0]
        # Prefer U-turn pocket tracks when available
        pocket_hits = [t for t in scored if t.get("in_uturn_pocket")]
        if pocket_hits:
            u = pocket_hits[0]
        high = (
            u.get("uturn_score", 0) >= 0.58
            or abs(u.get("net_heading", 0)) >= UTURN_NET_HIGH
            or bool(u.get("in_uturn_pocket") and u.get("uturn_score", 0) >= 0.42)
        )
        queue_est = max(1, len(scored))
        gap, gap_source = estimate_conflict_gap_sec(u, vehicles, junction)
        schedule = high and gap >= 3.0

        module3 = {
            "active": True,
            "reason": "Trajectory geometry consistent with U-turn / protected-turn demand",
            **_arena_reference_fields(best3, meta3),
            "demand_level": "High" if high else "Medium",
            "uturn_count": queue_est,
            "queue_estimate": queue_est,
            "gap_sec": gap,
            "gap_sec_source": gap_source,
            "in_uturn_pocket": bool(u.get("in_uturn_pocket")),
            "recommended_action": (
                "SCHEDULE PROTECTED U-TURN (+8s)" if schedule else "HOLD — gap / demand not ready"
            ),
            "track_id": f"VEH-{u['id']}",
            "turn_intensity": round(u.get("uturn_score", u.get("turn_amount", 0)), 3),
            "net_heading_rad": round(float(u.get("net_heading", 0)), 3),
        }
    else:
        module3 = {
            "active": False,
            "reason": "No strong U-turn geometry in tracked trajectories",
            **_arena_reference_fields(best3, meta3),
            "recommended_action": "Maintain standard phase plan",
            "demand_level": "Low",
            "uturn_count": 0,
            "queue_estimate": 0,
        }

    # ---------- UNIFIED DECISION (breakdown outranks bus priority) ----------
    actions = []
    if module2.get("status") == "INCIDENT DETECTED":
        actions.append("BROADCAST BREAKDOWN ALERT")
        if module1.get("priority_status") == "REQUESTED":
            actions.append("DEFER BUS PRIORITY (incident override)")
        else:
            actions.append("HOLD SIGNAL — clear incident approach")
    else:
        if module1.get("priority_status") == "REQUESTED":
            actions.append(f"EXTEND GREEN for {module1['bus_id']}")
        if (
            module3.get("active")
            and str(module3.get("recommended_action", "")).startswith("SCHEDULE")
        ):
            actions.append("SCHEDULE PROTECTED U-TURN")

    if not actions:
        actions.append("MAINTAIN CURRENT SIGNAL PLAN")

    return module1, module2, module3, actions, active, alerts
