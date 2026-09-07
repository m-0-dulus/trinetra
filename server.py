import argparse
import asyncio
import base64
import json
import math
import socket
import ssl
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web, WSMsgType

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from ai.target_manager import TargetManager

ROOT = Path(__file__).resolve().parent
CLIENTS = {}
FRAMES = {"JET-01": None, "JET-02": None}
STATES = {"JET-01": {}, "JET-02": {}}
PREVIEWS = {"JET-01": None, "JET-02": None}
MANAGERS = {"JET-01": TargetManager(), "JET-02": TargetManager()}
MODEL = None
CONF = 0.20
LOCK_TIME = 0.85
TARGET_ZONE = 0.22
FIRE_COOLDOWN = 1.5
OCCLUSION_MAX = 1.5
OCCLUSION_GRACE = 0.45
DETECT_LOST_TIMEOUT = 2.5
frame_counter = 0
PREVIEW_BYTES = {"JET-01": None, "JET-02": None}

GAME = {
    "round": 1,
    "status": "WAITING",
    "winner": None,
    "last_hit": None,
    "health": {"JET-01": 1, "JET-02": 1},
    "fire_cooldown": {"JET-01": 0.0, "JET-02": 0.0},
}

# Track IDs are ours, so the two camera streams cannot contaminate each other.
NEXT_IDS = {"JET-01": 1, "JET-02": 1}
TRACKS = {"JET-01": {}, "JET-02": {}}


def decode_frame(data):
    if not data:
        return None
    if data.startswith("data:image"):
        data = data.split(",", 1)[1]
    try:
        arr = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def hsv_hist(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 25, 25]), np.array([179, 255, 255]))
    hist = cv2.calcHist([hsv], [0, 1], mask, [8, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(float).tolist()


def skin_fraction(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 25, 45]), np.array([25, 210, 255]))
    return float(np.count_nonzero(mask)) / float(mask.size)


def contour_shape(frame, contour):
    area_px = cv2.contourArea(contour)
    if area_px <= 0:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    long_side = max(w, h)
    short_side = max(1, min(w, h))
    aspect = long_side / short_side
    rect_area = max(1.0, w * h)
    extent = area_px / rect_area
    hull = cv2.convexHull(contour)
    hull_area = max(1.0, cv2.contourArea(hull))
    solidity = area_px / hull_area
    peri = max(1.0, cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, 0.035 * peri, True)

    # Estimate whether the silhouette gets wider around the middle than at its
    # ends. That is a useful cheap proxy for wings + fuselage.
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) >= 5:
        mean, eig = cv2.PCACompute(pts, mean=None)[:2]
        center = mean[0]
        axis = eig[0]
        perp = np.array([-axis[1], axis[0]], dtype=np.float32)
        longitudinal = (pts - center) @ axis
        lateral = (pts - center) @ perp
        lo, hi = float(longitudinal.min()), float(longitudinal.max())
        span = max(1e-3, hi - lo)
        bins = []
        for a, b in ((0.0, .22), (.22, .45), (.45, .55), (.55, .78), (.78, 1.0)):
            sel = (longitudinal >= lo + a * span) & (longitudinal <= lo + b * span)
            bins.append(float(np.ptp(lateral[sel])) if np.any(sel) else 0.0)
        ends = (bins[0] + bins[-1]) / 2.0
        middle = max(bins[1], bins[2], bins[3])
        wing_score = min(1.0, max(0.0, (middle / max(1.0, ends) - 1.0) / 1.5))
    else:
        wing_score = 0.0

    # Jet-like score: elongated silhouette, reasonable solidity, and a wider
    # middle. This is a heuristic fallback for cardboard/foam jets when a
    # pretrained COCO model does not recognize the custom object.
    elong = min(1.0, max(0.0, (aspect - 1.15) / 2.6))
    solidity_score = 1.0 - min(1.0, abs(solidity - 0.76) / 0.35)
    extent_score = 1.0 - min(1.0, abs(extent - 0.38) / 0.38)
    vertex_score = 1.0 if 4 <= len(approx) <= 14 else 0.25
    jet_score = 0.38 * elong + 0.25 * wing_score + 0.18 * solidity_score + 0.12 * extent_score + 0.07 * vertex_score

    return {
        "aspect": float(aspect),
        "solidity": float(solidity),
        "extent": float(extent),
        "wing_score": float(wing_score),
        "vertices": int(len(approx)),
        "jet_score": float(jet_score),
        "xyxy": [float(x), float(y), float(x + w), float(y + h)],
    }


def detect_shape_jets(frame):
    """Robust tabletop fighter detector.

    The physical demo uses cardboard/foam aircraft, which stock YOLO11n does
    not know as a custom class. We therefore use a hybrid CV detector:
      1) cardboard/tan color segmentation to isolate the aircraft,
      2) contour geometry to reject compact faces/hands/background blobs,
      3) edge contours as a secondary fallback.

    The important change from V3 is that wide-wing fighters are accepted; the
    old detector assumed the aircraft had to be taller than it was wide, which
    rejected the actual tabletop silhouette in the demo.
    """
    h, w = frame.shape[:2]
    scale = min(1.0, 700.0 / max(w, 1))
    img = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
    ih, iw = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    candidates = []

    # Cardboard/tan mask. Two ranges handle warm cardboard under cool/warm
    # room lighting. Morphology joins the aircraft body and wings.
    masks = [
        cv2.inRange(hsv, np.array([5, 45, 35]), np.array([38, 255, 255])),
        cv2.inRange(hsv, np.array([0, 30, 55]), np.array([28, 210, 255])),
    ]
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            shape = contour_shape(img, c)
            if not shape:
                continue
            x1, y1, x2, y2 = shape["xyxy"]
            bw, bh = x2 - x1, y2 - y1
            area = (bw * bh) / max(1.0, iw * ih)
            contour_area = cv2.contourArea(c) / max(1.0, iw * ih)
            # The actual jet is broad-winged, so either orientation is valid.
            # Compact face/hand-like blobs are rejected by area + geometry.
            if area < 0.015 or area > 0.72 or contour_area < 0.008:
                continue
            if min(bw, bh) < max(18, 0.035 * min(iw, ih)):
                continue
            broad_or_elongated = max(bw, bh) / max(1.0, min(bw, bh)) >= 1.18
            if not broad_or_elongated:
                continue
            skin = skin_fraction(img, shape["xyxy"])
            if skin > 0.55:
                continue
            # Wide fighters can have a low edge-defined 'wing score' but the
            # silhouette should still have decent solidity/extent.
            geometry = (
                0.40 * min(1.0, shape["jet_score"] / 0.50) +
                0.25 * min(1.0, shape["solidity"] / 0.72) +
                0.20 * min(1.0, shape["extent"] / 0.48) +
                0.15 * min(1.0, max(0.0, (max(bw, bh) / max(1.0, min(bw, bh)) - 1.0) / 1.8))
            )
            if geometry < 0.42:
                continue
            if scale < 1:
                inv = 1.0 / scale
                shape["xyxy"] = [v * inv for v in shape["xyxy"]]
            shape["conf"] = min(0.99, 0.45 + 0.50 * geometry)
            shape["jet_score"] = max(shape.get("jet_score", 0.0), geometry)
            shape["source"] = "cardboard-shape"
            shape["appearance"] = hsv_hist(frame, shape["xyxy"])
            candidates.append(shape)

    # Edge fallback: useful if the cardboard is grey/poorly lit.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 125)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        shape = contour_shape(img, c)
        if not shape:
            continue
        x1, y1, x2, y2 = shape["xyxy"]
        bw, bh = x2 - x1, y2 - y1
        area = (bw * bh) / max(1.0, iw * ih)
        if area < 0.02 or area > 0.60:
            continue
        if max(bw, bh) / max(1.0, min(bw, bh)) < 1.18:
            continue
        if shape["jet_score"] < 0.34:
            continue
        skin = skin_fraction(img, shape["xyxy"])
        if skin > 0.50:
            continue
        if scale < 1:
            inv = 1.0 / scale
            shape["xyxy"] = [v * inv for v in shape["xyxy"]]
        shape["conf"] = min(0.92, 0.38 + 0.60 * shape["jet_score"])
        shape["source"] = "edge-shape"
        shape["appearance"] = hsv_hist(frame, shape["xyxy"])
        candidates.append(shape)

    # Deduplicate overlapping masks/fallbacks.
    candidates.sort(key=lambda d: d.get("conf", 0) * d.get("jet_score", 0), reverse=True)
    out = []
    for d in candidates:
        if any(iou(d["xyxy"], q["xyxy"]) > 0.55 for q in out):
            continue
        out.append(d)
        if len(out) >= 8:
            break
    return out

def yolo_airplanes(frame):
    if MODEL is None:
        return []
    try:
        result = MODEL.predict(source=frame, conf=CONF, verbose=False, imgsz=640)[0]
        out = []
        if result.boxes is None:
            return out
        for b in result.boxes:
            cls = int(b.cls[0].cpu().item()) if b.cls is not None else -1
            # COCO class 4 = airplane. Only accept airplane from YOLO; person,
            # face-like, hand-like, car, etc. are intentionally ignored.
            if cls != 4:
                continue
            xy = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf[0].cpu().item())
            x1, y1, x2, y2 = xy
            ww, hh = max(1, x2 - x1), max(1, y2 - y1)
            out.append({
                "xyxy": xy,
                "conf": conf,
                "jet_score": .92,
                "aspect": max(ww, hh) / min(ww, hh),
                "solidity": .8,
                "extent": .45,
                "wing_score": .75,
                "vertices": 0,
                "source": "yolo-airplane",
                "appearance": hsv_hist(frame, xy),
            })
        return out
    except Exception as e:
        return []


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    bb = max(1, bx2 - bx1) * max(1, by2 - by1)
    return inter / max(1.0, aa + bb - inter)


def assign_track_ids(jet, detections):
    global NEXT_IDS
    tracks = TRACKS[jet]
    now = time.time()
    used = set()
    result = []

    # Greedy nearest/appearance matching. Good enough for the small number of
    # tabletop jets and, unlike a global YOLO tracker, isolated per camera.
    for d in sorted(detections, key=lambda x: x.get("conf", 0), reverse=True):
        best_id, best_score = None, -1.0
        for tid, tr in tracks.items():
            if tid in used:
                continue
            dist = math.hypot(d["cx"] - tr["cx"], d["cy"] - tr["cy"])
            if dist > 0.30:
                continue
            score = 0.65 * (1.0 - min(1.0, dist / .30)) + 0.35 * (1.0 - min(1.0, abs(d.get("jet_score", .5) - tr.get("jet_score", .5))))
            if score > best_score:
                best_id, best_score = tid, score
        if best_id is None:
            best_id = NEXT_IDS[jet]
            NEXT_IDS[jet] += 1
        used.add(best_id)
        d["id"] = best_id
        tracks[best_id] = {
            "cx": d["cx"], "cy": d["cy"], "xyxy": d["xyxy"],
            "jet_score": d.get("jet_score", .5), "appearance": d.get("appearance", []),
            "last": now,
        }
        result.append(d)

    for tid in list(tracks):
        if now - tracks[tid]["last"] > 2.0:
            tracks.pop(tid, None)
    return result


def annotate_frame(frame, state):
    out = frame.copy()
    h, w = out.shape[:2]
    # Other candidates: cyan boxes. Locked target: green. Predicted target:
    # amber dashed cross when temporarily occluded.
    for d in state.get("detections", []):
        x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
        color = (180, 220, 220)
        thickness = 1
        if state.get("target_id") == d.get("id"):
            color = (60, 240, 100) if state.get("locked") else (60, 220, 255)
            thickness = 3
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"JET CANDIDATE {d.get('id', '?')} {d.get('jet_score', 0):.2f}"
        cv2.putText(out, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)

    target = state.get("target")
    if target:
        cx = int(target["cx"] * w)
        cy = int(target["cy"] * h)
        cv2.circle(out, (cx, cy), 28, (60, 240, 100) if state.get("locked") else (60, 220, 255), 2)
        cv2.line(out, (cx - 40, cy), (cx + 40, cy), (60, 240, 100), 1)
        cv2.line(out, (cx, cy - 40), (cx, cy + 40), (60, 240, 100), 1)

    pred = state.get("prediction")
    if state.get("state") == "OCCLUDED" and pred:
        px, py = int(pred[0] * w), int(pred[1] * h)
        for a in range(0, 360, 45):
            r1, r2 = 32, 44
            a1, a2 = math.radians(a), math.radians(a + 22)
            cv2.line(out, (px + int(r1 * math.cos(a1)), py + int(r1 * math.sin(a1))),
                     (px + int(r2 * math.cos(a2)), py + int(r2 * math.sin(a2))), (40, 180, 255), 3)
        cv2.putText(out, "PREDICTED TARGET", (max(5, px - 65), max(15, py - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (40, 180, 255), 1, cv2.LINE_AA)

    status = state.get("state", "SEARCHING")
    if state.get("locked"):
        status = "TARGET LOCKED"
    cv2.putText(out, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .7,
                (60, 240, 100) if state.get("locked") else (220, 240, 245), 2, cv2.LINE_AA)
    return out


def ai_process(jet, frame):
    m = MANAGERS[jet]
    h, w = frame.shape[:2]

    # Hybrid detector: use YOLO's airplane class when it recognizes the jet,
    # then supplement it with our shape detector for custom cardboard jets.
    detections = yolo_airplanes(frame)
    if not detections:
        detections = detect_shape_jets(frame)
    else:
        # Keep a shape candidate only when it doesn't overlap a YOLO airplane.
        shape = detect_shape_jets(frame)
        for s in shape:
            if all(iou(s["xyxy"], y["xyxy"]) < .25 for y in detections):
                detections.append(s)

    # Normalize first so the tracker can work in 0..1 coordinates.
    norm = []
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        d["cx"] = ((x1 + x2) / 2) / max(1, w)
        d["cy"] = ((y1 + y2) / 2) / max(1, h)
        d["area"] = ((x2 - x1) * (y2 - y1)) / max(1, w * h)
        norm.append(d)

    norm = assign_track_ids(jet, norm)
    m.update(norm, w, h)
    return m.snapshot()


def compute_lock(jet, state):
    now = time.time()
    s = STATES[jet]
    t = state.get("target")
    inside = False
    if t:
        inside = abs(t["cx"] - .5) <= TARGET_ZONE and abs(t["cy"] - .5) <= TARGET_ZONE
    if state.get("state") == "OCCLUDED" and state.get("prediction"):
        px, py = state["prediction"]
        inside = abs(px - .5) <= TARGET_ZONE and abs(py - .5) <= TARGET_ZONE
    if inside:
        if not s.get("lock_started"):
            s["lock_started"] = now
    else:
        s["lock_started"] = 0

    p = (now - s["lock_started"]) / LOCK_TIME if s.get("lock_started") else 0
    locked = p >= 1.0
    if locked:
        MANAGERS[jet].remember_lock()
    return locked, min(1.0, p)


async def send_to(cid, msg):
    ws = CLIENTS.get(cid)
    if ws and not ws.closed:
        try:
            await ws.send_str(json.dumps(msg))
        except Exception:
            pass


async def broadcast(msg):
    payload = json.dumps(msg)
    for cid, ws in list(CLIENTS.items()):
        if ws.closed:
            CLIENTS.pop(cid, None)
            continue
        try:
            await ws.send_str(payload)
        except Exception:
            CLIENTS.pop(cid, None)


def reset_round(increment=True):
    if increment:
        GAME["round"] += 1
    GAME.update({
        "status": "WAITING", "winner": None, "last_hit": None,
        "health": {"JET-01": 1, "JET-02": 1},
        "fire_cooldown": {"JET-01": 0.0, "JET-02": 0.0},
    })
    for s in STATES.values():
        s.clear()
        s["lock_started"] = 0
    for m in MANAGERS.values():
        m.target_id = None
        m.remembered = None
        m.last_target = None
        m.designated_point = None
        m._state = "SEARCHING"
    for k in TRACKS:
        TRACKS[k].clear()
        NEXT_IDS[k] = 1


async def send_state():
    await broadcast({"type": "state", "game": GAME, "states": STATES})


async def fire(attacker):
    target = "JET-02" if attacker == "JET-01" else "JET-01"
    now = time.time()
    if GAME["status"] == "DEAD":
        return
    if CLIENTS.get("JET-01") is None or CLIENTS.get("JET-02") is None or FRAMES.get("JET-01") is None or FRAMES.get("JET-02") is None:
        await send_to(attacker, {"type": "event", "event": "not_ready", "reason": "BOTH_JETS_REQUIRED"})
        await send_state()
        return
    if now < GAME["fire_cooldown"][attacker]:
        await send_to(attacker, {"type": "event", "event": "cooldown", "remaining": GAME["fire_cooldown"][attacker] - now})
        return
    GAME["fire_cooldown"][attacker] = now + FIRE_COOLDOWN
    state = STATES.get(attacker, {})
    if not state.get("locked", False):
        await send_to(attacker, {"type": "event", "event": "miss", "reason": "NO_LOCK"})
        await send_state()
        return

    GAME["status"] = "DEAD"
    GAME["health"][target] = 0
    GAME["winner"] = attacker
    GAME["last_hit"] = {"attacker": attacker, "target": target, "time": now}
    await send_to(target, {"type": "event", "event": "hit", "attacker": attacker, "target": target})
    await send_to(attacker, {"type": "event", "event": "confirmed_hit", "target": target})
    await send_state()

    async def delayed_reset():
        await asyncio.sleep(4)
        reset_round(True)
        await send_state()

    asyncio.create_task(delayed_reset())


async def ai_loop():
    global frame_counter
    while True:
        await asyncio.sleep(.08)
        for jet, frame in list(FRAMES.items()):
            if frame is None:
                continue
            frame_counter += 1
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))
            # YOLO/OpenCV inference is CPU-heavy and MUST NOT run directly on
            # aiohttp's event loop. Doing so can starve the WebSocket reader,
            # which makes phones look like their camera/command link is frozen.
            state = await asyncio.to_thread(ai_process, jet, frame)
            locked, progress = compute_lock(jet, state)
            state["locked"] = locked
            state["lock_progress"] = progress
            state["fire_ready"] = locked
            state["occlusion_max"] = OCCLUSION_MAX
            STATES[jet] = state
            if frame_counter % 2 == 0:
                try:
                    preview = annotate_frame(frame, state)
                    preview = cv2.resize(preview, (320, 180))
                    ok, enc = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
                    if ok:
                        PREVIEW_BYTES[jet] = enc.tobytes()
                except Exception:
                    pass
        if frame_counter % 2 == 0:
            # The battle is ready only when both pilot links are live and both
            # cameras have supplied frames.
            if GAME["status"] != "DEAD":
                GAME["status"] = "ACTIVE" if all(CLIENTS.get(j) and FRAMES.get(j) is not None for j in ("JET-01", "JET-02")) else "WAITING"
            await send_state()


async def mjpeg_stream(request):
    """Tiny MJPEG stream for the command dashboard.

    Using a real HTTP multipart stream instead of base64 frames over the
    dashboard WebSocket makes the two live feeds reliable even while the AI
    loop is busy running YOLO/OpenCV.
    """
    jet = request.match_info["jet"]
    if jet not in PREVIEW_BYTES:
        raise web.HTTPNotFound()
    response = web.StreamResponse(headers={
        "Content-Type": "multipart/x-mixed-replace; boundary=frame",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
    })
    await response.prepare(request)
    try:
        while True:
            data = PREVIEW_BYTES.get(jet)
            if data:
                await response.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            await asyncio.sleep(0.10)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            await response.write_eof()
        except Exception:
            pass
    return response


async def index(_):
    raise web.HTTPFound("/dashboard")


async def file(request, folder):
    p = folder / request.match_info["path"]
    if not p.exists() or not p.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(p)


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=4 * 1024 * 1024, heartbeat=20)
    await ws.prepare(request)
    cid = None
    try:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                break
            if msg.type != WSMsgType.TEXT:
                continue
            try:
                d = json.loads(msg.data)
            except Exception:
                continue
            typ = d.get("type")
            if typ == "hello":
                requested = d.get("jet")
                if requested not in ("JET-01", "JET-02", "DASHBOARD"):
                    await ws.send_str(json.dumps({"type": "error", "message": "Choose JET-01 or JET-02"}))
                    continue
                cid = requested
                old = CLIENTS.get(cid)
                if old and old is not ws:
                    try:
                        await old.close(code=4001, message=b"replaced by newer connection")
                    except Exception:
                        pass
                CLIENTS[cid] = ws
                await send_to(cid, {"type": "welcome", "jet": cid})
                await send_state()
                print(f"[WS] {cid} connected from {request.remote}", flush=True)
            elif typ == "ping":
                await ws.send_str(json.dumps({"type": "pong", "time": time.time()}))
            elif typ == "frame" and cid in FRAMES:
                f = decode_frame(d.get("data", ""))
                if f is not None:
                    FRAMES[cid] = f
            elif typ == "designate" and cid in FRAMES:
                try:
                    MANAGERS[cid].designate(float(d.get("x", .5)), float(d.get("y", .5)))
                    await send_to(cid, {"type": "event", "event": "designated"})
                except Exception:
                    pass
            elif typ == "fire" and cid in FRAMES:
                await fire(cid)
            elif typ == "reset" and cid in ("JET-01", "JET-02", "DASHBOARD"):
                reset_round(True)
                await send_state()
    except Exception as e:
        print(f"[WS] {cid or 'unknown'} error: {e}", flush=True)
    finally:
        if cid and CLIENTS.get(cid) is ws:
            CLIENTS.pop(cid, None)
        if cid in FRAMES:
            FRAMES[cid] = None
        print(f"[WS] {cid or 'unknown'} disconnected", flush=True)
    return ws


def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ssl_context(ip=None):
    ip = ip or local_ip()
    ca_cert = ROOT / "trinetra-ca.pem"
    ca_key = ROOT / "trinetra-ca-key.pem"
    cert = ROOT / "cert.pem"
    key = ROOT / "key.pem"
    cfg = ROOT / ".trinetra-openssl.cnf"
    try:
        if not ca_cert.exists() or not ca_key.exists():
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(ca_key), "-out", str(ca_cert), "-days", "3650",
                "-subj", "/CN=Trinetra Local CA"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cfg.write_text(
            "[req]\n"
            "distinguished_name = req_distinguished_name\n"
            "req_extensions = req_ext\n"
            "prompt = no\n"
            "[req_distinguished_name]\n"
            "CN = Trinetra Server\n"
            "[req_ext]\n"
            "subjectAltName = @alt_names\n"
            "[alt_names]\n"
            f"IP.1 = {ip}\n"
            "IP.2 = 127.0.0.1\n"
            "DNS.1 = localhost\n"
        )
        csr = ROOT / ".trinetra-server.csr"
        subprocess.run([
            "openssl", "req", "-new", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(csr), "-config", str(cfg)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([
            "openssl", "x509", "-req", "-in", str(csr),
            "-CA", str(ca_cert), "-CAkey", str(ca_key), "-CAcreateserial",
            "-out", str(cert), "-days", "825", "-sha256",
            "-extfile", str(cfg), "-extensions", "req_ext"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(str(cert), str(key))
        return ctx
    except Exception as e:
        print("HTTPS certificate setup failed:", e, flush=True)
        return None
    finally:
        for tmp in (cfg, ROOT / ".trinetra-server.csr", ROOT / "cert.pem.srl"):
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


async def async_main():
    global MODEL, CONF
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=.20)
    a = ap.parse_args()
    CONF = a.conf

    if YOLO:
        try:
            print("Loading", a.model, flush=True)
            MODEL = YOLO(a.model)
            print("YOLO ready", flush=True)
        except Exception as e:
            print("WARNING: YOLO unavailable:", e, flush=True)
    else:
        print("WARNING: ultralytics import failed; install requirements.txt", flush=True)

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/health", lambda r: web.json_response({
        "ok": True, "service": "TRINETRA V3", "clients": list(CLIENTS.keys())
    }))
    app.router.add_get("/api/status", lambda r: web.json_response({
        "game": GAME, "states": STATES, "clients": list(CLIENTS.keys()),
        "frames": {k: FRAMES[k] is not None for k in FRAMES},
    }))
    app.router.add_get("/phone", lambda r: web.FileResponse(ROOT / "phone/index.html"))
    app.router.add_get("/dashboard", lambda r: web.FileResponse(ROOT / "dashboard/index.html"))
    app.router.add_get("/phone/{path:.*}", lambda r: file(r, ROOT / "phone"))
    app.router.add_get("/dashboard/{path:.*}", lambda r: file(r, ROOT / "dashboard"))
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/stream/{jet}", mjpeg_stream)

    runner = web.AppRunner(app)
    await runner.setup()
    tasks = []
    try:
        tasks.append(asyncio.create_task(ai_loop()))
        ip = local_ip()
        ctx = ssl_context(ip)
        if ctx is None:
            raise RuntimeError("Could not create/load TLS certificate")
        https_site = web.TCPSite(runner, "0.0.0.0", 8443, ssl_context=ctx)
        await https_site.start()
        http_site = web.TCPSite(runner, "0.0.0.0", 8080)
        await http_site.start()
        print("", flush=True)
        print("=" * 64, flush=True)
        print("TRINETRA V3 SERVER IS LISTENING", flush=True)
        print(f"PHONE HTTPS: https://{ip}:8443/phone", flush=True)
        print(f"DASH HTTPS:  https://{ip}:8443/dashboard", flush=True)
        print(f"DASH HTTP:   http://127.0.0.1:8080/dashboard", flush=True)
        print(f"HEALTH:      https://127.0.0.1:8443/health", flush=True)
        print(f"TRUST CA:    {ROOT / 'trinetra-ca.pem'}", flush=True)
        print("", flush=True)
        print("Tracking: YOLO-airplane + shape fallback + re-ID + occlusion prediction", flush=True)
        print("", flush=True)
        await asyncio.Event().wait()
    finally:
        for t in tasks:
            t.cancel()
        await runner.cleanup()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nTRINETRA server stopped.", flush=True)


if __name__ == "__main__":
    main()
