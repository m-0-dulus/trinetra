import math
import time
from collections import deque


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


class ConstantVelocityPredictor:
    """2-D constant velocity predictor in normalized image coordinates."""

    def __init__(self):
        self.x = self.y = None
        self.vx = self.vy = 0.0
        self.last_t = None

    def update(self, x, y, t):
        if self.x is not None and self.last_t is not None:
            dt = max(1e-3, t - self.last_t)
            vx = (x - self.x) / dt
            vy = (y - self.y) / dt
            self.vx = 0.72 * self.vx + 0.28 * vx
            self.vy = 0.72 * self.vy + 0.28 * vy
        self.x, self.y, self.last_t = x, y, t

    def predict(self, dt):
        if self.x is None:
            return None
        return (
            clamp(self.x + self.vx * dt),
            clamp(self.y + self.vy * dt),
        )


class TargetManager:
    """Track a jet, preserve its identity, and reacquire it after occlusion.

    The manager deliberately works on *jet candidates*, not arbitrary YOLO
    objects. Each candidate carries a compact appearance/shape descriptor.
    Once a target is locked, that descriptor becomes the remembered identity.
    """

    def __init__(self):
        self.target_id = None
        self.remembered = None
        self.last_seen = 0.0
        self.last_target = None
        self.predictor = ConstantVelocityPredictor()
        self.identity_switches = 0
        self.reacquisitions = 0
        self.last_error = ""
        self._state = "SEARCHING"
        self._detections = []
        self._lost_since = None
        self._lock_history = deque(maxlen=8)
        self.designated_point = None
        self.visible_streak = 0
        self.missed_streak = 0

    @staticmethod
    def _appearance_distance(a, b):
        if not a or not b:
            return 1.0
        aa = a.get("appearance", [])
        bb = b.get("appearance", [])
        if not aa or not bb or len(aa) != len(bb):
            return 1.0
        # Histograms are L1 normalized.  Lower is better.
        d = sum(abs(float(x) - float(y)) for x, y in zip(aa, bb)) / 2.0
        return clamp(d)

    @staticmethod
    def _shape_distance(a, b):
        if not a or not b:
            return 1.0
        keys = ("aspect", "area", "solidity", "extent", "wing_score")
        vals = []
        for k in keys:
            av = float(a.get(k, 0.0))
            bv = float(b.get(k, 0.0))
            scale = max(0.12, abs(av), abs(bv))
            vals.append(min(1.0, abs(av - bv) / scale))
        return sum(vals) / len(vals)

    def _reid_score(self, candidate):
        if not self.remembered:
            return 0.0
        appearance = self._appearance_distance(candidate, self.remembered)
        shape = self._shape_distance(candidate, self.remembered)
        motion = 0.5
        pred = self.predictor.predict(min(1.2, max(0.0, time.time() - self.last_seen)))
        if pred:
            motion = math.hypot(candidate["cx"] - pred[0], candidate["cy"] - pred[1]) / 0.75
            motion = clamp(motion)
        # Higher is better.
        return 0.52 * (1.0 - appearance) + 0.28 * (1.0 - shape) + 0.20 * (1.0 - motion)

    def _choose_initial(self, candidates):
        if not candidates:
            return None
        # Prefer a large, confident jet close to the reticle. This is much less
        # likely to select a face/hand than simply taking the largest box.
        def score(d):
            center = math.hypot(d["cx"] - 0.5, d["cy"] - 0.5)
            central = 1.0 - min(1.0, center / 0.71)
            size = min(1.0, math.sqrt(max(0.0, d["area"])) * 4.0)
            return 0.45 * d.get("jet_score", 0.0) + 0.30 * d.get("conf", 0.0) + 0.20 * central + 0.05 * size
        return max(candidates, key=score)

    def update(self, detections, width, height):
        now = time.time()
        normalized = []
        for d in detections:
            x1, y1, x2, y2 = d["xyxy"]
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            area = (w * h) / max(1.0, width * height)
            item = dict(d)
            item.update({
                "cx": clamp(((x1 + x2) / 2) / max(1, width)),
                "cy": clamp(((y1 + y2) / 2) / max(1, height)),
                "w": clamp(w / max(1, width)),
                "h": clamp(h / max(1, height)),
                "area": clamp(area),
            })
            normalized.append(item)

        target = None
        # If the pilot tapped the video, choose the candidate nearest that point.
        if self.target_id is None and self.designated_point and normalized:
            dx, dy = self.designated_point
            target = min(normalized, key=lambda d: math.hypot(d["cx"] - dx, d["cy"] - dy))
            if math.hypot(target["cx"] - dx, target["cy"] - dy) <= 0.22:
                self.target_id = target.get("id", -1)
                self.reacquisitions += 1
            else:
                target = None
            self.designated_point = None

        # If we already have a local tracker id, use it first.
        if target is None:
            target = None
        if self.target_id is not None:
            target = next((d for d in normalized if d.get("id") == self.target_id), None)

        # Tracker id disappeared: use the remembered identity, not the next
        # arbitrary object. This is the key re-identification step.
        if target is None and normalized and self.remembered is not None:
            ranked = sorted(normalized, key=self._reid_score, reverse=True)
            best = ranked[0]
            score = self._reid_score(best)
            if score >= 0.55:
                old = self.target_id
                self.target_id = best.get("id", -1)
                target = best
                if old != self.target_id:
                    self.identity_switches += 1
                self.reacquisitions += 1

        if target is None and self.target_id is None:
            target = self._choose_initial(normalized)
            if target is not None:
                self.target_id = target.get("id", -1)
                self.reacquisitions += 1
            self.designated_point = None

        if target is not None:
            self.last_seen = now
            self._lost_since = None
            self.visible_streak += 1
            self.missed_streak = 0
            self.predictor.update(target["cx"], target["cy"], now)
            self.last_target = dict(target)
            self._state = "TRACKING"
            # Keep the identity profile fresh, but don't let a wildly different
            # object overwrite a locked target's identity.
            if self.remembered is None:
                self.remembered = dict(target)
        elif self.target_id is not None:
            self.visible_streak = 0
            self.missed_streak += 1
            if self._lost_since is None:
                self._lost_since = now
            lost_age = now - self.last_seen
            # Brief detector misses are NOT occlusion. This prevents the UI from
            # claiming an occlusion every time the contour detector skips a frame.
            if lost_age < 0.45:
                self._state = "TRACKING"
            elif lost_age <= 1.5:
                self._state = "OCCLUDED"
            else:
                self._state = "SEARCHING"
        else:
            self.visible_streak = 0
            self.missed_streak = 0
            self._state = "SEARCHING"

        self._detections = normalized

    def designate(self, cx, cy):
        """Ask the manager to prefer the candidate nearest a user-designated point."""
        self.designated_point = (clamp(cx), clamp(cy))

    def remember_lock(self):
        """Freeze the currently tracked jet's descriptor as its identity."""
        if self.last_target:
            self.remembered = dict(self.last_target)
            self._lock_history.append({
                "target_id": self.target_id,
                "appearance": list(self.remembered.get("appearance", [])),
                "shape": {
                    k: self.remembered.get(k)
                    for k in ("aspect", "area", "solidity", "extent", "wing_score")
                },
            })

    def snapshot(self):
        target = self.last_target.copy() if self.last_target else None
        prediction = None
        occlusion_age = 0.0
        if self._state == "OCCLUDED":
            occlusion_age = max(0.0, time.time() - self.last_seen)
            prediction = self.predictor.predict(min(1.5, occlusion_age))
        return {
            "state": self._state,
            "target": target,
            "prediction": prediction,
            "target_id": self.target_id,
            "remembered": self.remembered is not None,
            "detections": self._detections,
            "identity_switches": self.identity_switches,
            "reacquisitions": self.reacquisitions,
            "occlusion_age": occlusion_age,
            "visible_streak": self.visible_streak,
            "missed_streak": self.missed_streak,
            "last_error": self.last_error,
        }
