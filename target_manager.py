import cv2
import numpy as np
from .kalman import ConstantVelocityPredictor


class TargetManager:
    def __init__(self):
        self.target_id = None
        self.target_appearance = None
        self.target_center = None
        self.predicted_center = None
        self.last_seen_frame = -1
        self.frame_index = 0
        self.state = "SEARCHING"
        self.predictor = ConstantVelocityPredictor()
        self.demo_occlusion = False
        self.reacquisition_count = 0
        self.identity_switches = 0

    @staticmethod
    def appearance_descriptor(crop):
        if crop is None or crop.size == 0:
            return None
        small = cv2.resize(crop, (32, 32))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    @staticmethod
    def appearance_similarity(a, b):
        if a is None or b is None:
            return 0.0
        return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))

    @staticmethod
    def center(bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    def select_largest(self, tracks):
        if not tracks:
            print("[Trinetra] No tracks available.")
            return
        chosen = max(
            tracks,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1])
        )
        self.target_id = chosen["id"]
        self.target_appearance = chosen["appearance"]
        self.target_center = self.center(chosen["bbox"])
        self.predicted_center = self.target_center.copy()
        self.predictor.reset()
        self.predictor.update(self.target_center)
        self.state = "TRACKING"
        self.last_seen_frame = self.frame_index
        print(f"[Trinetra] Target selected: JET-{self.target_id}")

    def reset(self):
        self.target_id = None
        self.target_appearance = None
        self.target_center = None
        self.predicted_center = None
        self.predictor.reset()
        self.state = "SEARCHING"

    def update(self, tracks, frame_shape):
        self.frame_index += 1

        if self.target_id is None:
            self.state = "SEARCHING"
            return

        target = next((t for t in tracks if t["id"] == self.target_id), None)

        # Demo switch lets you force the target into an occluded state.
        if self.demo_occlusion:
            target = None

        if target is not None:
            c = self.center(target["bbox"])
            self.target_center = c
            self.predictor.update(c)
            self.predicted_center = c.copy()
            self.target_appearance = (
                0.85 * self.target_appearance + 0.15 * target["appearance"]
                if self.target_appearance is not None and target["appearance"] is not None
                else target["appearance"]
            )
            self.last_seen_frame = self.frame_index
            self.state = "TRACKING"
            return

        # Target isn't currently visible: predict.
        self.state = "OCCLUDED"
        predicted = self.predictor.predict(1.0)
        if predicted is not None:
            h, w = frame_shape[:2]
            predicted[0] = np.clip(predicted[0], 0, w - 1)
            predicted[1] = np.clip(predicted[1], 0, h - 1)
            self.predicted_center = predicted

        # Look for a candidate matching the target appearance and prediction.
        if tracks:
            candidates = []
            for t in tracks:
                c = self.center(t["bbox"])
                appearance = self.appearance_similarity(
                    self.target_appearance, t["appearance"]
                )
                if self.predicted_center is not None:
                    dist = np.linalg.norm(c - self.predicted_center)
                    diag = np.linalg.norm([frame_shape[1], frame_shape[0]])
                    spatial = max(0.0, 1.0 - dist / max(diag * 0.35, 1.0))
                else:
                    spatial = 0.0

                score = 0.65 * max(appearance, 0.0) + 0.35 * spatial
                candidates.append((score, t))

            candidates.sort(key=lambda x: x[0], reverse=True)
            score, best = candidates[0]

            if score > 0.50:
                old_id = self.target_id
                self.target_id = best["id"]
                self.target_center = self.center(best["bbox"])
                self.predictor.update(self.target_center)
                self.predicted_center = self.target_center.copy()
                self.target_appearance = best["appearance"]
                self.state = "REACQUIRED"
                self.reacquisition_count += 1

                # Track IDs may legitimately change across an occlusion.
                if old_id != self.target_id:
                    self.identity_switches += 1

    def state_dict(self):
        return {
            "target": f"JET-{self.target_id}" if self.target_id is not None else "NONE",
            "state": self.state,
            "last_seen_frame": self.last_seen_frame,
            "reacquisitions": self.reacquisition_count,
            "identity_switches": self.identity_switches,
            "predicted_center": (
                tuple(np.round(self.predicted_center, 1))
                if self.predicted_center is not None else None
            ),
        }
