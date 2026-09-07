import time

class ConstantVelocityPredictor:
    def __init__(self):
        self.x = self.y = None
        self.vx = self.vy = 0.0
        self.last_t = None

    def update(self, x, y, t):
        if self.x is not None and self.last_t is not None:
            dt = max(1e-3, t - self.last_t)
            self.vx = 0.7*self.vx + 0.3*((x-self.x)/dt)
            self.vy = 0.7*self.vy + 0.3*((y-self.y)/dt)
        self.x, self.y, self.last_t = x, y, t

    def predict(self, dt):
        if self.x is None:
            return None
        return (max(0.0,min(1.0,self.x+self.vx*dt)),
                max(0.0,min(1.0,self.y+self.vy*dt)))

class TargetManager:
    def __init__(self):
        self.target_id = None
        self.last_seen = 0.0
        self.last_target = None
        self.predictor = ConstantVelocityPredictor()
        self.identity_switches = 0
        self.reacquisitions = 0
        self.last_error = ""
        self._state = "SEARCHING"
        self._detections = []

    def update(self, detections, width, height):
        now = time.time()
        normalized = []
        for d in detections:
            x1,y1,x2,y2 = d["xyxy"]
            normalized.append({
                "id": d.get("id",-1),
                "conf": float(d.get("conf",0.0)),
                "cx": float(((x1+x2)/2)/max(1,width)),
                "cy": float(((y1+y2)/2)/max(1,height)),
                "w": float((x2-x1)/max(1,width)),
                "h": float((y2-y1)/max(1,height)),
                "xyxy": [float(x1),float(y1),float(x2),float(y2)]
            })

        candidate = max(normalized, key=lambda d:d["conf"], default=None)
        if self.target_id is None and candidate:
            self.target_id = candidate["id"]
            self.reacquisitions += 1
        elif candidate and self.target_id != candidate["id"] and now-self.last_seen > 1.0:
            self.target_id = candidate["id"]
            self.identity_switches += 1
            self.reacquisitions += 1

        target = next((d for d in normalized if d["id"] == self.target_id), None)
        if target:
            self.last_seen = now
            self.predictor.update(target["cx"], target["cy"], now)
            self.last_target = target
            self._state = "TRACKING"
        elif self.target_id is not None and now-self.last_seen <= 1.0:
            self._state = "OCCLUDED"
        else:
            self._state = "SEARCHING"
        self._detections = normalized

    def snapshot(self):
        target = self.last_target.copy() if self.last_target else None
        prediction = None
        if self._state == "OCCLUDED":
            prediction = self.predictor.predict(min(1.5,max(0.0,time.time()-self.last_seen)))
        return {
            "state": self._state,
            "target": target,
            "prediction": prediction,
            "target_id": self.target_id,
            "detections": self._detections,
            "identity_switches": self.identity_switches,
            "reacquisitions": self.reacquisitions,
            "last_error": self.last_error,
        }
