import numpy as np


class ConstantVelocityPredictor:
    """Small, dependency-light 2D constant-velocity predictor."""

    def __init__(self):
        self.position = None
        self.velocity = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.position = None
        self.velocity[:] = 0

    def update(self, position, dt=1.0):
        p = np.asarray(position, dtype=np.float32)
        if self.position is not None and dt > 1e-5:
            measured_velocity = (p - self.position) / dt
            self.velocity = 0.65 * self.velocity + 0.35 * measured_velocity
        self.position = p.copy()

    def predict(self, dt=1.0):
        if self.position is None:
            return None
        return self.position + self.velocity * dt
