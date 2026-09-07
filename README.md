# TRINETRA V3 — Hardware Demo

This build is tuned for the cardboard fighter-jet tabletop demo.

## Run
```bash
cd /Users/ansh/Downloads/trinetra_v3_complete
source .venv/bin/activate
python3 server.py
```

Phones: `https://<MAC-IP>:8443/phone`
Dashboard: `http://127.0.0.1:8080/dashboard`

Both phones should select their own jet and tap **CONNECT**. The server uses one HTTPS port for both phones and one local HTTP dashboard port.

## Tracking pipeline
- YOLO accepts **airplane class only**.
- A cardboard/shape detector supplements YOLO for the physical tabletop jets.
- Shape filtering rejects compact face/hand-like regions.
- A tapped target is assigned a stable local ID.
- Appearance + shape + motion are remembered for re-identification.
- Brief detector misses stay `TRACKING`; only a sustained loss becomes `OCCLUDED`.
- During real occlusion, a constant-velocity predictor supplies the predicted target position.
- Lock requires the visible target to stay inside the reticle for the lock dwell time.
- FIRE is enabled only after lock.
- A confirmed hit sends `JET DOWN` to the target phone and updates the dashboard.

## Dashboard feeds
The dashboard uses `/stream/JET-01` and `/stream/JET-02` MJPEG feeds. This is intentionally separate from the command WebSocket for reliable live video while AI processing runs.
