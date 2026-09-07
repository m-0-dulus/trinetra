# Trinetra V2 — Two-Jet Phone Battle

A local Wi-Fi hackathon prototype: two phones act as JET-01 and JET-02, while the laptop runs the Trinetra brain and judge dashboard.

## Included
- Two phone browser pilot UIs
- Camera capture with rear-camera preference
- WebSocket video/state transport
- YOLO + ByteTrack tracking on each phone feed
- Constant-velocity occlusion prediction/reacquisition state
- Target reticle + lock progress
- FIRE button + cooldown + centralized hit validation
- Hit/death screen
- Android vibration when `navigator.vibrate()` is available
- Visual/audio fallback when vibration is unavailable
- Judge dashboard for both jets
- Round reset and event log
- Test-camera mode for reliable UI/game-flow demos

## Setup
Python 3.10+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open the printed `/phone` URL on each phone and `/dashboard` on the laptop.

Both devices must be on the same Wi-Fi/LAN.

### Camera note
Mobile browsers commonly require a secure context for `getUserMedia()`. If the phone blocks camera access over plain HTTP, try:

```bash
python server.py --https
```

Then use the printed HTTPS URL. The included TEST CAMERA mode can demonstrate the complete UI/game flow even when camera permission is unavailable.

## Model
Default model: `yolo11n.pt`. For cardboard jets, a custom trained aircraft/jet model is recommended:

```bash
python server.py --model path/to/best.pt
```

## Demo
1. Connect JET-01 and JET-02.
2. Point each phone at the opponent.
3. Wait for TARGET LOCK.
4. Press FIRE.
5. The target receives HIT and shows JET DOWN; vibration runs where supported.
6. The round resets automatically after a short delay.

This is a simulated tabletop game. FIRE/HIT/KILL are software game events only; no physical weapon or actuator is controlled.
