import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

from tracking.target_manager import TargetManager
from visualization.hud import draw_hud


def parse_source(value):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser(description="Trinetra V1 visual tracking demo")
    parser.add_argument("--source", default="0", help="webcam index or video path")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    print(f"[Trinetra] Loading model: {args.model}")
    model = YOLO(args.model)

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    manager = TargetManager()
    last_time = time.perf_counter()
    fps = 0.0

    print("\nTrinetra V1 running.")
    print("t = select largest track | r = reset | o = demo occlusion | p = state | q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO tracking with ByteTrack.
        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            verbose=False,
        )[0]

        tracks = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else np.full(len(boxes), -1)
            )

            for bbox, conf, track_id in zip(boxes, confs, ids):
                x1, y1, x2, y2 = bbox.astype(int)
                crop = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
                tracks.append({
                    "id": int(track_id),
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(conf),
                    "appearance": manager.appearance_descriptor(crop),
                })

        manager.update(tracks, frame.shape)

        # Handle keyboard controls.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            manager.select_largest(tracks)
        elif key == ord("r"):
            manager.reset()
        elif key == ord("o"):
            manager.demo_occlusion = not manager.demo_occlusion
        elif key == ord("p"):
            print(manager.state_dict())

        now = time.perf_counter()
        dt = max(now - last_time, 1e-6)
        last_time = now
        instant_fps = 1.0 / dt
        fps = 0.9 * fps + 0.1 * instant_fps if fps else instant_fps

        display = frame.copy()

        # Draw all tracks.
        for tr in tracks:
            x1, y1, x2, y2 = tr["bbox"]
            is_target = manager.target_id == tr["id"]
            label = f"JET-{tr['id']}"
            if is_target:
                label += "  TARGET"

            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                display, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

        # Draw predicted target when tracking/occluded.
        if manager.predicted_center is not None:
            px, py = manager.predicted_center
            cv2.circle(display, (int(px), int(py)), 12, (255, 255, 255), 2)
            cv2.line(
                display,
                (int(px) - 20, int(py)),
                (int(px) + 20, int(py)),
                (255, 255, 255), 2
            )
            cv2.line(
                display,
                (int(px), int(py) - 20),
                (int(px), int(py) + 20),
                (255, 255, 255), 2
            )
            cv2.putText(
                display, "PREDICTED",
                (int(px) + 15, int(py) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        draw_hud(
            display,
            manager.state_dict(),
            fps=fps,
        )

        cv2.imshow("TRINETRA V1", display)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
