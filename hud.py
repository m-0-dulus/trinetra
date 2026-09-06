import cv2


def draw_hud(frame, state, fps=0.0):
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 82), (0, 0, 0), -1)
    cv2.putText(
        frame, "TRINETRA", (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2
    )
    cv2.putText(
        frame, "Persistent Visual Tracking", (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1
    )

    # Right panel
    panel_x = max(0, w - 320)
    cv2.rectangle(frame, (panel_x, 95), (w - 10, 300), (0, 0, 0), -1)

    lines = [
        f"TARGET: {state['target']}",
        f"STATUS: {state['state']}",
        f"FPS: {fps:.1f}",
        f"REACQUISITIONS: {state['reacquisitions']}",
        f"ID SWITCHES: {state['identity_switches']}",
    ]

    y = 125
    for line in lines:
        cv2.putText(
            frame, line, (panel_x + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1
        )
        y += 32

    cv2.putText(
        frame, "T target | R reset | O occlude | Q quit",
        (15, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1
    )
