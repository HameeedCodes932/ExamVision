import cv2
import numpy as np


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def overlay_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in d.get("bbox", (0, 0, 0, 0))]
        label = d.get("class_name", "object")
        conf = d.get("confidence", 0.0)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(out, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def overlay_text(
    frame: np.ndarray, lines: list[str], position: tuple[int, int] = (10, 30)
) -> np.ndarray:
    out = frame.copy()
    y = position[1]
    for line in lines:
        cv2.putText(out, line, (position[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 25
    return out
