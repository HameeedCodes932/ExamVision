import math

from app.services.face.models import HeadPose


def classify_head_direction(yaw: float, pitch: float) -> str:
    return HeadPose(yaw=yaw, pitch=pitch, roll=0.0).direction


def get_eye_aspect_ratio(landmarks: list[tuple[float, float, float]]) -> float:
    if len(landmarks) < 468:
        return 0.0

    left_eye_idxs = [33, 160, 158, 133, 153, 144]
    right_eye_idxs = [362, 385, 387, 263, 373, 380]

    def _ear(idxs: list[int]) -> float:
        p1, p2, p3, p4, p5, p6 = [landmarks[i][:2] for i in idxs]
        a = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
        b = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
        c = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
        return (a + b) / (2.0 * c) if c != 0 else 0.0

    left_ear = _ear(left_eye_idxs)
    right_ear = _ear(right_eye_idxs)
    return (left_ear + right_ear) / 2.0


def get_mouth_aspect_ratio(landmarks: list[tuple[float, float, float]]) -> float:
    if len(landmarks) < 468:
        return 0.0

    upper_lip = landmarks[13][:2]
    lower_lip = landmarks[14][:2]
    left_mouth = landmarks[61][:2]
    right_mouth = landmarks[291][:2]

    vertical = math.hypot(upper_lip[0] - lower_lip[0], upper_lip[1] - lower_lip[1])
    horizontal = math.hypot(left_mouth[0] - right_mouth[0], left_mouth[1] - right_mouth[1])

    return vertical / horizontal if horizontal != 0 else 0.0


def is_face_visible(landmarks: list[tuple[float, float, float]]) -> bool:
    return len(landmarks) >= 468
