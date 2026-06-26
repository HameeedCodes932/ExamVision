import math

from app.services.pose.keypoints import Keypoint, keypoints_to_objects


def get_angle(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
    a = math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
    b = math.sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)
    c = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if a * c == 0:
        return 0.0
    cos_angle = (a**2 + c**2 - b**2) / (2 * a * c)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def filter_keypoints(
    raw_keypoints: list[tuple[float, float, float]], conf_thresh: float = 0.5
) -> list[Keypoint]:
    objs = keypoints_to_objects(raw_keypoints)
    return [kp for kp in objs if kp.confidence >= conf_thresh]


def get_valid_kp(
    raw_keypoints: list[tuple[float, float, float]], index: int, conf_thresh: float = 0.5
) -> tuple[float, float] | None:
    if index < 0 or index >= len(raw_keypoints):
        return None
    x, y, conf = raw_keypoints[index]
    if conf < conf_thresh:
        return None
    return (x, y)


def is_standing(
    raw_keypoints: list[tuple[float, float, float]],
    conf_thresh: float = 0.5,
    height_ratio_thresh: float = 1.5,
) -> bool:
    nose = get_valid_kp(raw_keypoints, 0, conf_thresh)
    lhip = get_valid_kp(raw_keypoints, 11, conf_thresh)
    rhip = get_valid_kp(raw_keypoints, 12, conf_thresh)
    lankle = get_valid_kp(raw_keypoints, 15, conf_thresh)
    rankle = get_valid_kp(raw_keypoints, 16, conf_thresh)

    if nose is None:
        return False

    hip_y = min(
        (lhip[1] if lhip else float("inf")),
        (rhip[1] if rhip else float("inf")),
    )
    ankle_y = min(
        (lankle[1] if lankle else float("inf")),
        (rankle[1] if rankle else float("inf")),
    )

    if hip_y == float("inf") or ankle_y == float("inf"):
        return False

    body_height = ankle_y - nose[1]
    shoulder_to_hip = hip_y - nose[1]

    if body_height <= 0 or shoulder_to_hip <= 0:
        return False

    return body_height / shoulder_to_hip > height_ratio_thresh


def is_hand_raised(
    raw_keypoints: list[tuple[float, float, float]],
    side: str = "left",
    conf_thresh: float = 0.5,
) -> bool:
    wrist_idx = 9 if side == "left" else 10
    shoulder_idx = 5 if side == "left" else 6

    wrist = get_valid_kp(raw_keypoints, wrist_idx, conf_thresh)
    shoulder = get_valid_kp(raw_keypoints, shoulder_idx, conf_thresh)

    if wrist is None or shoulder is None:
        return False

    return wrist[1] < shoulder[1]


def is_looking_away(
    raw_keypoints: list[tuple[float, float, float]],
    conf_thresh: float = 0.5,
) -> bool:
    nose = get_valid_kp(raw_keypoints, 0, conf_thresh)
    leye = get_valid_kp(raw_keypoints, 1, conf_thresh)
    reye = get_valid_kp(raw_keypoints, 2, conf_thresh)
    rear = get_valid_kp(raw_keypoints, 4, conf_thresh)
    lear = get_valid_kp(raw_keypoints, 3, conf_thresh)

    if nose is None:
        return False

    if leye is not None and reye is not None:
        eye_distance = math.sqrt((leye[0] - reye[0]) ** 2 + (leye[1] - reye[1]) ** 2)
        if eye_distance < 10:
            return True

    if rear is not None and lear is not None:
        ear_distance = abs(rear[0] - lear[0])
        if ear_distance > 20:
            return False

    return False


def estimate_head_direction(
    raw_keypoints: list[tuple[float, float, float]],
    conf_thresh: float = 0.5,
) -> str:
    nose = get_valid_kp(raw_keypoints, 0, conf_thresh)
    leye = get_valid_kp(raw_keypoints, 1, conf_thresh)
    reye = get_valid_kp(raw_keypoints, 2, conf_thresh)
    lear = get_valid_kp(raw_keypoints, 3, conf_thresh)
    rear = get_valid_kp(raw_keypoints, 4, conf_thresh)

    if nose is None:
        return "unknown"

    if leye is not None and reye is not None:
        eye_center_x = (leye[0] + reye[0]) / 2
        if nose[0] < eye_center_x - 10:
            return "left"
        if nose[0] > eye_center_x + 10:
            return "right"

    if lear is not None and rear is not None:
        ear_dist = abs(lear[0] - rear[0])
        if ear_dist < 15:
            return "back"

    return "forward"


def compute_skeleton(
    raw_keypoints: list[tuple[float, float, float]], conf_thresh: float = 0.5
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for i, name in enumerate([
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]):
        kp = get_valid_kp(raw_keypoints, i, conf_thresh)
        result[name] = {"x": kp[0], "y": kp[1]} if kp else {"x": 0.0, "y": 0.0, "visible": False}
        if kp:
            result[name]["visible"] = True
    return result
