from dataclasses import dataclass

COCO_KEYPOINTS: list[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

NUM_KEYPOINTS = 17

SKELETON: list[tuple[int, int, str]] = [
    (5, 6, "shoulders"),
    (5, 7, "left_upper_arm"),
    (7, 9, "left_forearm"),
    (6, 8, "right_upper_arm"),
    (8, 10, "right_forearm"),
    (5, 11, "left_torso"),
    (6, 12, "right_torso"),
    (11, 12, "hips"),
    (11, 13, "left_thigh"),
    (13, 15, "left_shin"),
    (12, 14, "right_thigh"),
    (14, 16, "right_shin"),
    (0, 1, "nose_to_left_eye"),
    (0, 2, "nose_to_right_eye"),
    (1, 3, "left_eye_to_ear"),
    (2, 4, "right_eye_to_ear"),
    (0, 5, "nose_to_left_shoulder"),
    (0, 6, "nose_to_right_shoulder"),
]

SKELETON_COLORS: list[tuple[int, int, int]] = [
    (255, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (255, 0, 255),
    (255, 0, 255),
    (255, 0, 255),
    (255, 0, 255),
    (0, 165, 255),
    (0, 165, 255),
]

KEYPOINT_COLORS: list[tuple[int, int, int]] = [
    (255, 255, 255),
    (255, 0, 0),
    (255, 0, 0),
    (255, 128, 0),
    (255, 128, 0),
    (255, 255, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 0, 255),
    (0, 0, 255),
    (255, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (0, 255, 255),
    (128, 128, 0),
    (128, 128, 0),
]

HEAD_KEYPOINTS = {0, 1, 2, 3, 4}
UPPER_BODY_KEYPOINTS = {5, 6, 7, 8, 9, 10}
LOWER_BODY_KEYPOINTS = {11, 12, 13, 14, 15, 16}


def keypoint_name(idx: int) -> str:
    if 0 <= idx < NUM_KEYPOINTS:
        return COCO_KEYPOINTS[idx]
    return f"unknown_{idx}"


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float
    index: int

    @property
    def name(self) -> str:
        return keypoint_name(self.index)

    @property
    def is_visible(self) -> bool:
        return self.confidence > 0.5


def keypoints_to_objects(
    raw_keypoints: list[tuple[float, float, float]],
) -> list[Keypoint]:
    return [
        Keypoint(x=k[0], y=k[1], confidence=k[2], index=i)
        for i, k in enumerate(raw_keypoints)
        if i < NUM_KEYPOINTS
    ]
