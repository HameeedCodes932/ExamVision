import math

from app.services.gaze.models import GazeTarget

YAW_THRESHOLD_FORWARD = 20.0
YAW_THRESHOLD_BEHIND = 55.0
PITCH_THRESHOLD_FORWARD = 15.0
PITCH_THRESHOLD_PAPER = 25.0


def classify_gaze_target(gaze_yaw: float, gaze_pitch: float) -> GazeTarget:
    if gaze_pitch > PITCH_THRESHOLD_PAPER:
        return GazeTarget.LOOKING_AT_PAPER

    if abs(gaze_yaw) >= YAW_THRESHOLD_BEHIND:
        return GazeTarget.LOOKING_BEHIND

    if gaze_yaw > YAW_THRESHOLD_FORWARD:
        if gaze_yaw < YAW_THRESHOLD_BEHIND:
            return GazeTarget.LOOKING_LEFT
        return GazeTarget.LOOKING_BEHIND

    if gaze_yaw < -YAW_THRESHOLD_FORWARD:
        if gaze_yaw > -YAW_THRESHOLD_BEHIND:
            return GazeTarget.LOOKING_RIGHT
        return GazeTarget.LOOKING_BEHIND

    if abs(gaze_pitch) > PITCH_THRESHOLD_FORWARD:
        if gaze_pitch > 0:
            return GazeTarget.LOOKING_AT_PAPER
        return GazeTarget.LOOKING_FORWARD

    return GazeTarget.LOOKING_FORWARD


def compute_gaze_from_iris(
    eye_corners: tuple[float, float, float, float],
    iris_center: tuple[float, float],
    head_yaw: float = 0.0,
    head_pitch: float = 0.0,
    max_gaze_yaw: float = 30.0,
    max_gaze_pitch: float = 25.0,
) -> tuple[float, float]:
    left_x, left_y, right_x, right_y = eye_corners
    eye_center_x = (left_x + right_x) / 2
    eye_center_y = (left_y + right_y) / 2
    eye_width = math.hypot(right_x - left_x, right_y - left_y)
    if eye_width < 1e-6:
        return head_yaw, head_pitch

    dx = (iris_center[0] - eye_center_x) / eye_width
    dy = (iris_center[1] - eye_center_y) / eye_width

    gaze_yaw = dx * max_gaze_yaw + head_yaw
    gaze_pitch = -dy * max_gaze_pitch + head_pitch

    return gaze_yaw, gaze_pitch
