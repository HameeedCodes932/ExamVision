from dataclasses import dataclass
from enum import Enum


class GazeTarget(str, Enum):
    LOOKING_FORWARD = "looking_forward"
    LOOKING_LEFT = "looking_left"
    LOOKING_RIGHT = "looking_right"
    LOOKING_AT_PAPER = "looking_at_paper"
    LOOKING_BEHIND = "looking_behind"
    LOOKING_AT_ANOTHER_STUDENT = "looking_at_another_student"


@dataclass
class GazeVector:
    gaze_yaw: float
    gaze_pitch: float
    target: GazeTarget = GazeTarget.LOOKING_FORWARD

    @property
    def target_label(self) -> str:
        return self.target.value

    def to_dict(self) -> dict[str, float | str]:
        return {
            "gaze_yaw": round(self.gaze_yaw, 2),
            "gaze_pitch": round(self.gaze_pitch, 2),
            "target": self.target_label,
        }
