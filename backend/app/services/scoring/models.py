from dataclasses import dataclass, field

SCORE_WEIGHTS: dict[str, float] = {
    "phone_detected": 40.0,
    "student_left_seat": 30.0,
    "head_down": 25.0,
    "looking_left": 20.0,
    "standing": 15.0,
    "repeated_head_turns": 15.0,
    "body_twisting": 15.0,
}

NORMAL_THRESHOLD = 20.0
OBSERVE_THRESHOLD = 50.0
WARNING_THRESHOLD = 80.0


def classify_score(total: float) -> str:
    if total >= WARNING_THRESHOLD:
        return "critical"
    if total >= OBSERVE_THRESHOLD:
        return "warning"
    if total >= NORMAL_THRESHOLD:
        return "observe"
    return "normal"


@dataclass
class SuspicionScore:
    track_id: int
    total: float
    breakdown: dict[str, float] = field(default_factory=dict)
    level: str = "normal"

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "total": round(self.total, 1),
            "breakdown": {k: round(v, 1) for k, v in self.breakdown.items()},
            "level": self.level,
        }
