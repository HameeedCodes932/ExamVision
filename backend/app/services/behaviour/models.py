from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    LOOKING_LEFT = "looking_left"
    PHONE_DETECTED = "phone_detected"
    STANDING = "standing"
    HEAD_DOWN = "head_down"
    REPEATED_HEAD_TURNS = "repeated_head_turns"
    STUDENT_LEFT_SEAT = "student_left_seat"
    BODY_TWISTING = "body_twisting"


@dataclass
class BehaviourEvent:
    track_id: int
    event_type: EventType
    severity: Severity
    timestamp: float
    details: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "details": self.details,
            "confidence": self.confidence,
        }
