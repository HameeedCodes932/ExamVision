from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class EventOut(BaseModel):
    id: UUID
    student_id: UUID
    timestamp: datetime
    event_type: str
    confidence: float | None = None
    details: str | None = None

    model_config = {"from_attributes": True}


class AlertOut(BaseModel):
    id: UUID
    student_id: UUID
    alert_type: str
    severity: str
    message: str
    created_at: datetime
    resolved_at: datetime | None = None
    resolved: bool

    model_config = {"from_attributes": True}


class StudentOut(BaseModel):
    id: UUID
    track_id: int
    seat_label: str | None = None
    roll_number: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class StudentWithEvents(StudentOut):
    events: list[EventOut] = []
    alerts: list[AlertOut] = []


class DetectionOut(BaseModel):
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float


class SuspicionScoreOut(BaseModel):
    student_id: UUID
    track_id: int
    total: float
    breakdown: dict[str, float]
    level: str


# ── Phase 11: Reporting & Dashboard ────────────────────────────────────


class StreamOut(BaseModel):
    camera_id: str
    uri: str
    source_type: str
    active: bool
    connected_clients: int = 0


# ── Phase 13: Enhanced Reports ─────────────────────────────────────────


EVENT_WEIGHTS: dict[str, float] = {
    "phone_detected": 40.0,
    "student_left_seat": 30.0,
    "head_down": 25.0,
    "looking_left": 20.0,
    "standing": 15.0,
    "repeated_head_turns": 15.0,
    "body_twisting": 15.0,
}

SEVERITY_SCORES: dict[str, float] = {
    "low": 10.0,
    "medium": 25.0,
    "high": 50.0,
    "critical": 80.0,
}

RISK_LEVELS: list[tuple[str, float]] = [
    ("critical", 80.0),
    ("warning", 50.0),
    ("observe", 20.0),
    ("normal", 0.0),
]


def classify_risk(total_score: float) -> str:
    for level, threshold in RISK_LEVELS:
        if total_score >= threshold:
            return level
    return "normal"


class TimelineEntryOut(BaseModel):
    timestamp: datetime
    event_type: str | None = None
    confidence: float | None = None
    details: str | None = None
    alert_type: str | None = None
    severity: str | None = None
    message: str | None = None


class RiskAssessmentOut(BaseModel):
    risk_score: float
    risk_level: str
    event_risk_contribution: dict[str, float]
    alert_risk_contribution: float


class StudentReportOut(BaseModel):
    student_id: UUID
    track_id: int
    seat_label: str | None = None
    roll_number: str | None = None
    total_events: int = 0
    event_breakdown: dict[str, int] = {}
    total_alerts: int = 0
    unresolved_alerts: int = 0
    max_severity: str | None = None
    suspicion_score: float | None = None
    suspicion_level: str | None = None
    risk_assessment: RiskAssessmentOut | None = None
    timeline: list[TimelineEntryOut] = []


class ReportOut(BaseModel):
    exam_id: str
    generated_at: datetime
    total_students: int
    total_events: int
    total_alerts: int
    students: list[StudentReportOut]
