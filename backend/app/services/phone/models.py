from dataclasses import dataclass

from app.services.detection.models import BBox


@dataclass
class PhoneDetection:
    bbox: BBox
    confidence: float
    student_track_id: int | None = None
    assignment_method: str | None = None

    @property
    def is_assigned(self) -> bool:
        return self.student_track_id is not None

    @property
    def assignment(self) -> str:
        if self.student_track_id is None:
            return "unassigned"
        return f"student_{self.student_track_id}"

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.to_list(),
            "confidence": round(self.confidence, 4),
            "student_track_id": self.student_track_id,
            "assignment_method": self.assignment_method,
            "assignment": self.assignment,
        }
