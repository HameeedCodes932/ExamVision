from dataclasses import dataclass, field


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None
    keypoints: list[tuple[float, float, float]] | None = None


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    frame_shape: tuple[int, int] = (0, 0)
    timestamp: float = 0.0
    inference_ms: float = 0.0


COCO_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    67: "cell phone",
}

PERSON_CLASS_ID = 0
CELL_PHONE_CLASS_ID = 67
