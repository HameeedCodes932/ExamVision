from dataclasses import dataclass, field


@dataclass
class FaceDetection:
    bbox: tuple[float, float, float, float]
    confidence: float
    landmarks: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def num_landmarks(self) -> int:
        return len(self.landmarks)

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "landmarks": [list(lm) for lm in self.landmarks],
        }


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float

    @property
    def direction(self) -> str:
        if abs(self.yaw) < 20 and abs(self.pitch) < 20:
            return "FORWARD"
        if self.yaw > 20:
            return "LEFT"
        if self.yaw < -20:
            return "RIGHT"
        if self.pitch > 20:
            return "DOWN"
        if self.pitch < -20:
            return "UP"
        return "FORWARD"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "direction": self.direction,
        }
