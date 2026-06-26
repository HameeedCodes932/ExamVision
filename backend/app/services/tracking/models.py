from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class TrackState(Enum):
    NEW = auto()
    CONFIRMED = auto()
    LOST = auto()
    REMOVED = auto()


@dataclass
class STrack:
    track_id: int
    state: TrackState = TrackState.NEW
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None
    bbox: np.ndarray | None = None
    confidence: float = 0.0
    class_id: int = 0
    frame_since_update: int = 0
    start_frame: int = 0
    bbox_history: list[np.ndarray] = field(default_factory=list)
    max_history: int = 30

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_removed(self) -> bool:
        return self.state == TrackState.REMOVED

    @property
    def tlwh(self) -> np.ndarray:
        if self.bbox is None:
            return np.array([0, 0, 0, 0], dtype=float)
        return self.bbox.copy()

    @property
    def tlbr(self) -> np.ndarray:
        tlwh = self.tlwh
        return np.array(
            [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]],
            dtype=float,
        )

    @property
    def xyxy(self) -> np.ndarray:
        return self.tlbr

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "state": self.state.name.lower(),
            "bbox": self.xyxy.tolist() if self.bbox is not None else None,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "frame_since_update": self.frame_since_update,
        }

    def mark_confirmed(self) -> None:
        self.state = TrackState.CONFIRMED

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED


@dataclass
class TrackResult:
    tracks: list[STrack] = field(default_factory=list)
    frame_index: int = 0
    timestamp: float = 0.0
