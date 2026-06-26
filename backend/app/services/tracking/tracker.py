import logging

from app.services.detection.models import DetectionResult
from app.services.tracking.byte_track import ByteTrack
from app.services.tracking.models import TrackResult

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        max_time_lost: int = 30,
    ) -> None:
        self._byte_track = ByteTrack(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            max_time_lost=max_time_lost,
        )

    @property
    def tracks(self):
        return self._byte_track.tracks

    def update(self, detections: list[dict]) -> TrackResult:
        return self._byte_track.update(detections)

    def process_detection_result(
        self, detection_result: DetectionResult
    ) -> TrackResult:
        raw = [
            {
                "bbox": d.bbox.to_list(),
                "confidence": d.confidence,
                "class_id": d.class_id,
            }
            for d in detection_result.detections
        ]
        return self.update(raw)

    def reset(self) -> None:
        self._byte_track.reset()
        logger.info("Tracker reset")
