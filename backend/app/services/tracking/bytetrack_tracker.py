class ByteTrackTracker:
    def __init__(self, max_lost: int = 30, iou_threshold: float = 0.3) -> None:
        self._max_lost = max_lost
        self._iou_threshold = iou_threshold

    def update(self, detections: list[dict]) -> list[dict]:
        ...

    def reset(self) -> None:
        ...
