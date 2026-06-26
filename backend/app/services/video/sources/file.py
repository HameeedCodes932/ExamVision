import cv2

from app.services.video.models import SourceConfig
from app.services.video.sources.base import VideoSource


class FileSource(VideoSource):
    def __init__(self, config: SourceConfig) -> None:
        super().__init__(config)
        self._total_frames: int = 0

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.config.uri)
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap

    def read(self):
        frame = super().read()
        if frame is None and self.config.file_loop and self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_index = 0
            frame = super().read()
        return frame
