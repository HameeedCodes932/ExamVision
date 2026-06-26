import cv2

from app.services.video.models import SourceConfig
from app.services.video.sources.base import VideoSource


class RTSPSource(VideoSource):
    def __init__(self, config: SourceConfig) -> None:
        super().__init__(config)

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.config.uri, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        return cap
