import cv2

from app.services.video.models import SourceConfig
from app.services.video.sources.base import VideoSource


class USBSource(VideoSource):
    def __init__(self, config: SourceConfig) -> None:
        super().__init__(config)

    def _open_capture(self) -> cv2.VideoCapture:
        device_index = int(self.config.uri)
        cap = cv2.VideoCapture(device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.target_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
