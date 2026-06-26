import abc
import logging

import cv2

from app.services.video.models import Frame, SourceConfig

logger = logging.getLogger(__name__)


class VideoSource(abc.ABC):
    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._reconnect_attempts = 0

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @abc.abstractmethod
    def _open_capture(self) -> cv2.VideoCapture:
        ...

    def open(self) -> None:
        if self.is_opened:
            return
        self._cap = self._open_capture()
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "Source %s opened: %dx%d @ %.1f FPS",
            self.config.camera_id,
            actual_w,
            actual_h,
            self._cap.get(cv2.CAP_PROP_FPS),
        )

    def read(self) -> Frame | None:
        if not self.is_opened or self._cap is None:
            return None
        ret, raw = self._cap.read()
        if not ret:
            logger.warning("Source %s: read failed", self.config.camera_id)
            return None
        self._frame_index += 1
        return Frame(
            camera_id=self.config.camera_id,
            timestamp=cv2.getTickCount() / cv2.getTickFrequency(),
            frame_index=self._frame_index,
            raw=raw,
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Source %s released", self.config.camera_id)

    def __del__(self) -> None:
        self.release()
