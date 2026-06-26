from dataclasses import dataclass
from enum import Enum

import numpy as np


class SourceType(Enum):
    RTSP = "rtsp"
    USB = "usb"
    FILE = "file"


@dataclass
class Frame:
    camera_id: str
    timestamp: float
    frame_index: int
    raw: np.ndarray
    preprocessed: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.raw.shape

    @property
    def height(self) -> int:
        return self.raw.shape[0]

    @property
    def width(self) -> int:
        return self.raw.shape[1]

    def encode_jpeg(self, quality: int = 85) -> bytes:
        import cv2

        success, buf = cv2.imencode(".jpg", self.raw, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()


@dataclass
class SourceConfig:
    uri: str
    source_type: SourceType
    camera_id: str
    target_fps: int = 15
    target_width: int = 1280
    target_height: int = 720
    buffer_size: int = 128
    reconnect_delay: float = 2.0
    max_reconnect_attempts: int = 0
    file_loop: bool = False
