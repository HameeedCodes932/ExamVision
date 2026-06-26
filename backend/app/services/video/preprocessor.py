import cv2
import numpy as np


class FramePreprocessor:
    def __init__(
        self,
        target_width: int = 640,
        target_height: int = 640,
        normalize: bool = True,
        to_rgb: bool = True,
    ) -> None:
        self._target_width = target_width
        self._target_height = target_height
        self._normalize = normalize
        self._to_rgb = to_rgb

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self._to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (
            frame.shape[1] != self._target_width
            or frame.shape[0] != self._target_height
        ):
            frame = cv2.resize(
                frame, (self._target_width, self._target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        if self._normalize:
            frame = frame.astype(np.float32) / 255.0
        return frame
