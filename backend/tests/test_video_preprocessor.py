import numpy as np

from app.services.video.preprocessor import FramePreprocessor


class TestFramePreprocessor:
    def test_resize(self) -> None:
        pp = FramePreprocessor(target_width=320, target_height=240, normalize=False, to_rgb=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pp.process(frame)
        assert result.shape == (240, 320, 3)

    def test_to_rgb(self) -> None:
        pp = FramePreprocessor(target_width=640, target_height=480, normalize=False, to_rgb=True)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255
        rgb = pp.process(bgr)
        assert rgb[0, 0, 0] == 0
        assert rgb[0, 0, 2] == 255

    def test_normalize(self) -> None:
        pp = FramePreprocessor(target_width=100, target_height=100, normalize=True, to_rgb=False)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = pp.process(frame)
        assert result.dtype == np.float32
        assert abs(result[0, 0, 0] - 128.0 / 255.0) < 1e-6

    def test_no_resize_needed(self) -> None:
        pp = FramePreprocessor(target_width=640, target_height=480, normalize=False, to_rgb=False)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pp.process(frame)
        assert result.shape == (480, 640, 3)
