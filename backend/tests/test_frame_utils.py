import numpy as np

from app.services.video.frame_utils import encode_jpeg, overlay_detections, overlay_text


class TestEncodeJpeg:
    def test_encodes_jpeg_bytes(self) -> None:
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        data = encode_jpeg(frame, quality=50)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_different_quality_affects_size(self) -> None:
        frame = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
        low = len(encode_jpeg(frame, quality=10))
        high = len(encode_jpeg(frame, quality=95))
        assert low < high


class TestOverlayDetections:
    def test_returns_copy_with_boxes(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {"bbox": (10, 10, 50, 50), "class_name": "person", "confidence": 0.95}
        ]
        result = overlay_detections(frame, detections)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
        assert not np.array_equal(result, frame)

    def test_empty_detections_returns_same_shape(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = overlay_detections(frame, [])
        assert result.shape == frame.shape


class TestOverlayText:
    def test_draws_text_lines(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = overlay_text(frame, ["line1", "line2"])
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_empty_lines(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = overlay_text(frame, [])
        assert result.shape == frame.shape
