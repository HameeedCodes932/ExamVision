import numpy as np

from app.services.video.models import Frame, SourceConfig, SourceType


class TestSourceConfig:
    def test_defaults(self) -> None:
        cfg = SourceConfig(uri="rtsp://cam", source_type=SourceType.RTSP, camera_id="cam1")
        assert cfg.target_fps == 15
        assert cfg.target_width == 1280
        assert cfg.target_height == 720

    def test_file_default_loop_off(self) -> None:
        cfg = SourceConfig(uri="test.mp4", source_type=SourceType.FILE, camera_id="f1")
        assert cfg.file_loop is False


class TestFrame:
    def test_shape_properties(self) -> None:
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(camera_id="cam1", timestamp=100.0, frame_index=1, raw=raw)
        assert frame.height == 480
        assert frame.width == 640
        assert frame.shape == (480, 640, 3)

    def test_encode_jpeg(self) -> None:
        raw = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frame = Frame(camera_id="cam1", timestamp=100.0, frame_index=1, raw=raw)
        jpeg = frame.encode_jpeg(quality=50)
        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0
