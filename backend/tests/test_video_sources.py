
from app.services.video.models import SourceConfig, SourceType
from app.services.video.sources import RTSPSource, USBSource


class TestRTSPSource:
    def test_init(self) -> None:
        cfg = SourceConfig(
            uri="rtsp://192.168.1.100:554/stream1",
            source_type=SourceType.RTSP,
            camera_id="cam1",
        )
        src = RTSPSource(cfg)
        assert src.config.camera_id == "cam1"
        assert src.is_opened is False


class TestUSBSource:
    def test_init(self) -> None:
        cfg = SourceConfig(uri="0", source_type=SourceType.USB, camera_id="cam_usb")
        src = USBSource(cfg)
        assert src.config.camera_id == "cam_usb"
        assert src.is_opened is False
