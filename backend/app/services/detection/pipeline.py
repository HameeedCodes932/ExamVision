import logging
import time
from dataclasses import dataclass

import numpy as np

from app.core.config import settings
from app.services.detection.models import DetectionResult
from app.services.detection.post_processor import build_result
from app.services.detection.yolo_detector import YOLODetector
from app.services.profiling import FrameProfiler

logger = logging.getLogger(__name__)


@dataclass
class DetectionPipelineConfig:
    model_size: str = "nano"
    confidence: float = 0.5
    iou_threshold: float = 0.45
    person_only: bool = True
    min_area: int = 900
    apply_nms: bool = True
    nms_iou: float = 0.5


class DetectionPipeline:
    def __init__(
        self,
        config: DetectionPipelineConfig | None = None,
        profiler: FrameProfiler | None = None,
    ) -> None:
        self._config = config or DetectionPipelineConfig()
        self._detector: YOLODetector | None = None
        self._loaded = False
        self._profiler = profiler

    async def load(self) -> None:
        if self._loaded:
            return
        self._detector = YOLODetector(
            model_size=self._config.model_size,
            confidence=self._config.confidence,
            iou_threshold=self._config.iou_threshold,
            backend=settings.detection_inference_backend,
            quantization=settings.detection_quantization,
            executor_workers=settings.detection_executor_workers,
        )
        await self._detector.load()
        self._loaded = True
        logger.info(
            "Detection pipeline ready (model=%s, conf=%.2f, backend=%s, quant=%s)",
            self._config.model_size,
            self._config.confidence,
            settings.detection_inference_backend,
            settings.detection_quantization,
        )

    async def process(self, frame: np.ndarray) -> DetectionResult:
        if not self._loaded or self._detector is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")

        h, w = frame.shape[:2]
        ts = time.time()

        if self._profiler:
            with self._profiler.timer("detection"):
                raw = await self._detector.detect(frame)
        else:
            raw = await self._detector.detect(frame)

        t0 = time.perf_counter()
        result = build_result(
            raw=raw,
            frame_shape=(h, w),
            timestamp=ts,
            inference_ms=0.0,
            person_only=self._config.person_only,
            min_area=self._config.min_area,
            apply_nms_flag=self._config.apply_nms,
            iou_threshold=self._config.nms_iou,
        )
        result.inference_ms = (time.perf_counter() - t0) * 1000

        return result

    async def unload(self) -> None:
        if self._detector is not None:
            await self._detector.unload()
        self._loaded = False
        logger.info("Detection pipeline unloaded")
