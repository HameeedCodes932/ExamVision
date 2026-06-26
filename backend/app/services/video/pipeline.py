import asyncio
import contextlib
import logging

from app.core.config import settings
from app.services.detection import DetectionPipeline
from app.services.detection.models import DetectionResult
from app.services.profiling import FrameProfiler
from app.services.video.models import Frame
from app.services.video.preprocessor import FramePreprocessor
from app.services.video.stream_manager import StreamManager

logger = logging.getLogger(__name__)


class VideoIngestionPipeline:
    def __init__(
        self,
        stream_manager: StreamManager,
        detection_pipeline: DetectionPipeline,
        preprocessor: FramePreprocessor | None = None,
        profiler: FrameProfiler | None = None,
    ) -> None:
        self._stream_manager = stream_manager
        self._detection_pipeline = detection_pipeline
        self._preprocessor = preprocessor or FramePreprocessor()
        self._profiler = profiler
        self._running = False
        self._task: asyncio.Task | None = None
        self._frame_counter = 0
        self._process_every_n = settings.video_process_every_n

    async def process_frame(self, frame: Frame) -> DetectionResult:
        self._frame_counter += 1
        if self._profiler:
            self._profiler.log_frame()
            with self._profiler.timer("preprocess"):
                preprocessed = self._preprocessor.process(frame.raw)
        else:
            preprocessed = self._preprocessor.process(frame.raw)
        frame.preprocessed = preprocessed

        if self._profiler:
            with self._profiler.timer("detect"):
                result = await self._detection_pipeline.process(frame.raw)
        else:
            result = await self._detection_pipeline.process(frame.raw)
        return result

    def _should_skip(self) -> bool:
        if self._process_every_n <= 1:
            return False
        return self._frame_counter % self._process_every_n != 0

    async def run_stream(self, camera_id: str, callback=None) -> None:
        q = await self._stream_manager.subscribe(camera_id)
        logger.info("Pipeline consuming stream: %s", camera_id)
        try:
            while True:
                frame = await q.get()
                if self._should_skip():
                    continue
                result = await self.process_frame(frame)
                if callback:
                    await callback(frame, result)
        except asyncio.CancelledError:
            pass
        finally:
            self._stream_manager.unsubscribe(camera_id, q)

    async def start(self, camera_id: str, callback=None) -> None:
        self._running = True
        self._task = asyncio.create_task(self.run_stream(camera_id, callback))

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
