import asyncio
import contextlib
import logging
from collections import defaultdict

from app.services.video.models import SourceConfig, SourceType
from app.services.video.sources import FileSource, RTSPSource, USBSource, VideoSource

logger = logging.getLogger(__name__)


class StreamManager:
    def __init__(self) -> None:
        self._sources: dict[str, VideoSource] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._running = False

    def _build_source(self, config: SourceConfig) -> VideoSource:
        match config.source_type:
            case SourceType.RTSP:
                return RTSPSource(config)
            case SourceType.USB:
                return USBSource(config)
            case SourceType.FILE:
                return FileSource(config)
            case _:
                raise ValueError(f"Unknown source type: {config.source_type}")

    async def add_source(self, config: SourceConfig) -> None:
        if config.camera_id in self._sources:
            logger.warning("Source %s already exists", config.camera_id)
            return
        source = self._build_source(config)
        self._sources[config.camera_id] = source
        logger.info("Source %s added (%s)", config.camera_id, config.uri)

    async def remove_source(self, camera_id: str) -> None:
        await self.stop_source(camera_id)
        self._sources.pop(camera_id, None)
        self._queues.pop(camera_id, None)
        logger.info("Source %s removed", camera_id)

    async def start_source(self, camera_id: str) -> None:
        source = self._sources.get(camera_id)
        if source is None:
            raise ValueError(f"Source {camera_id} not found")
        if camera_id in self._tasks:
            return
        source.open()
        self._tasks[camera_id] = asyncio.create_task(
            self._read_loop(camera_id, source)
        )
        logger.info("Source %s reading started", camera_id)

    async def stop_source(self, camera_id: str) -> None:
        task = self._tasks.pop(camera_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        source = self._sources.get(camera_id)
        if source is not None:
            source.release()
        logger.info("Source %s stopped", camera_id)

    async def start_all(self) -> None:
        for cid in list(self._sources.keys()):
            await self.start_source(cid)

    async def stop_all(self) -> None:
        for cid in list(self._tasks.keys()):
            await self.stop_source(cid)

    async def subscribe(self, camera_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._queues[camera_id].append(q)
        return q

    def unsubscribe(self, camera_id: str, q: asyncio.Queue) -> None:
        if camera_id in self._queues:
            self._queues[camera_id] = [x for x in self._queues[camera_id] if x is not q]

    def get_active_cameras(self) -> list[str]:
        return list(self._tasks.keys())

    def subscriber_count(self, camera_id: str) -> int:
        return len(self._queues.get(camera_id, []))

    def get_all_cameras(self) -> list[dict]:
        return [
            {
                "camera_id": cid,
                "uri": src.config.uri,
                "source_type": src.config.source_type.value,
                "active": cid in self._tasks,
            }
            for cid, src in self._sources.items()
        ]

    async def _read_loop(self, camera_id: str, source: VideoSource) -> None:
        loop_interval = 1.0 / source.config.target_fps
        while True:
            t0 = asyncio.get_event_loop().time()
            frame = await asyncio.to_thread(source.read)
            if frame is not None:
                for q in self._queues.get(camera_id, []):
                    with contextlib.suppress(asyncio.QueueFull):
                        q.put_nowait(frame)
            elapsed = asyncio.get_event_loop().time() - t0
            sleep = max(0.0, loop_interval - elapsed)
            await asyncio.sleep(sleep)
