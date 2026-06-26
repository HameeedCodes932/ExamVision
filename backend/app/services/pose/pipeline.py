import asyncio
import logging

import numpy as np

from app.services.detection.models import DetectionResult
from app.services.pose.pose_estimator import PoseEstimator
from app.services.profiling import FrameProfiler

logger = logging.getLogger(__name__)


class PoseResult:
    def __init__(self, person_results: list[dict]) -> None:
        self.person_results = person_results

    @property
    def num_people(self) -> int:
        return len(self.person_results)

    def get_keypoints(self, index: int = 0) -> list[tuple[float, float, float]]:
        if index < len(self.person_results):
            return self.person_results[index].get("keypoints", [])
        return []

    def to_dict(self) -> list[dict]:
        return self.person_results


class PosePipeline:
    def __init__(
        self,
        estimator: PoseEstimator,
        profiler: FrameProfiler | None = None,
    ) -> None:
        self._estimator = estimator
        self._profiler = profiler

    async def process_frame(self, frame: np.ndarray) -> PoseResult:
        if self._profiler:
            with self._profiler.timer("pose_frame"):
                raw = await self._estimator.estimate(frame)
        else:
            raw = await self._estimator.estimate(frame)
        return PoseResult(raw)

    async def process_detections(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult | None = None,
    ) -> PoseResult:
        if detection_result is None or not detection_result.detections:
            return await self.process_frame(frame)

        async def _estimate_person(det: object) -> list[dict]:
            d = det
            bbox = (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)
            if self._profiler:
                with self._profiler.timer("pose_person"):
                    return await self._estimator.estimate_person(frame, bbox)
            return await self._estimator.estimate_person(frame, bbox)

        tasks = [_estimate_person(d) for d in detection_result.detections]
        results = await asyncio.gather(*tasks)

        all_results: list[dict] = []
        for person_kps in results:
            if person_kps:
                all_results.extend(person_kps)

        if not all_results:
            all_results = await self._estimator.estimate(frame)

        return PoseResult(all_results)
