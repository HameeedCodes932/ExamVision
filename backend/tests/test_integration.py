import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.behaviour import BehaviourAnalyser
from app.services.detection.models import BBox, Detection, DetectionResult
from app.services.detection.pipeline import DetectionPipeline
from app.services.profiling import FrameProfiler
from app.services.scoring import SuspicionScorer


@pytest.fixture(autouse=True)
def mock_yolo_model():
    with patch("app.services.detection.yolo_detector.YOLO") as mock_yolo:
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = [[100, 100, 200, 400]]
        mock_result.boxes.conf = [0.95]
        mock_result.boxes.cls = [0]
        mock_result.boxes.is_track = False
        mock_result.names = {0: "person"}
        mock_result.keypoints = None
        mock_yolo.return_value = mock_instance
        mock_instance.return_value = [mock_result]
        yield mock_yolo


@pytest.fixture
def dummy_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def profiler() -> FrameProfiler:
    return FrameProfiler(window=10)


class TestFullPipeline:
    async def test_detection_then_pose(
        self, dummy_frame: np.ndarray, profiler: FrameProfiler
    ) -> None:
        pipeline = DetectionPipeline(profiler=profiler)
        await pipeline.load()

        result = await pipeline.process(dummy_frame)
        assert isinstance(result, DetectionResult)
        assert result.frame_shape == (480, 640)
        assert result.timestamp > 0
        assert result.inference_ms >= 0

        await pipeline.unload()

    async def test_detection_with_profiler(self, dummy_frame: np.ndarray) -> None:
        prof = FrameProfiler(window=5)
        pipeline = DetectionPipeline(profiler=prof)
        await pipeline.load()

        for _ in range(3):
            prof.log_frame()
            with prof.timer("detect_test"):
                await pipeline.process(dummy_frame)

        summary = prof.summary()
        assert "detect_test" in summary
        assert summary["detect_test"]["count"] == 3
        assert summary["total_frames"] == 3

        await pipeline.unload()

    async def test_behaviour_scoring_integration(self) -> None:
        analyser = BehaviourAnalyser()
        scorer = SuspicionScorer()

        states = [
            {
                "track_id": 1,
                "head_direction": "left",
                "is_standing": False,
                "phone_detected": False,
            },
            {
                "track_id": 1,
                "head_direction": "left",
                "is_standing": False,
                "phone_detected": False,
            },
            {"track_id": 1, "head_direction": "left", "is_standing": True, "phone_detected": False},
        ]
        for s in states:
            events = analyser.analyse(s["track_id"], s)
            for ev in events:
                scorer.update(s["track_id"], [ev])

        score = scorer.get_score(1)
        assert score is not None
        assert score.total > 0

        analyser.reset()
        scorer.reset()

    def test_profiler_reset(self, profiler: FrameProfiler) -> None:
        profiler.log_frame()
        assert profiler.total_frames == 1
        profiler.reset()
        assert profiler.total_frames == 0

    def test_profiler_benchmark_report(self, profiler: FrameProfiler) -> None:
        profiler.log_frame()
        with profiler.timer("stage_a"):
            pass
        with profiler.timer("stage_b"):
            pass
        report = profiler.generate_benchmark_report()
        assert "Benchmark Report" in report
        assert "stage_a" in report
        assert "stage_b" in report

    async def test_get_profiler_disabled(self) -> None:
        with patch("app.api.routes.settings") as mock_settings:
            mock_settings.profiling_enabled = False
            from app.api.routes import get_profiler

            result = await get_profiler()
            assert result is None

    async def test_full_detection_result_fields(self, dummy_frame: np.ndarray) -> None:
        pipeline = DetectionPipeline()
        await pipeline.load()

        result = await pipeline.process(dummy_frame)
        assert isinstance(result.frame_shape, tuple)
        assert len(result.frame_shape) == 2
        assert isinstance(result.detections, list)
        for det in result.detections:
            assert isinstance(det, Detection)
            assert isinstance(det.bbox, BBox)
            assert det.bbox.x1 < det.bbox.x2
            assert det.bbox.y1 < det.bbox.y2
            assert det.confidence > 0

        await pipeline.unload()

    async def test_behaviour_e2e_flow(self) -> None:
        analyser = BehaviourAnalyser()
        scorer = SuspicionScorer()

        frame_count = 5
        for i in range(frame_count):
            state = {
                "track_id": 1,
                "head_direction": "left" if i % 2 == 0 else "right",
                "is_standing": i > 2,
                "phone_detected": i == 4,
                "gaze_target": "away",
            }
            events = analyser.analyse(1, state)
            for ev in events:
                scorer.update(1, [ev])

        score = scorer.get_score(1)
        assert score is not None
        assert score.total > 0

        breakdown_human = {k: round(v, 1) for k, v in score.breakdown.items()}
        assert any(v > 0 for v in breakdown_human.values())

        analyser.reset()
        scorer.reset()

    async def test_concurrent_profiler(self, profiler: FrameProfiler) -> None:
        async def worker(name: str, count: int) -> None:
            for _ in range(count):
                profiler.log_frame()
                with profiler.timer(name):
                    await asyncio.sleep(0.001)

        await asyncio.gather(
            worker("task_a", 5),
            worker("task_b", 5),
            worker("task_a", 5),
        )

        summary = profiler.summary()
        assert summary["total_frames"] == 15
        assert summary["task_a"]["count"] == 10
        assert summary["task_b"]["count"] == 5

    async def test_benchmark_function(self) -> None:
        from app.services.profiling.benchmark import run_benchmark

        with (
            patch("app.services.detection.yolo_detector.YOLO") as mock_yolo,
            patch("app.services.pose.pose_estimator.YOLO") as mock_pose_yolo,
        ):
            mock_yolo.return_value = MagicMock()
            mock_pose_yolo.return_value = MagicMock()
            result = await run_benchmark(num_frames=2)
        assert isinstance(result, str)
        assert "Benchmark" in result
