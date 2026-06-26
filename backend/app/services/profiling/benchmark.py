import csv
import io
import logging

import numpy as np

from app.services.detection.pipeline import DetectionPipeline, DetectionPipelineConfig
from app.services.pose.pipeline import PosePipeline
from app.services.pose.pose_estimator import PoseEstimator
from app.services.profiling import FrameProfiler
from app.services.video.preprocessor import FramePreprocessor

logger = logging.getLogger(__name__)


async def run_benchmark(
    num_frames: int = 100,
    frame_size: tuple[int, int] = (1280, 720),
    model_size: str = "nano",
    backend: str = "pytorch",
    quantization: str = "none",
) -> str:
    profiler = FrameProfiler(window=num_frames)

    det_config = DetectionPipelineConfig(model_size=model_size, confidence=0.5)
    detection = DetectionPipeline(config=det_config, profiler=profiler)
    await detection.load()

    pose_est = PoseEstimator(
        model_size=model_size,
        backend="pytorch",
        quantization="none",
        executor_workers=2,
    )
    await pose_est.load()
    pose_pipeline = PosePipeline(estimator=pose_est, profiler=profiler)

    preprocessor = FramePreprocessor()

    h, w = frame_size
    dummy_frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)

    logger.info(
        "Running benchmark: %d frames, %s model, %s backend",
        num_frames,
        model_size,
        backend,
    )

    for i in range(num_frames):
        profiler.log_frame()

        with profiler.timer("preprocess"):
            preprocessor.process(dummy_frame)

        with profiler.timer("detect"):
            det_result = await detection.process(dummy_frame)

        if det_result.detections:
            with profiler.timer("pose"):
                await pose_pipeline.process_detections(dummy_frame, det_result)

        if (i + 1) % 20 == 0 or i == 0:
            logger.info("  Frame %d/%d", i + 1, num_frames)

    await detection.unload()
    await pose_est.unload()

    return profiler.generate_benchmark_report()


def benchmark_to_csv(report: str) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    headers = [
        "Stage",
        "Count",
        "Mean (ms)",
        "Median (ms)",
        "Min (ms)",
        "Max (ms)",
        "P95 (ms)",
    ]
    writer.writerow(headers)
    for line in report.split("\n"):
        parts = line.split()
        if len(parts) == 7 and parts[0] != "Stage":
            try:
                int(parts[1])
                writer.writerow(parts)
            except ValueError:
                pass
    return buf.getvalue()
