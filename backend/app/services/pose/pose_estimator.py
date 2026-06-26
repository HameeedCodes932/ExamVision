import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ultralytics import YOLO

from app.services.detection.yolo_detector import _resolve_model_path

logger = logging.getLogger(__name__)

POSE_MODEL_MAP: dict[str, str] = {
    "nano": "yolov8n-pose",
    "small": "yolov8s-pose",
    "medium": "yolov8m-pose",
}


class PoseEstimator:
    def __init__(
        self,
        model_size: str = "nano",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        backend: str = "pytorch",
        quantization: str = "none",
        executor_workers: int = 2,
    ) -> None:
        self._model_size = model_size
        self._confidence = confidence
        self._iou = iou_threshold
        self._backend = backend
        self._quantization = quantization
        self._model: YOLO | None = None
        self._executor = ThreadPoolExecutor(max_workers=executor_workers)
        self._device = self._detect_device()
        self._half = quantization == "fp16"

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @property
    def model_name(self) -> str:
        return POSE_MODEL_MAP.get(self._model_size, POSE_MODEL_MAP["nano"])

    @property
    def model_path(self) -> str:
        return _resolve_model_path(self.model_name, self._backend, self._quantization)

    async def load(self) -> None:
        if self._model is not None:
            return
        path = self.model_path
        logger.info(
            "Loading pose model: %s (device=%s, backend=%s, quant=%s)",
            path,
            self._device,
            self._backend,
            self._quantization,
        )

        def _load() -> YOLO:
            model = YOLO(path)
            if self._half and self._device == "cuda:0":
                model.model.half()
                logger.info("Pose model converted to FP16")
            return model

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(self._executor, _load)
        logger.info("Pose model loaded")

    async def estimate(self, frame: np.ndarray) -> list[dict]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()

        def _infer() -> list[dict]:
            kwargs: dict = {
                "conf": self._confidence,
                "iou": self._iou,
                "device": self._device,
                "verbose": False,
            }
            if self._half:
                kwargs["half"] = True
            results = self._model(frame, **kwargs)
            return _parse_pose_results(results)

        return await loop.run_in_executor(self._executor, _infer)

    async def estimate_person(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> list[dict]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        results = await self.estimate(crop)
        for r in results:
            kps = r.get("keypoints", [])
            shifted = []
            for kp in kps:
                shifted.append((kp[0] + x1, kp[1] + y1, kp[2]))
            r["keypoints"] = shifted
            r["bbox"] = [float(x1), float(y1), float(x2), float(y2)]
        return results

    async def unload(self) -> None:
        self._model = None
        self._executor.shutdown(wait=False)
        logger.info("Pose model unloaded")


def _parse_pose_results(results: list) -> list[dict]:
    parsed: list[dict] = []
    for r in results:
        boxes = r.boxes
        kps_data = r.keypoints
        if boxes is None:
            continue
        for i in range(len(boxes)):
            entry: dict = {
                "bbox": boxes.xyxy[i].tolist(),
                "confidence": float(boxes.conf[i]),
                "class_id": int(boxes.cls[i]),
                "class_name": r.names[int(boxes.cls[i])],
                "keypoints": [],
            }
            if kps_data is not None and i < len(kps_data.data):
                for kp in kps_data.data[i]:
                    x, y, conf = kp.tolist()
                    entry["keypoints"].append((float(x), float(y), float(conf)))
            parsed.append(entry)
    return parsed
