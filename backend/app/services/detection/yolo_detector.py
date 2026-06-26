import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ultralytics import YOLO

from app.core.config import settings

logger = logging.getLogger(__name__)

MODEL_MAP: dict[str, str] = {
    "nano": "yolov8n",
    "small": "yolov8s",
    "medium": "yolov8m",
}

POSE_MODEL_MAP: dict[str, str] = {
    "nano": "yolov8n-pose",
    "small": "yolov8s-pose",
    "medium": "yolov8m-pose",
}


def _resolve_model_path(
    model_name: str,
    backend: str,
    quantization: str,
) -> str:
    name = model_name
    if backend == "onnx":
        name += ".onnx"
    elif quantization == "fp16":
        name += "_fp16.pt"
    elif quantization == "int8":
        name += "_int8.pt"
    else:
        name += ".pt"
    candidate = settings.model_dir / name
    if candidate.exists():
        return str(candidate)
    if backend == "onnx":
        alt = settings.model_dir / f"{model_name}.pt"
        if alt.exists():
            return str(alt)
    return model_name


class YOLODetector:
    def __init__(
        self,
        model_size: str = "nano",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        pose: bool = False,
        backend: str = "pytorch",
        quantization: str = "none",
        executor_workers: int = 2,
    ) -> None:
        self._model_size = model_size
        self._confidence = confidence
        self._iou = iou_threshold
        self._pose = pose
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
        model_map = POSE_MODEL_MAP if self._pose else MODEL_MAP
        return model_map.get(self._model_size, MODEL_MAP["nano"])

    @property
    def model_path(self) -> str:
        return _resolve_model_path(self.model_name, self._backend, self._quantization)

    async def load(self) -> None:
        if self._model is not None:
            return
        path = self.model_path
        logger.info(
            "Loading model: %s (device=%s, backend=%s, quant=%s)",
            path,
            self._device,
            self._backend,
            self._quantization,
        )

        def _load() -> YOLO:
            model = YOLO(path)
            if self._half and self._device == "cuda:0":
                model.model.half()
                logger.info("Model converted to FP16")
            return model

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(self._executor, _load)
        logger.info("Model loaded successfully")

    async def detect(self, frame: np.ndarray) -> list[dict]:
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
            return _parse_results(results)

        return await loop.run_in_executor(self._executor, _infer)

    async def unload(self) -> None:
        self._model = None
        self._executor.shutdown(wait=False)
        logger.info("Model unloaded")


def _parse_results(results: list) -> list[dict]:
    parsed: list[dict] = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            parsed.append(
                {
                    "bbox": xyxy,
                    "confidence": float(boxes.conf[i]),
                    "class_id": int(boxes.cls[i]),
                    "class_name": r.names[int(boxes.cls[i])],
                }
            )
        keypoints = r.keypoints
        if keypoints is not None:
            for det, kps in zip(parsed, keypoints.data, strict=False):
                kp_list = []
                for kp in kps:
                    x, y, conf = kp.tolist()
                    kp_list.append((float(x), float(y), float(conf)))
                det["keypoints"] = kp_list
    return parsed
