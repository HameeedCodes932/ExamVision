import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.services.detection.models import CELL_PHONE_CLASS_ID, BBox
from app.services.detection.yolo_detector import YOLODetector
from app.services.phone.models import PhoneDetection
from app.services.pose.utils import get_valid_kp

logger = logging.getLogger(__name__)

IOU_ASSIGN_THRESHOLD = 0.05
HAND_DISTANCE_THRESHOLD = 150.0

LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10


class PhoneDetector:
    def __init__(
        self,
        model_size: str = "nano",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        iou_assign_threshold: float = IOU_ASSIGN_THRESHOLD,
        hand_distance_threshold: float = HAND_DISTANCE_THRESHOLD,
    ) -> None:
        self._model_size = model_size
        self._confidence = confidence
        self._iou = iou_threshold
        self._iou_assign = iou_assign_threshold
        self._hand_dist = hand_distance_threshold
        self._detector: YOLODetector | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def load(self) -> None:
        if self._detector is not None:
            return
        self._detector = YOLODetector(
            model_size=self._model_size,
            confidence=self._confidence,
            iou_threshold=self._iou,
        )
        await self._detector.load()
        logger.info("Phone detector loaded")

    async def detect_phones(self, frame: np.ndarray) -> list[dict]:
        if self._detector is None:
            raise RuntimeError("Phone detector not loaded. Call load() first.")

        raw = await self._detector.detect(frame)
        return [d for d in raw if d["class_id"] == CELL_PHONE_CLASS_ID]

    async def process(
        self,
        frame: np.ndarray,
        person_detections: list[dict] | None = None,
        person_keypoints: list[list[tuple[float, float, float]]] | None = None,
    ) -> list[PhoneDetection]:
        phone_raw = await self.detect_phones(frame)
        return await self.associate_phones(phone_raw, person_detections, person_keypoints)

    async def associate_phones(
        self,
        phone_raw: list[dict],
        person_detections: list[dict] | None = None,
        person_keypoints: list[list[tuple[float, float, float]]] | None = None,
    ) -> list[PhoneDetection]:
        detections = [_raw_to_phone(p) for p in phone_raw]

        if not detections or not person_detections:
            return detections

        loop = asyncio.get_running_loop()

        def _assign() -> list[PhoneDetection]:
            for phone in detections:
                assigned_id, method = self._find_best_match(
                    phone, person_detections, person_keypoints
                )
                if assigned_id is not None:
                    phone.student_track_id = assigned_id
                    phone.assignment_method = method
            return detections

        return await loop.run_in_executor(self._executor, _assign)

    async def unload(self) -> None:
        if self._detector is not None:
            await self._detector.unload()
        self._executor.shutdown(wait=False)
        logger.info("Phone detector unloaded")

    def _find_best_match(
        self,
        phone: PhoneDetection,
        person_detections: list[dict],
        person_keypoints: list[list[tuple[float, float, float]]] | None,
    ) -> tuple[int | None, str | None]:
        best_track_id: int | None = None
        best_method: str | None = None
        best_score = 0.0

        for _, person in enumerate(person_detections):
            person_bbox = BBox(*person["bbox"])
            track_id: int | None = person.get("track_id")
            if track_id is None:
                continue

            iou = phone.bbox.iou(person_bbox)
            if iou > self._iou_assign and iou > best_score:
                best_track_id = track_id
                best_method = "iou"
                best_score = iou

        if best_track_id is not None and best_method == "iou":
            return best_track_id, best_method

        phone_cx, phone_cy = phone.bbox.cx, phone.bbox.cy
        for i, person in enumerate(person_detections):
            track_id: int | None = person.get("track_id")
            if track_id is None:
                continue

            kps = person_keypoints[i] if person_keypoints and i < len(person_keypoints) else None
            if kps is None:
                continue

            for wrist_idx in (LEFT_WRIST_IDX, RIGHT_WRIST_IDX):
                wrist = get_valid_kp(kps, wrist_idx, conf_thresh=0.3)
                if wrist is None:
                    continue
                dist = np.hypot(wrist[0] - phone_cx, wrist[1] - phone_cy)
                if dist < self._hand_dist:
                    score = 1.0 - (dist / self._hand_dist)
                    if score > best_score:
                        best_track_id = track_id
                        best_method = "hand_proximity"
                        best_score = score

        return best_track_id, best_method


def _raw_to_phone(raw: dict) -> PhoneDetection:
    bbox_data = raw["bbox"]
    return PhoneDetection(
        bbox=BBox(*bbox_data),
        confidence=raw["confidence"],
    )
