import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np

from app.services.gaze.models import GazeVector
from app.services.gaze.utils import classify_gaze_target, compute_gaze_from_iris

logger = logging.getLogger(__name__)

# Eye corner landmark indices from MediaPipe Face Mesh (478 total with iris)
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_OUTER = 362
_RIGHT_EYE_INNER = 263

# Iris landmark indices (available when refine_landmarks=True)
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 472
_LEFT_EYE_CENTER = 476
_RIGHT_EYE_CENTER = 477


class GazeEstimator:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_gaze_yaw: float = 30.0,
        max_gaze_pitch: float = 25.0,
    ) -> None:
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._max_gaze_yaw = max_gaze_yaw
        self._max_gaze_pitch = max_gaze_pitch
        self._face_mesh: mp.solutions.face_mesh.FaceMesh | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def load(self) -> None:
        if self._face_mesh is not None:
            return

        def _init() -> None:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _init)
        logger.info("Gaze estimator loaded (MediaPipe Face Mesh with iris)")

    async def estimate(
        self,
        face_roi: np.ndarray,
        head_angles: dict[str, float] | None = None,
    ) -> GazeVector | None:
        if self._face_mesh is None:
            raise RuntimeError("Gaze estimator not loaded. Call load() first.")

        if face_roi.size == 0:
            return None

        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        head_yaw = (head_angles or {}).get("yaw", 0.0)
        head_pitch = (head_angles or {}).get("pitch", 0.0)

        loop = asyncio.get_running_loop()

        def _estimate() -> GazeVector | None:
            results = self._face_mesh.process(rgb)
            if results is None or not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            h, w = rgb.shape[:2]

            if len(landmarks) < 478:
                yaw, pitch = head_yaw, head_pitch
            else:
                yaw, pitch = self._compute_gaze_from_mesh(landmarks, w, h, head_yaw, head_pitch)

            target = classify_gaze_target(yaw, pitch)
            return GazeVector(gaze_yaw=yaw, gaze_pitch=pitch, target=target)

        return await loop.run_in_executor(self._executor, _estimate)

    async def estimate_from_landmarks(
        self,
        landmarks_468: list[tuple[float, float, float]],
        image_shape: tuple[int, int],
        head_angles: dict[str, float] | None = None,
    ) -> GazeVector | None:
        if len(landmarks_468) < 468:
            return None

        head_yaw = (head_angles or {}).get("yaw", 0.0)
        head_pitch = (head_angles or {}).get("pitch", 0.0)

        def _run() -> GazeVector:
            yaw, pitch = self._compute_gaze_from_468_landmarks(
                landmarks_468, image_shape, head_yaw, head_pitch
            )
            target = classify_gaze_target(yaw, pitch)
            return GazeVector(gaze_yaw=yaw, gaze_pitch=pitch, target=target)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _run)

    async def unload(self) -> None:
        self._face_mesh = None
        self._executor.shutdown(wait=False)
        logger.info("Gaze estimator unloaded")

    def _compute_gaze_from_mesh(
        self,
        landmarks: list,
        img_w: int,
        img_h: int,
        head_yaw: float,
        head_pitch: float,
    ) -> tuple[float, float]:
        def _to_px(idx: int) -> tuple[float, float]:
            lm = landmarks[idx]
            return lm.x * img_w, lm.y * img_h

        left_outer = _to_px(_LEFT_EYE_OUTER)
        left_inner = _to_px(_LEFT_EYE_INNER)
        right_outer = _to_px(_RIGHT_EYE_OUTER)
        right_inner = _to_px(_RIGHT_EYE_INNER)

        left_iris = _to_px(_LEFT_IRIS_CENTER)
        right_iris = _to_px(_RIGHT_IRIS_CENTER)

        yaw_l, pitch_l = compute_gaze_from_iris(
            (*left_outer, *left_inner),
            left_iris,
            head_yaw,
            head_pitch,
            self._max_gaze_yaw,
            self._max_gaze_pitch,
        )
        yaw_r, pitch_r = compute_gaze_from_iris(
            (*right_inner, *right_outer),
            right_iris,
            head_yaw,
            head_pitch,
            self._max_gaze_yaw,
            self._max_gaze_pitch,
        )

        return (yaw_l + yaw_r) / 2, (pitch_l + pitch_r) / 2

    def _compute_gaze_from_468_landmarks(
        self,
        landmarks: list[tuple[float, float, float]],
        image_shape: tuple[int, int],
        head_yaw: float,
        head_pitch: float,
    ) -> tuple[float, float]:
        h, w = image_shape

        left_outer = (landmarks[_LEFT_EYE_OUTER][0], landmarks[_LEFT_EYE_OUTER][1])
        left_inner = (landmarks[_LEFT_EYE_INNER][0], landmarks[_LEFT_EYE_INNER][1])
        right_outer = (landmarks[_RIGHT_EYE_OUTER][0], landmarks[_RIGHT_EYE_OUTER][1])
        right_inner = (landmarks[_RIGHT_EYE_INNER][0], landmarks[_RIGHT_EYE_INNER][1])

        left_iris = self._estimate_iris_from_eye_landmarks(landmarks, "left", w, h)
        right_iris = self._estimate_iris_from_eye_landmarks(landmarks, "right", w, h)

        if left_iris is None or right_iris is None:
            return head_yaw, head_pitch

        yaw_l, pitch_l = compute_gaze_from_iris(
            (*left_outer, *left_inner),
            left_iris,
            head_yaw,
            head_pitch,
            self._max_gaze_yaw,
            self._max_gaze_pitch,
        )
        yaw_r, pitch_r = compute_gaze_from_iris(
            (*right_inner, *right_outer),
            right_iris,
            head_yaw,
            head_pitch,
            self._max_gaze_yaw,
            self._max_gaze_pitch,
        )

        return (yaw_l + yaw_r) / 2, (pitch_l + pitch_r) / 2

    @staticmethod
    def _estimate_iris_from_eye_landmarks(
        landmarks: list[tuple[float, float, float]],
        side: str,
        img_w: int,
        img_h: int,
    ) -> tuple[float, float] | None:
        if side == "left":
            eye_indices = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144]
        else:
            eye_indices = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380]

        valid = []
        for idx in eye_indices:
            if idx < len(landmarks):
                valid.append((landmarks[idx][0], landmarks[idx][1]))

        if len(valid) < 4:
            return None

        cx = sum(p[0] for p in valid) / len(valid)
        cy = sum(p[1] for p in valid) / len(valid)
        return (cx, cy)
