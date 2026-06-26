import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np

from app.services.face.models import FaceDetection, HeadPose

logger = logging.getLogger(__name__)

# 6 key facial landmark indices from MediaPipe Face Mesh (468 total)
# Used for SolvePnP head pose estimation
_KEY_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# Corresponding 3D model points (in mm) for a generic face
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip (idx 1)
        (0.0, -330.0, -65.0),  # Chin (idx 152)
        (-225.0, 170.0, -135.0),  # Left eye outer corner (idx 33)
        (225.0, 170.0, -135.0),  # Right eye outer corner (idx 263)
        (-150.0, -150.0, -125.0),  # Left mouth corner (idx 61)
        (150.0, -150.0, -125.0),  # Right mouth corner (idx 291)
    ],
    dtype=np.float64,
)


class FaceDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._face_detection: mp.solutions.face_detection.FaceDetection | None = None
        self._face_mesh: mp.solutions.face_mesh.FaceMesh | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def load(self) -> None:
        if self._face_detection is not None:
            return

        def _init() -> None:
            self._face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self._min_detection_confidence,
            )
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, _init)
        logger.info("Face detector loaded (MediaPipe Face Detection + Face Mesh)")

    async def detect(
        self,
        frame: np.ndarray,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> FaceDetection | None:
        if self._face_detection is None:
            raise RuntimeError("Face detector not loaded. Call load() first.")

        roi = self._crop_roi(frame, bbox) if bbox is not None else frame
        if roi.size == 0:
            return None

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        loop = asyncio.get_running_loop()

        def _run_detection() -> FaceDetection | None:
            det_results = self._face_detection.process(rgb)
            if det_results is None or not det_results.detections:
                return None

            det = det_results.detections[0]
            score = det.score[0]
            rel_bbox = det.location_data.relative_bounding_box

            h, w = rgb.shape[:2]
            fx = rel_bbox.xmin * w
            fy = rel_bbox.ymin * h
            fw = rel_bbox.width * w
            fh = rel_bbox.height * h
            local_bbox = (fx, fy, fx + fw, fy + fh)

            face_crop = self._crop_roi(rgb, local_bbox)
            landmarks: list[tuple[float, float, float]] = []
            if face_crop.size > 0:
                mesh_results = self._face_mesh.process(face_crop)
                if mesh_results and mesh_results.multi_face_landmarks:
                    crop_h, crop_w = face_crop.shape[:2]
                    offset_x, offset_y = local_bbox[0], local_bbox[1]
                    for lm in mesh_results.multi_face_landmarks[0].landmark:
                        lx = lm.x * crop_w + offset_x
                        ly = lm.y * crop_h + offset_y
                        landmarks.append((float(lx), float(ly), float(lm.z)))

            if bbox is not None:
                bx1, by1, _, _ = bbox
                global_bbox = (
                    local_bbox[0] + bx1,
                    local_bbox[1] + by1,
                    local_bbox[2] + bx1,
                    local_bbox[3] + by1,
                )
            else:
                global_bbox = local_bbox

            return FaceDetection(bbox=global_bbox, confidence=score, landmarks=landmarks)

        return await loop.run_in_executor(self._executor, _run_detection)

    async def estimate_head_pose(
        self,
        landmarks: list[tuple[float, float, float]],
        image_shape: tuple[int, int],
    ) -> HeadPose | None:
        if len(landmarks) < 468:
            return None

        image_points = self._get_key_image_points(landmarks)
        if image_points is None:
            return None

        focal_length = image_shape[1]
        center = (image_shape[1] / 2, image_shape[0] / 2)
        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        def _solve() -> HeadPose | None:
            success, rvec, _ = cv2.solvePnP(
                MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs
            )
            if not success:
                return None
            return self._rvec_to_head_pose(rvec)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, _solve)

    async def unload(self) -> None:
        self._face_detection = None
        self._face_mesh = None
        self._executor.shutdown(wait=False)
        logger.info("Face detector unloaded")

    @staticmethod
    def _crop_roi(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _get_key_image_points(
        landmarks: list[tuple[float, float, float]],
    ) -> np.ndarray | None:
        pts: list[tuple[float, float]] = []
        for idx in _KEY_LANDMARK_INDICES:
            if idx < len(landmarks):
                pts.append((landmarks[idx][0], landmarks[idx][1]))
        if len(pts) < 4:
            return None
        return np.array(pts, dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def _rvec_to_head_pose(rvec: np.ndarray) -> HeadPose:
        rot, _ = cv2.Rodrigues(rvec)
        yaw = math.degrees(math.atan2(rot[1][0], rot[0][0]))
        pitch = math.degrees(math.atan2(-rot[2][0], math.sqrt(rot[2][1] ** 2 + rot[2][2] ** 2)))
        roll = math.degrees(math.atan2(rot[2][1], rot[2][2]))
        return HeadPose(yaw=yaw, pitch=pitch, roll=roll)
