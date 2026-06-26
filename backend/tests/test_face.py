from app.services.face.models import FaceDetection, HeadPose
from app.services.face.utils import (
    classify_head_direction,
    get_eye_aspect_ratio,
    get_mouth_aspect_ratio,
    is_face_visible,
)


def _make_landmarks(num: int = 468) -> list[tuple[float, float, float]]:
    return [(i * 0.5, i * 0.3, i * -0.01) for i in range(num)]


class TestFaceDetectionModel:
    def test_bbox_properties(self) -> None:
        det = FaceDetection(bbox=(10, 20, 110, 170), confidence=0.9)
        assert det.x1 == 10
        assert det.y1 == 20
        assert det.x2 == 110
        assert det.y2 == 170
        assert det.width == 100
        assert det.height == 150

    def test_no_landmarks(self) -> None:
        det = FaceDetection(bbox=(0, 0, 50, 50), confidence=0.8)
        assert det.num_landmarks == 0

    def test_with_landmarks(self) -> None:
        det = FaceDetection(
            bbox=(0, 0, 50, 50),
            confidence=0.8,
            landmarks=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        )
        assert det.num_landmarks == 2

    def test_to_dict(self) -> None:
        det = FaceDetection(
            bbox=(0, 0, 50, 50),
            confidence=0.85,
            landmarks=[(1.0, 2.0, 0.5)],
        )
        d = det.to_dict()
        assert d["bbox"] == [0, 0, 50, 50]
        assert d["confidence"] == 0.85
        assert d["landmarks"] == [[1.0, 2.0, 0.5]]


class TestHeadPoseModel:
    def test_forward(self) -> None:
        pose = HeadPose(yaw=0, pitch=0, roll=0)
        assert pose.direction == "FORWARD"

    def test_left(self) -> None:
        pose = HeadPose(yaw=30, pitch=0, roll=0)
        assert pose.direction == "LEFT"

    def test_right(self) -> None:
        pose = HeadPose(yaw=-30, pitch=0, roll=0)
        assert pose.direction == "RIGHT"

    def test_down(self) -> None:
        pose = HeadPose(yaw=0, pitch=30, roll=0)
        assert pose.direction == "DOWN"

    def test_up(self) -> None:
        pose = HeadPose(yaw=0, pitch=-30, roll=0)
        assert pose.direction == "UP"

    def test_left_edge(self) -> None:
        pose = HeadPose(yaw=20.1, pitch=0, roll=0)
        assert pose.direction == "LEFT"

    def test_forward_edge(self) -> None:
        pose = HeadPose(yaw=19.9, pitch=19.9, roll=0)
        assert pose.direction == "FORWARD"

    def test_to_dict(self) -> None:
        pose = HeadPose(yaw=15.123, pitch=-5.456, roll=2.789)
        d = pose.to_dict()
        assert d["yaw"] == 15.12
        assert d["pitch"] == -5.46
        assert d["roll"] == 2.79
        assert d["direction"] == "FORWARD"


class TestClassifyHeadDirection:
    def test_forward(self) -> None:
        assert classify_head_direction(0, 0) == "FORWARD"

    def test_left(self) -> None:
        assert classify_head_direction(30, 0) == "LEFT"

    def test_right(self) -> None:
        assert classify_head_direction(-30, 0) == "RIGHT"

    def test_down(self) -> None:
        assert classify_head_direction(0, 30) == "DOWN"

    def test_up(self) -> None:
        assert classify_head_direction(0, -30) == "UP"


class TestEyeAspectRatio:
    def test_computes_ear(self) -> None:
        lmks = _make_landmarks(468)
        ear = get_eye_aspect_ratio(lmks)
        assert isinstance(ear, float)
        assert ear >= 0

    def test_not_enough_landmarks(self) -> None:
        lmks = [(0.0, 0.0, 0.0) for _ in range(10)]
        assert get_eye_aspect_ratio(lmks) == 0.0


class TestMouthAspectRatio:
    def test_computes_mar(self) -> None:
        lmks = _make_landmarks(468)
        mar = get_mouth_aspect_ratio(lmks)
        assert isinstance(mar, float)
        assert mar >= 0

    def test_not_enough_landmarks(self) -> None:
        lmks = [(0.0, 0.0, 0.0) for _ in range(10)]
        assert get_mouth_aspect_ratio(lmks) == 0.0


class TestIsFaceVisible:
    def test_visible(self) -> None:
        lmks = _make_landmarks(468)
        assert is_face_visible(lmks) is True

    def test_not_visible(self) -> None:
        lmks = [(0.0, 0.0, 0.0) for _ in range(100)]
        assert is_face_visible(lmks) is False
