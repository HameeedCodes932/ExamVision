from app.services.detection.models import BBox
from app.services.phone.models import PhoneDetection
from app.services.phone.phone_detector import (
    PhoneDetector,
    _raw_to_phone,
)


def _make_raw_phone(
    bbox: list[float],
    confidence: float = 0.8,
    class_id: int = 67,
    class_name: str = "cell phone",
) -> dict:
    return {"bbox": bbox, "confidence": confidence, "class_id": class_id, "class_name": class_name}


def _make_person(
    bbox: list[float],
    track_id: int = 1,
    confidence: float = 0.9,
) -> dict:
    return {
        "bbox": bbox,
        "confidence": confidence,
        "class_id": 0,
        "class_name": "person",
        "track_id": track_id,
    }


def _make_kps(
    left_wrist: tuple[float, float] | None = None,
    right_wrist: tuple[float, float] | None = None,
    conf: float = 0.9,
) -> list[tuple[float, float, float]]:
    kps = [(0.0, 0.0, 0.0)] * 17
    if left_wrist:
        kps[9] = (*left_wrist, conf)
    if right_wrist:
        kps[10] = (*right_wrist, conf)
    return kps


class TestPhoneDetectionModel:
    def test_unassigned(self) -> None:
        pd = PhoneDetection(bbox=BBox(0, 0, 10, 10), confidence=0.8)
        assert pd.is_assigned is False
        assert pd.assignment == "unassigned"
        assert pd.student_track_id is None
        assert pd.assignment_method is None

    def test_assigned(self) -> None:
        pd = PhoneDetection(
            bbox=BBox(0, 0, 10, 10),
            confidence=0.8,
            student_track_id=5,
            assignment_method="iou",
        )
        assert pd.is_assigned is True
        assert pd.assignment == "student_5"
        assert pd.student_track_id == 5
        assert pd.assignment_method == "iou"

    def test_to_dict_unassigned(self) -> None:
        pd = PhoneDetection(bbox=BBox(10, 20, 110, 120), confidence=0.85)
        d = pd.to_dict()
        assert d["bbox"] == [10, 20, 110, 120]
        assert d["confidence"] == 0.85
        assert d["student_track_id"] is None
        assert d["assignment"] == "unassigned"


class TestRawToPhone:
    def test_converts_raw(self) -> None:
        raw = _make_raw_phone([10, 20, 110, 120], confidence=0.85)
        phone = _raw_to_phone(raw)
        assert phone.bbox.x1 == 10
        assert phone.bbox.y1 == 20
        assert phone.bbox.x2 == 110
        assert phone.bbox.y2 == 120
        assert phone.confidence == 0.85
        assert phone.student_track_id is None


class TestFindBestMatch:
    def test_assigns_by_iou(self) -> None:
        phone = PhoneDetection(bbox=BBox(0, 0, 100, 50), confidence=0.9)
        persons = [_make_person([0, 0, 100, 200], track_id=3)]
        detector = PhoneDetector(iou_assign_threshold=0.05)
        tid, method = detector._find_best_match(phone, persons, None)
        assert tid == 3
        assert method == "iou"

    def test_no_assignment_no_overlap(self) -> None:
        phone = PhoneDetection(bbox=BBox(500, 500, 510, 510), confidence=0.9)
        persons = [_make_person([0, 0, 100, 200], track_id=3)]
        detector = PhoneDetector()
        tid, method = detector._find_best_match(phone, persons, None)
        assert tid is None
        assert method is None

    def test_assigns_by_hand_proximity(self) -> None:
        phone = PhoneDetection(bbox=BBox(200, 100, 220, 120), confidence=0.9)
        persons = [_make_person([0, 0, 100, 200], track_id=3)]
        kps_list = [_make_kps(left_wrist=(210, 110))]
        detector = PhoneDetector(hand_distance_threshold=150)
        tid, method = detector._find_best_match(phone, persons, kps_list)
        assert tid == 3
        assert method == "hand_proximity"

    def test_prefers_iou_over_hand(self) -> None:
        phone = PhoneDetection(bbox=BBox(50, 50, 150, 150), confidence=0.9)
        persons = [_make_person([0, 0, 200, 200], track_id=3)]
        kps_list = [_make_kps(left_wrist=(210, 110))]
        detector = PhoneDetector(iou_assign_threshold=0.05)
        tid, method = detector._find_best_match(phone, persons, kps_list)
        assert tid == 3
        assert method == "iou"

    def test_no_person_keypoints_falls_back(self) -> None:
        phone = PhoneDetection(bbox=BBox(200, 100, 220, 120), confidence=0.9)
        persons = [_make_person([0, 0, 100, 200], track_id=3)]
        detector = PhoneDetector()
        tid, method = detector._find_best_match(phone, persons, None)
        assert tid is None
        assert method is None

    def test_no_track_id_skipped(self) -> None:
        phone = PhoneDetection(bbox=BBox(80, 80, 120, 120), confidence=0.9)
        persons = [{"bbox": [0, 0, 100, 200], "confidence": 0.9, "class_id": 0}]
        detector = PhoneDetector()
        tid, method = detector._find_best_match(phone, persons, None)
        assert tid is None
        assert method is None

    def test_multiple_persons_best_wins(self) -> None:
        phone = PhoneDetection(bbox=BBox(180, 80, 220, 120), confidence=0.9)
        persons = [
            _make_person([0, 0, 100, 200], track_id=1),
            _make_person([150, 0, 250, 200], track_id=2),
        ]
        detector = PhoneDetector(iou_assign_threshold=0.05)
        tid, method = detector._find_best_match(phone, persons, None)
        assert tid == 2
        assert method == "iou"
