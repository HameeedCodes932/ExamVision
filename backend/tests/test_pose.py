
from app.services.pose.keypoints import (
    COCO_KEYPOINTS,
    SKELETON,
    keypoint_name,
    keypoints_to_objects,
)
from app.services.pose.utils import (
    compute_skeleton,
    estimate_head_direction,
    filter_keypoints,
    get_angle,
    is_hand_raised,
    is_standing,
)


def _kps(nose=(100, 100), leye=(95, 95), reye=(105, 95), lear=(90, 100), rear=(110, 100),
          lsho=(80, 130), rsho=(120, 130), lelb=(70, 170), relb=(130, 170),
          lwri=(60, 210), rwri=(140, 210), lhip=(85, 200), rhip=(115, 200),
          lknee=(85, 260), rknee=(115, 260), lank=(85, 320), rank=(115, 320),
          conf=0.9):
    pts = [(0, 0, 0)] * 17
    vals = [nose, leye, reye, lear, rear, lsho, rsho, lelb, relb, lwri, rwri,
            lhip, rhip, lknee, rknee, lank, rank]
    for i, v in enumerate(vals):
        pts[i] = (v[0], v[1], conf)
    return pts


class TestKeypointConstants:
    def test_coco_count(self) -> None:
        assert len(COCO_KEYPOINTS) == 17

    def test_skeleton_count(self) -> None:
        assert len(SKELETON) == 18

    def test_keypoint_name(self) -> None:
        assert keypoint_name(0) == "nose"
        assert keypoint_name(5) == "left_shoulder"

    def test_keypoint_name_unknown(self) -> None:
        assert "unknown" in keypoint_name(99)


class TestKeypointObject:
    def test_from_raw(self) -> None:
        raw = [(100, 200, 0.9), (110, 210, 0.8)]
        objs = keypoints_to_objects(raw)
        assert len(objs) == 2
        assert objs[0].x == 100
        assert objs[0].y == 200
        assert objs[0].confidence == 0.9
        assert objs[0].name == "nose"

    def test_visibility(self) -> None:
        raw = [(100, 200, 0.9), (110, 210, 0.3)]
        objs = keypoints_to_objects(raw)
        assert objs[0].is_visible is True
        assert objs[1].is_visible is False


class TestFilterKeypoints:
    def test_filters_by_confidence(self) -> None:
        raw = [(100, 200, 0.9), (110, 210, 0.3), (120, 220, 0.7)]
        filtered = filter_keypoints(raw, conf_thresh=0.5)
        assert len(filtered) == 2

    def test_empty_input(self) -> None:
        assert filter_keypoints([]) == []


class TestGetAngle:
    def test_right_angle(self) -> None:
        angle = get_angle((0, 0), (0, 1), (1, 1))
        assert abs(angle - 90) < 1.0

    def test_straight_line(self) -> None:
        angle = get_angle((0, 0), (1, 1), (2, 2))
        assert angle < 1.0 or angle > 179.0

    def test_zero_leg(self) -> None:
        angle = get_angle((0, 0), (0, 0), (1, 1))
        assert angle == 0.0


class TestIsStanding:
    def test_standing(self) -> None:
        kps = _kps(nose=(100, 50), lhip=(85, 180), rhip=(115, 180),
                    lank=(85, 350), rank=(115, 350))
        assert is_standing(kps) is True

    def test_sitting(self) -> None:
        kps = _kps(nose=(100, 50), lhip=(85, 100), rhip=(115, 100),
                    lank=(85, 110), rank=(115, 110))
        assert is_standing(kps, height_ratio_thresh=1.5) is False

    def test_no_keypoints(self) -> None:
        kps = [(0, 0, 0)] * 17
        assert is_standing(kps) is False


class TestIsHandRaised:
    def test_left_hand_raised(self) -> None:
        kps = _kps(lwri=(80, 110), lelb=(85, 120), lsho=(80, 130))
        assert is_hand_raised(kps, "left") is True

    def test_left_hand_down(self) -> None:
        kps = _kps(lwri=(80, 200), lelb=(85, 170), lsho=(80, 130))
        assert is_hand_raised(kps, "left") is False

    def test_right_hand_raised(self) -> None:
        kps = _kps(rwri=(120, 110), relb=(115, 120), rsho=(120, 130))
        assert is_hand_raised(kps, "right") is True

    def test_no_wrist(self) -> None:
        kps = _kps(rwri=(0, 0), conf=0.0)
        assert is_hand_raised(kps, "right") is False


class TestEstimateHeadDirection:
    def test_forward(self) -> None:
        kps = _kps(nose=(100, 100), leye=(93, 95), reye=(107, 95))
        direction = estimate_head_direction(kps)
        assert direction == "forward"

    def test_back(self) -> None:
        kps = _kps(nose=(100, 100), leye=(100, 100), reye=(100, 100), conf=0.0)
        direction = estimate_head_direction(kps)
        assert direction in ("forward", "unknown")

    def test_unknown(self) -> None:
        kps = _kps(nose=(0, 0), conf=0.0)
        direction = estimate_head_direction(kps)
        assert direction == "unknown"


class TestComputeSkeleton:
    def test_returns_all_parts(self) -> None:
        kps = _kps()
        skeleton = compute_skeleton(kps, conf_thresh=0.5)
        assert "nose" in skeleton
        assert "left_shoulder" in skeleton
        assert skeleton["nose"]["visible"] is True

    def test_low_confidence_hidden(self) -> None:
        kps = _kps(conf=0.1)
        skeleton = compute_skeleton(kps, conf_thresh=0.5)
        assert skeleton["nose"]["visible"] is False
