import numpy as np

from app.services.tracking.byte_track import ByteTrack
from app.services.tracking.kalman_filter import KalmanFilter
from app.services.tracking.matching import iou_bbox, linear_assignment


def _det(bbox: list[float], conf: float = 0.7, cid: int = 0) -> dict:
    return {"bbox": bbox, "confidence": conf, "class_id": cid}


class TestIOU:
    def test_overlap(self) -> None:
        b1 = np.array([0, 0, 10, 10])
        b2 = np.array([5, 5, 15, 15])
        assert iou_bbox(b1, b2) > 0

    def test_no_overlap(self) -> None:
        b1 = np.array([0, 0, 10, 10])
        b2 = np.array([20, 20, 30, 30])
        assert iou_bbox(b1, b2) == 0.0

    def test_identical(self) -> None:
        b1 = np.array([0, 0, 10, 10])
        assert iou_bbox(b1, b1) == 1.0


class TestHungarian:
    def test_square(self) -> None:
        cost = np.array([[0.1, 0.9], [0.8, 0.2]])
        matches = linear_assignment(cost)
        assert len(matches) == 2

    def test_rectangular(self) -> None:
        cost = np.array([[0.1, 0.9, 0.7], [0.8, 0.2, 0.6]])
        matches = linear_assignment(cost)
        assert len(matches) == 2


class TestKalmanFilter:
    def test_initiate(self) -> None:
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([50, 50, 1000, 1.0]))
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)

    def test_predict_cycle(self) -> None:
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([50, 50, 1000, 1.0]))
        mean2, cov2 = kf.predict(mean, cov)
        assert mean2.shape == (8,)
        assert cov2.shape == (8, 8)

    def test_update_cycle(self) -> None:
        kf = KalmanFilter()
        mean, cov = kf.initiate(np.array([50, 50, 1000, 1.0]))
        mean2, cov2 = kf.predict(mean, cov)
        mean3, cov3 = kf.update(mean2, cov2, np.array([55, 55, 1100, 1.0]))
        assert mean3.shape == (8,)
        assert cov3.shape == (8, 8)


class TestByteTrack:
    def test_no_detections(self) -> None:
        tracker = ByteTrack()
        result = tracker.update([])
        assert len(result.tracks) == 0
        assert result.frame_index == 1

    def test_new_track_creation(self) -> None:
        tracker = ByteTrack(new_track_thresh=0.1)
        result = tracker.update([_det([0, 0, 50, 50], conf=0.9)])
        assert len(result.tracks) == 1
        assert result.tracks[0].track_id == 1
        assert result.tracks[0].is_confirmed is False

    def test_track_persistence(self) -> None:
        tracker = ByteTrack(new_track_thresh=0.1, min_confirmed_frames=1)
        det = _det([0, 0, 50, 50], conf=0.9)
        result1 = tracker.update([det])
        tid = result1.tracks[0].track_id
        tracker.update([det])
        result3 = tracker.update([det])
        assert len(result3.tracks) == 1
        assert result3.tracks[0].track_id == tid
        assert result3.tracks[0].is_confirmed is True

    def test_multiple_tracks(self) -> None:
        tracker = ByteTrack(new_track_thresh=0.1, min_confirmed_frames=1)
        dets = [
            _det([0, 0, 50, 50], conf=0.9),
            _det([100, 100, 150, 150], conf=0.8),
        ]
        result = tracker.update(dets)
        assert len(result.tracks) == 2
        tids = {t.track_id for t in result.tracks}
        assert len(tids) == 2

    def test_track_id_consistency(self) -> None:
        tracker = ByteTrack(new_track_thresh=0.1, min_confirmed_frames=1, match_thresh=0.5)
        det = _det([0, 0, 50, 50], conf=0.9)
        r1 = tracker.update([det])
        tid1 = r1.tracks[0].track_id
        det2 = _det([2, 2, 52, 52], conf=0.9)
        r2 = tracker.update([det2])
        assert len(r2.tracks) == 1
        assert r2.tracks[0].track_id == tid1

    def test_low_confidence_matching(self) -> None:
        tracker = ByteTrack(
            track_thresh=0.5, track_low_thresh=0.1, new_track_thresh=0.1,
            min_confirmed_frames=1,
        )
        det_high = _det([0, 0, 50, 50], conf=0.9)
        r1 = tracker.update([det_high])
        tid = r1.tracks[0].track_id
        det_low = _det([5, 5, 55, 55], conf=0.3)
        r2 = tracker.update([det_low])
        assert len(r2.tracks) == 1
        assert r2.tracks[0].track_id == tid

    def test_track_is_lost_then_removed(self) -> None:
        tracker = ByteTrack(new_track_thresh=0.1, max_time_lost=3)
        det = _det([0, 0, 50, 50], conf=0.9)
        tracker.update([det])
        tracker.update([])
        tracker.update([])
        tracker.update([])
        result = tracker.update([])
        assert len(result.tracks) == 0

    def test_low_conf_not_new_track(self) -> None:
        tracker = ByteTrack(new_track_thresh=0.7)
        result = tracker.update([_det([0, 0, 50, 50], conf=0.5)])
        assert len(result.tracks) == 0

    def test_reset(self) -> None:
        tracker = ByteTrack()
        tracker.update([_det([0, 0, 50, 50], conf=0.9)])
        tracker.reset()
        result = tracker.update([_det([0, 0, 50, 50], conf=0.9)])
        assert result.tracks[0].track_id == 1
