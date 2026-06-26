from app.services.detection.post_processor import (
    build_result,
    filter_by_class,
    nms,
)


def _make_det(
    bbox: list[float], confidence: float, class_id: int = 0, class_name: str = "person"
) -> dict:
    return {"bbox": bbox, "confidence": confidence, "class_id": class_id, "class_name": class_name}


class TestFilterByClass:
    def test_filters_by_class_id(self) -> None:
        dets = [
            _make_det([0, 0, 50, 50], 0.9, class_id=0),
            _make_det([0, 0, 50, 50], 0.9, class_id=67),
        ]
        result = filter_by_class(dets, class_ids={0}, min_area=0)
        assert len(result) == 1
        assert result[0]["class_id"] == 0

    def test_filters_by_min_confidence(self) -> None:
        dets = [
            _make_det([0, 0, 50, 50], 0.3),
            _make_det([0, 0, 50, 50], 0.7),
        ]
        result = filter_by_class(dets, min_confidence=0.5, min_area=0)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.7

    def test_filters_by_min_area(self) -> None:
        dets = [
            _make_det([0, 0, 10, 10], 0.9),
            _make_det([0, 0, 50, 50], 0.9),
        ]
        result = filter_by_class(dets, min_area=900)
        assert len(result) == 1
        assert result[0]["bbox"] == [0, 0, 50, 50]

    def test_empty_input(self) -> None:
        assert filter_by_class([]) == []


class TestNMS:
    def test_removes_overlapping(self) -> None:
        overlapping = [
            _make_det([10, 10, 100, 100], 0.9),
            _make_det([15, 15, 95, 95], 0.8),
        ]
        result = nms(overlapping, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_keeps_non_overlapping(self) -> None:
        non_overlapping = [
            _make_det([0, 0, 50, 50], 0.9),
            _make_det([100, 100, 150, 150], 0.8),
        ]
        result = nms(non_overlapping, iou_threshold=0.5)
        assert len(result) == 2

    def test_sorts_by_confidence(self) -> None:
        dets = [
            _make_det([0, 0, 100, 100], 0.5),
            _make_det([5, 5, 95, 95], 0.9),
        ]
        result = nms(dets, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_empty_input(self) -> None:
        assert nms([]) == []


class TestBuildResult:
    def test_integration(self) -> None:
        raw = [
            {"bbox": [0, 0, 50, 50], "confidence": 0.9, "class_id": 0, "class_name": "person"},
            {"bbox": [200, 200, 300, 300], "confidence": 0.7, "class_id": 67,
             "class_name": "cell phone"},
        ]
        result = build_result(
            raw=raw,
            frame_shape=(480, 640),
            timestamp=123.0,
            inference_ms=15.0,
            person_only=True,
        )
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "person"
        assert result.frame_shape == (480, 640)
        assert result.inference_ms == 15.0
