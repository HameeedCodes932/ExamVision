

from app.services.detection.models import (
    COCO_CLASSES,
    BBox,
    Detection,
    DetectionResult,
)


def filter_by_class(
    raw: list[dict],
    class_ids: set[int] | None = None,
    min_confidence: float = 0.0,
    min_area: int = 900,
) -> list[dict]:
    filtered: list[dict] = []
    for det in raw:
        if det["confidence"] < min_confidence:
            continue
        if class_ids is not None and det["class_id"] not in class_ids:
            continue
        bbox = det["bbox"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < min_area:
            continue
        filtered.append(det)
    return filtered


def nms(
    detections: list[dict],
    iou_threshold: float = 0.5,
) -> list[dict]:
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep: list[dict] = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        b1 = BBox(*best["bbox"])
        dets = [
            d
            for d in dets
            if b1.iou(BBox(*d["bbox"])) < iou_threshold
        ]

    return keep


def to_detection(raw: dict) -> Detection:
    return Detection(
        bbox=BBox(*raw["bbox"]),
        confidence=raw["confidence"],
        class_id=raw["class_id"],
        class_name=raw.get("class_name", COCO_CLASSES.get(raw["class_id"], "unknown")),
        keypoints=raw.get("keypoints"),
    )


def build_result(
    raw: list[dict],
    frame_shape: tuple[int, int],
    timestamp: float,
    inference_ms: float,
    person_only: bool = True,
    min_confidence: float | None = None,
    min_area: int = 900,
    apply_nms_flag: bool = True,
    iou_threshold: float = 0.5,
) -> DetectionResult:
    if person_only:
        raw = filter_by_class(raw, class_ids={0}, min_area=min_area)
    elif min_confidence is not None:
        raw = filter_by_class(raw, min_confidence=min_confidence, min_area=min_area)

    if apply_nms_flag:
        raw = nms(raw, iou_threshold=iou_threshold)

    return DetectionResult(
        detections=[to_detection(d) for d in raw],
        frame_shape=frame_shape,
        timestamp=timestamp,
        inference_ms=inference_ms,
    )
