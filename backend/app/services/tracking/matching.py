import numpy as np
from scipy.optimize import linear_sum_assignment

from app.services.tracking.models import STrack


def iou_bbox(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    ix1 = max(bbox1[0], bbox2[0])
    iy1 = max(bbox1[1], bbox2[1])
    ix2 = min(bbox1[2], bbox2[2])
    iy2 = min(bbox1[3], bbox2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area1 = max(0.0, (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
    area2 = max(0.0, (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def iou_distance(tracks: list[STrack], detections: np.ndarray) -> np.ndarray:
    cost = np.zeros((len(tracks), len(detections)), dtype=float)
    for ti, track in enumerate(tracks):
        t_tlbr = track.xyxy
        for di in range(len(detections)):
            cost[ti, di] = 1.0 - iou_bbox(t_tlbr, detections[di])
    return cost


def linear_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind, col_ind, strict=False))
