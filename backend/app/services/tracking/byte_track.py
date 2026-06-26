import logging

import numpy as np

from app.services.tracking.kalman_filter import KalmanFilter
from app.services.tracking.matching import iou_distance, linear_assignment
from app.services.tracking.models import STrack, TrackResult, TrackState

logger = logging.getLogger(__name__)


class ByteTrack:
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        max_time_lost: int = 30,
        min_confirmed_frames: int = 3,
    ) -> None:
        self._track_thresh = track_thresh
        self._track_low_thresh = track_low_thresh
        self._new_track_thresh = new_track_thresh
        self._match_thresh = match_thresh
        self._max_time_lost = max_time_lost
        self._min_confirmed_frames = min_confirmed_frames

        self._next_id = 1
        self._tracks: list[STrack] = []
        self._frame_count = 0
        self._kf = KalmanFilter()

    @property
    def tracks(self) -> list[STrack]:
        return [t for t in self._tracks if not t.is_removed]

    @property
    def active_tracks(self) -> list[STrack]:
        return self.tracks

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0

    def update(self, detections: list[dict]) -> TrackResult:
        self._frame_count += 1

        if not detections:
            self._age_all_tracks()
            self._remove_lost_tracks()
            return TrackResult(tracks=self.active_tracks, frame_index=self._frame_count)

        scores = np.array([d["confidence"] for d in detections], dtype=float)
        class_ids = [d.get("class_id", 0) for d in detections]

        dets_xyxy = np.array([d["bbox"] for d in detections], dtype=float)
        if dets_xyxy.ndim == 1:
            dets_xyxy = dets_xyxy.reshape(1, -1)

        high_mask = scores >= self._track_thresh
        low_mask = (scores >= self._track_low_thresh) & ~high_mask

        high_dets = dets_xyxy[high_mask]
        high_scores = scores[high_mask]
        low_dets = dets_xyxy[low_mask]
        low_scores = scores[low_mask]
        high_class_ids = [class_ids[i] for i, m in enumerate(high_mask) if m]

        self._predict_tracks()

        active = self.active_tracks

        # Stage 1: match all active tracks with high-score detections
        high_matches = self._match(active, high_dets)
        matched_high_dets: set[int] = set()

        for ti, di in high_matches:
            self._update_track(active[ti], high_dets[di], high_scores[di])
            matched_high_dets.add(di)

        # Stage 2: match unmatched active tracks with low-score detections
        unmatched_tracks = [
            t for i, t in enumerate(active) if i not in {m[0] for m in high_matches}
        ]
        low_matches = self._match(unmatched_tracks, low_dets)

        for ti, di in low_matches:
            self._update_track(unmatched_tracks[ti], low_dets[di], low_scores[di])

        # Stage 3: create new tracks from unmatched high-score detections
        unmatched_high = [
            i for i in range(len(high_dets)) if i not in matched_high_dets
        ]
        self._create_new_tracks(
            [high_dets[i] for i in unmatched_high],
            [high_scores[i] for i in unmatched_high],
            [high_class_ids[i] for i in unmatched_high],
        )

        # Age all tracks not updated this frame
        updated_indices = {id(t) for t in active if t.frame_since_update == 0}
        for track in self._tracks:
            if id(track) not in updated_indices and not track.is_removed:
                track.frame_since_update += 1

        self._remove_lost_tracks()
        return TrackResult(tracks=self.active_tracks, frame_index=self._frame_count)

    def _predict_tracks(self) -> None:
        for track in self._tracks:
            if track.is_removed:
                continue
            if track.mean is not None and track.covariance is not None:
                mean, cov = self._kf.predict(track.mean, track.covariance)
                track.mean = mean
                track.covariance = cov

    def _match(
        self, tracks: list[STrack], dets: np.ndarray
    ) -> list[tuple[int, int]]:
        if not tracks or len(dets) == 0:
            return []
        cost = iou_distance(tracks, dets)
        matches = linear_assignment(cost)
        return [
            (ti, di)
            for ti, di in matches
            if cost[ti, di] <= (1.0 - self._match_thresh)
        ]

    def _update_track(self, track: STrack, det_xyxy: np.ndarray, score: float) -> None:
        tlwh = np.array(
            [
                det_xyxy[0],
                det_xyxy[1],
                det_xyxy[2] - det_xyxy[0],
                det_xyxy[3] - det_xyxy[1],
            ],
            dtype=float,
        )
        track.bbox = tlwh
        track.confidence = score
        track.frame_since_update = 0

        history = track.bbox_history
        history.append(tlwh)
        if len(history) > track.max_history:
            history.pop(0)

        if track.mean is not None and track.covariance is not None:
            cx = tlwh[0] + tlwh[2] / 2
            cy = tlwh[1] + tlwh[3] / 2
            s = max(tlwh[2] * tlwh[3], 1e-6)
            r = max(tlwh[2] / tlwh[3] if tlwh[3] > 0 else 1.0, 1e-6)
            z = np.array([cx, cy, s, r], dtype=float)
            mean, cov = self._kf.update(track.mean, track.covariance, z)
            track.mean = mean
            track.covariance = cov

        if track.state == TrackState.NEW:
            age = self._frame_count - track.start_frame
            if age >= self._min_confirmed_frames:
                track.mark_confirmed()

    def _create_new_tracks(
        self, dets: list[np.ndarray], scores: list[float], class_ids: list[int]
    ) -> None:
        for det, score, cid in zip(dets, scores, class_ids, strict=True):
            if score < self._new_track_thresh:
                continue
            tlwh = np.array(
                [det[0], det[1], det[2] - det[0], det[3] - det[1]], dtype=float
            )
            cx = tlwh[0] + tlwh[2] / 2
            cy = tlwh[1] + tlwh[3] / 2
            s = max(tlwh[2] * tlwh[3], 1e-6)
            r = max(tlwh[2] / tlwh[3] if tlwh[3] > 0 else 1.0, 1e-6)
            z = np.array([cx, cy, s, r], dtype=float)
            mean, cov = self._kf.initiate(z)

            track = STrack(
                track_id=self._next_id,
                mean=mean,
                covariance=cov,
                bbox=tlwh,
                confidence=score,
                class_id=cid,
                start_frame=self._frame_count,
            )
            self._next_id += 1
            self._tracks.append(track)

    def _age_all_tracks(self) -> None:
        for track in self._tracks:
            if not track.is_removed:
                track.frame_since_update += 1

    def _remove_lost_tracks(self) -> None:
        dead: list[STrack] = []
        for track in self._tracks:
            if track.is_removed:
                continue
            if track.frame_since_update > self._max_time_lost:
                track.mark_removed()
                dead.append(track)
            elif track.frame_since_update > 1 and track.is_confirmed:
                track.mark_lost()
        for tr in dead:
            self._tracks.remove(tr)
