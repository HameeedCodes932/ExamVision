import time

from app.services.behaviour import BehaviourEvent
from app.services.scoring.models import (
    SCORE_WEIGHTS,
    SuspicionScore,
    classify_score,
)


class SuspicionScorer:
    def __init__(self, decay_minutes: int = 5) -> None:
        self._decay_minutes = decay_minutes
        self._events: dict[int, list[tuple[float, str, float]]] = {}

    def _ensure_events(self, track_id: int) -> list[tuple[float, str, float]]:
        if track_id not in self._events:
            self._events[track_id] = []
        return self._events[track_id]

    def _weight_for(self, event_type: str) -> float:
        return SCORE_WEIGHTS.get(event_type, 10.0)

    def _decay(self, age_seconds: float) -> float:
        half_life = self._decay_minutes * 60.0
        if half_life <= 0:
            return 1.0
        return float(2.0 ** (-age_seconds / half_life))

    def _prune(self, track_id: int, now: float) -> None:
        buf = self._events.get(track_id)
        if buf is None:
            return
        threshold = now - self._decay_minutes * 60.0 * 5
        self._events[track_id] = [(ts, et, w) for ts, et, w in buf if ts >= threshold]

    def update(
        self, track_id: int, events: list[BehaviourEvent], now: float | None = None
    ) -> SuspicionScore:
        if now is None:
            now = time.time()
        buf = self._ensure_events(track_id)
        for ev in events:
            weight = self._weight_for(ev.event_type.value) * ev.confidence
            buf.append((ev.timestamp, ev.event_type.value, weight))
        self._prune(track_id, now)
        total = 0.0
        breakdown: dict[str, float] = {}
        for ts, event_type, weight in buf:
            age = now - ts
            decayed = weight * self._decay(age)
            total += decayed
            breakdown[event_type] = breakdown.get(event_type, 0) + decayed
        level = classify_score(total)
        return SuspicionScore(
            track_id=track_id,
            total=total,
            breakdown=breakdown,
            level=level,
        )

    def get_score(self, track_id: int, now: float | None = None) -> SuspicionScore | None:
        if track_id not in self._events or not self._events[track_id]:
            return None
        return self.update(track_id, [], now)

    def reset(self, track_id: int | None = None) -> None:
        if track_id is not None:
            self._events.pop(track_id, None)
        else:
            self._events.clear()
