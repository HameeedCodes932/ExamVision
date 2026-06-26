import time
from collections import deque
from collections.abc import Sequence
from typing import Any

from app.services.behaviour.models import BehaviourEvent, EventType, Severity
from app.services.behaviour.utils import (
    BODY_TWIST_COUNT_THRESHOLD,
    HEAD_TURN_COUNT_THRESHOLD,
    HEAD_TURN_WINDOW,
    MAR_TALKING_THRESHOLD,
)

LOOKING_LEFT_DURATION = 5.0
HEAD_DOWN_DURATION = 30.0

State = dict[str, Any]


class BehaviourAnalyser:
    def __init__(self, window_seconds: int = 30) -> None:
        self._window_seconds = window_seconds
        self._buffers: dict[int, deque[tuple[float, State]]] = {}
        self._active_events: dict[int, dict[str, bool]] = {}

    def _ensure_buffer(self, track_id: int) -> deque[tuple[float, State]]:
        if track_id not in self._buffers:
            self._buffers[track_id] = deque()
        return self._buffers[track_id]

    def _prune(self, track_id: int, now: float) -> None:
        buf = self._buffers.get(track_id)
        if buf is None:
            return
        cutoff = now - self._window_seconds
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def _window(self, track_id: int, now: float) -> list[tuple[float, State]]:
        self._prune(track_id, now)
        buf = self._buffers.get(track_id)
        return list(buf) if buf else []

    def _append(self, track_id: int, ts: float, state: State) -> None:
        buf = self._ensure_buffer(track_id)
        buf.append((ts, state))

    def _transitions(
        self, window: Sequence[tuple[float, State]], key: str
    ) -> int:
        count = 0
        prev: str | None = None
        for _, state in window:
            val = state.get(key)
            if isinstance(val, str) and prev is not None and val != prev:
                count += 1
            if isinstance(val, str):
                prev = val
        return count

    def _sustained_duration(
        self,
        window: Sequence[tuple[float, State]],
        key: str,
        value: str,
        required_duration: float,
        ratio_threshold: float = 0.8,
    ) -> float | None:
        if not window:
            return None
        latest_ts = window[-1][0]
        if window[-1][1].get(key) != value:
            return None
        cutoff = latest_ts - required_duration
        relevant = [(ts, s) for ts, s in window if ts >= cutoff]
        if len(relevant) < 2:
            return None
        matches = sum(1 for _, s in relevant if s.get(key) == value)
        ratio = matches / len(relevant)
        span = latest_ts - relevant[0][0]
        if ratio >= ratio_threshold and span >= required_duration * 0.8:
            return latest_ts - relevant[0][0]
        return None

    def _is_active(self, track_id: int, event_type: str) -> bool:
        return self._active_events.get(track_id, {}).get(event_type, False)

    def _set_active(self, track_id: int, event_type: str, active: bool) -> None:
        if track_id not in self._active_events:
            self._active_events[track_id] = {}
        self._active_events[track_id][event_type] = active

    def _evaluate_looking_left(
        self, track_id: int, window: list[tuple[float, State]], now: float
    ) -> BehaviourEvent | None:
        sustained = self._sustained_duration(
            window, "head_direction", "left", LOOKING_LEFT_DURATION
        )
        if sustained is not None and not self._is_active(track_id, "looking_left"):
            self._set_active(track_id, "looking_left", True)
            return BehaviourEvent(
                track_id=track_id,
                event_type=EventType.LOOKING_LEFT,
                severity=Severity.HIGH,
                timestamp=now,
                details=f"Looking left for {sustained:.1f}s",
                confidence=min(sustained / LOOKING_LEFT_DURATION, 1.0),
            )
        if sustained is None:
            self._set_active(track_id, "looking_left", False)
        return None

    def _evaluate_phone_detected(
        self, track_id: int, state: State, now: float
    ) -> BehaviourEvent | None:
        phone = state.get("phone_detected", False)
        if phone and not self._is_active(track_id, "phone_detected"):
            self._set_active(track_id, "phone_detected", True)
            return BehaviourEvent(
                track_id=track_id,
                event_type=EventType.PHONE_DETECTED,
                severity=Severity.CRITICAL,
                timestamp=now,
                details="Phone detected during exam",
                confidence=1.0,
            )
        if not phone:
            self._set_active(track_id, "phone_detected", False)
        return None

    def _evaluate_standing(
        self, track_id: int, state: State, now: float
    ) -> BehaviourEvent | None:
        standing = state.get("is_standing", False)
        if standing and not self._is_active(track_id, "standing"):
            self._set_active(track_id, "standing", True)
            return BehaviourEvent(
                track_id=track_id,
                event_type=EventType.STANDING,
                severity=Severity.MEDIUM,
                timestamp=now,
                details="Student is standing",
                confidence=1.0,
            )
        if not standing:
            self._set_active(track_id, "standing", False)
        return None

    def _evaluate_head_down(
        self, track_id: int, window: list[tuple[float, State]], now: float
    ) -> BehaviourEvent | None:
        sustained = self._sustained_duration(
            window, "head_direction", "down", HEAD_DOWN_DURATION
        )
        if sustained is not None and not self._is_active(track_id, "head_down"):
            self._set_active(track_id, "head_down", True)
            return BehaviourEvent(
                track_id=track_id,
                event_type=EventType.HEAD_DOWN,
                severity=Severity.HIGH,
                timestamp=now,
                details=f"Head down for {sustained:.1f}s — possible phone usage",
                confidence=min(sustained / HEAD_DOWN_DURATION, 1.0),
            )
        if sustained is None:
            self._set_active(track_id, "head_down", False)
        return None

    def _evaluate_repeated_head_turns(
        self, track_id: int, window: list[tuple[float, State]], state: State, now: float
    ) -> BehaviourEvent | None:
        head_turns = self._transitions(window, "head_direction")
        mar = state.get("mouth_aspect_ratio")
        if head_turns >= HEAD_TURN_COUNT_THRESHOLD and not self._is_active(
            track_id, "repeated_head_turns"
        ):
            details = f"Repeated head direction changes ({head_turns} in {HEAD_TURN_WINDOW:.0f}s)"
            if mar is not None and mar > MAR_TALKING_THRESHOLD:
                details += " with mouth movement detected"
            self._set_active(track_id, "repeated_head_turns", True)
            return BehaviourEvent(
                track_id=track_id,
                event_type=EventType.REPEATED_HEAD_TURNS,
                severity=Severity.MEDIUM,
                timestamp=now,
                details=details,
                confidence=min(head_turns / HEAD_TURN_COUNT_THRESHOLD, 1.0),
            )
        if head_turns < HEAD_TURN_COUNT_THRESHOLD:
            self._set_active(track_id, "repeated_head_turns", False)
        return None

    def _evaluate_left_seat(
        self, track_id: int, window: list[tuple[float, State]], state: State, now: float
    ) -> BehaviourEvent | None:
        if len(window) < 2:
            return None
        prev_state = window[-2][1]
        prev_seated = prev_state.get("is_seated", True)
        curr_seated = state.get("is_seated", True)
        if prev_seated and not curr_seated and not self._is_active(track_id, "student_left_seat"):
            self._set_active(track_id, "student_left_seat", True)
            return BehaviourEvent(
                track_id=track_id,
                event_type=EventType.STUDENT_LEFT_SEAT,
                severity=Severity.CRITICAL,
                timestamp=now,
                details="Student has left their seat",
                confidence=1.0,
            )
        if not curr_seated:
            self._set_active(track_id, "student_left_seat", True)
        else:
            self._set_active(track_id, "student_left_seat", False)
        return None

    def _evaluate_body_twisting(
        self, track_id: int, window: list[tuple[float, State]], now: float
    ) -> BehaviourEvent | None:
        twists = self._transitions(window, "body_orientation")
        if twists >= BODY_TWIST_COUNT_THRESHOLD and not self._is_active(
            track_id, "body_twisting"
        ):
            self._set_active(track_id, "body_twisting", True)
            return BehaviourEvent(
                track_id=track_id,
                event_type=EventType.BODY_TWISTING,
                severity=Severity.MEDIUM,
                timestamp=now,
                details=f"Frequent body twisting detected ({twists} changes)",
                confidence=min(twists / BODY_TWIST_COUNT_THRESHOLD, 1.0),
            )
        if twists < BODY_TWIST_COUNT_THRESHOLD:
            self._set_active(track_id, "body_twisting", False)
        return None

    def analyse(self, track_id: int, state: State) -> list[BehaviourEvent]:
        ts = state.get("timestamp", time.time())
        self._append(track_id, ts, state)
        window = self._window(track_id, ts)
        events: list[BehaviourEvent] = []
        for evaluator in [
            self._evaluate_looking_left,
            self._evaluate_phone_detected,
            self._evaluate_standing,
            self._evaluate_head_down,
            self._evaluate_repeated_head_turns,
            self._evaluate_left_seat,
            self._evaluate_body_twisting,
        ]:
            if evaluator in (
                self._evaluate_phone_detected,
                self._evaluate_standing,
            ):
                event = evaluator(track_id, state, ts)
            elif evaluator in (self._evaluate_left_seat, self._evaluate_repeated_head_turns):
                event = evaluator(track_id, window, state, ts)
            elif evaluator in (
                self._evaluate_looking_left,
                self._evaluate_head_down,
                self._evaluate_body_twisting,
            ):
                event = evaluator(track_id, window, ts)
            else:
                event = None
            if event is not None:
                events.append(event)
        return events

    def reset(self) -> None:
        self._buffers.clear()
        self._active_events.clear()
