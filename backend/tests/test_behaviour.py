import time

from app.services.behaviour import BehaviourAnalyser, BehaviourEvent, EventType, Severity
from app.services.behaviour.analyser import (
    HEAD_DOWN_DURATION,
    LOOKING_LEFT_DURATION,
)
from app.services.behaviour.utils import (
    BODY_TWIST_COUNT_THRESHOLD,
    HEAD_TURN_COUNT_THRESHOLD,
    MAR_TALKING_THRESHOLD,
)


def _state(
    head_direction: str | None = "forward",
    is_standing: bool = False,
    phone_detected: bool = False,
    is_seated: bool = True,
    body_orientation: str | None = "forward",
    mouth_aspect_ratio: float | None = None,
    head_yaw: float | None = None,
    head_pitch: float | None = None,
) -> dict:
    s: dict = {
        "head_direction": head_direction,
        "is_standing": is_standing,
        "phone_detected": phone_detected,
        "is_seated": is_seated,
        "body_orientation": body_orientation,
    }
    if mouth_aspect_ratio is not None:
        s["mouth_aspect_ratio"] = mouth_aspect_ratio
    if head_yaw is not None:
        s["head_yaw"] = head_yaw
    if head_pitch is not None:
        s["head_pitch"] = head_pitch
    return s


def _feed(
    analyser: BehaviourAnalyser,
    track_id: int,
    states: list[dict],
    interval: float = 1.0,
) -> list[BehaviourEvent]:
    now = time.time()
    events: list[BehaviourEvent] = []
    for i, st in enumerate(states):
        st["timestamp"] = now - (len(states) - 1 - i) * interval
        evts = analyser.analyse(track_id, st)
        events.extend(evts)
    return events


class TestBehaviourEventModel:
    def test_to_dict(self) -> None:
        ev = BehaviourEvent(
            track_id=1,
            event_type=EventType.LOOKING_LEFT,
            severity=Severity.HIGH,
            timestamp=1000.0,
            details="looking left",
            confidence=0.8,
        )
        d = ev.to_dict()
        assert d["track_id"] == 1
        assert d["event_type"] == "looking_left"
        assert d["severity"] == "high"
        assert d["timestamp"] == 1000.0
        assert d["details"] == "looking left"
        assert d["confidence"] == 0.8

    def test_defaults(self) -> None:
        ev = BehaviourEvent(
            track_id=2,
            event_type=EventType.STANDING,
            severity=Severity.MEDIUM,
            timestamp=2000.0,
        )
        assert ev.details is None
        assert ev.confidence == 1.0

    def test_to_dict_defaults(self) -> None:
        ev = BehaviourEvent(
            track_id=2,
            event_type=EventType.STANDING,
            severity=Severity.MEDIUM,
            timestamp=2000.0,
        )
        d = ev.to_dict()
        assert d["details"] is None
        assert d["confidence"] == 1.0


class TestSeverity:
    def test_values(self) -> None:
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"


class TestEventType:
    def test_values(self) -> None:
        assert EventType.LOOKING_LEFT.value == "looking_left"
        assert EventType.PHONE_DETECTED.value == "phone_detected"
        assert EventType.STANDING.value == "standing"
        assert EventType.HEAD_DOWN.value == "head_down"
        assert EventType.REPEATED_HEAD_TURNS.value == "repeated_head_turns"
        assert EventType.STUDENT_LEFT_SEAT.value == "student_left_seat"
        assert EventType.BODY_TWISTING.value == "body_twisting"


class TestRuleLookingLeft:
    def test_fires_after_sustained_left(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(head_direction="left")] * int(LOOKING_LEFT_DURATION + 2)
        events = _feed(analyser, 1, states)
        assert any(e.event_type == EventType.LOOKING_LEFT for e in events)

    def test_does_not_fire_if_not_long_enough(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(head_direction="left")] * 2
        events = _feed(analyser, 1, states)
        assert not any(e.event_type == EventType.LOOKING_LEFT for e in events)

    def test_does_not_fire_for_forward(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(head_direction="forward")] * int(LOOKING_LEFT_DURATION + 2)
        events = _feed(analyser, 1, states)
        assert not any(e.event_type == EventType.LOOKING_LEFT for e in events)

    def test_stops_firing_after_looking_away(self) -> None:
        analyser = BehaviourAnalyser(window_seconds=60)
        states = [_state(head_direction="left")] * int(LOOKING_LEFT_DURATION + 2)
        events = _feed(analyser, 1, states)
        assert sum(1 for e in events if e.event_type == EventType.LOOKING_LEFT) == 1
        analyser.reset()
        states2 = [_state(head_direction="forward")] * 2
        events2 = _feed(analyser, 1, states2)
        assert not any(e.event_type == EventType.LOOKING_LEFT for e in events2)


class TestRulePhoneDetected:
    def test_fires_on_phone_detected(self) -> None:
        analyser = BehaviourAnalyser()
        events = _feed(analyser, 1, [_state(phone_detected=True)])
        assert any(e.event_type == EventType.PHONE_DETECTED for e in events)

    def test_does_not_fire_without_phone(self) -> None:
        analyser = BehaviourAnalyser()
        events = _feed(analyser, 1, [_state(phone_detected=False)])
        assert not any(e.event_type == EventType.PHONE_DETECTED for e in events)

    def test_stops_firing_after_phone_gone(self) -> None:
        analyser = BehaviourAnalyser()
        events = _feed(analyser, 1, [_state(phone_detected=True)])
        assert sum(1 for e in events if e.event_type == EventType.PHONE_DETECTED) == 1
        events = _feed(analyser, 1, [_state(phone_detected=False)])
        events = _feed(analyser, 1, [_state(phone_detected=True)])
        assert sum(1 for e in events if e.event_type == EventType.PHONE_DETECTED) == 1


class TestRuleStanding:
    def test_fires_on_standing(self) -> None:
        analyser = BehaviourAnalyser()
        events = _feed(analyser, 1, [_state(is_standing=True)])
        assert any(e.event_type == EventType.STANDING for e in events)

    def test_does_not_fire_when_seated(self) -> None:
        analyser = BehaviourAnalyser()
        events = _feed(analyser, 1, [_state(is_standing=False)])
        assert not any(e.event_type == EventType.STANDING for e in events)

    def test_stops_firing_after_sitting(self) -> None:
        analyser = BehaviourAnalyser()
        events = _feed(analyser, 1, [_state(is_standing=True)])
        assert sum(1 for e in events if e.event_type == EventType.STANDING) == 1
        events = _feed(analyser, 1, [_state(is_standing=False)])
        events = _feed(analyser, 1, [_state(is_standing=True)])
        assert sum(1 for e in events if e.event_type == EventType.STANDING) == 1


class TestRuleHeadDown:
    def test_fires_after_sustained_down(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(head_direction="down")] * int(HEAD_DOWN_DURATION + 2)
        events = _feed(analyser, 1, states)
        assert any(e.event_type == EventType.HEAD_DOWN for e in events)

    def test_does_not_fire_if_not_long_enough(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(head_direction="down")] * 2
        events = _feed(analyser, 1, states)
        assert not any(e.event_type == EventType.HEAD_DOWN for e in events)

    def test_does_not_fire_for_forward(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(head_direction="forward")] * int(HEAD_DOWN_DURATION + 2)
        events = _feed(analyser, 1, states)
        assert not any(e.event_type == EventType.HEAD_DOWN for e in events)

    def test_stops_firing_after_looking_up(self) -> None:
        analyser = BehaviourAnalyser(window_seconds=60)
        states = [_state(head_direction="down")] * int(HEAD_DOWN_DURATION + 2)
        events = _feed(analyser, 1, states)
        assert sum(1 for e in events if e.event_type == EventType.HEAD_DOWN) == 1
        analyser.reset()
        states2 = [_state(head_direction="forward")] * 2
        events2 = _feed(analyser, 1, states2)
        assert not any(e.event_type == EventType.HEAD_DOWN for e in events2)


class TestRuleRepeatedHeadTurns:
    def test_fires_on_repeated_turns(self) -> None:
        analyser = BehaviourAnalyser()
        states: list[dict] = []
        for i in range(HEAD_TURN_COUNT_THRESHOLD + 2):
            direction = "left" if i % 2 == 0 else "right"
            states.append(_state(head_direction=direction))
        events = _feed(analyser, 1, states)
        assert any(e.event_type == EventType.REPEATED_HEAD_TURNS for e in events)

    def test_does_not_fire_on_straight(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(head_direction="forward")] * (HEAD_TURN_COUNT_THRESHOLD + 5)
        events = _feed(analyser, 1, states)
        assert not any(e.event_type == EventType.REPEATED_HEAD_TURNS for e in events)

    def test_stops_firing_after_turns_stop(self) -> None:
        analyser = BehaviourAnalyser()
        states: list[dict] = []
        for i in range(HEAD_TURN_COUNT_THRESHOLD + 2):
            direction = "left" if i % 2 == 0 else "right"
            states.append(_state(head_direction=direction))
        events = _feed(analyser, 1, states)
        fired = sum(1 for e in events if e.event_type == EventType.REPEATED_HEAD_TURNS)
        assert fired == 1
        states2 = [_state(head_direction="forward")] * 5
        events2 = _feed(analyser, 1, states2)
        assert not any(e.event_type == EventType.REPEATED_HEAD_TURNS for e in events2)

    def test_includes_mar_in_details(self) -> None:
        analyser = BehaviourAnalyser()
        states: list[dict] = []
        for i in range(HEAD_TURN_COUNT_THRESHOLD + 2):
            direction = "left" if i % 2 == 0 else "right"
            states.append(
                _state(head_direction=direction, mouth_aspect_ratio=MAR_TALKING_THRESHOLD + 0.1)
            )
        events = _feed(analyser, 1, states)
        turned_events = [e for e in events if e.event_type == EventType.REPEATED_HEAD_TURNS]
        assert len(turned_events) >= 1
        assert "mouth movement" in turned_events[-1].details


class TestRuleLeftSeat:
    def test_fires_when_leaves_seat(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(is_seated=True), _state(is_seated=False)]
        events = _feed(analyser, 1, states)
        assert any(e.event_type == EventType.STUDENT_LEFT_SEAT for e in events)

    def test_does_not_fire_when_stays_seated(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(is_seated=True), _state(is_seated=True)]
        events = _feed(analyser, 1, states)
        assert not any(e.event_type == EventType.STUDENT_LEFT_SEAT for e in events)

    def test_does_not_fire_when_already_away(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(is_seated=False), _state(is_seated=False)]
        events = _feed(analyser, 1, states)
        assert not any(e.event_type == EventType.STUDENT_LEFT_SEAT for e in events)

    def test_stops_firing_after_returning(self) -> None:
        analyser = BehaviourAnalyser()
        events = _feed(analyser, 1, [_state(is_seated=True), _state(is_seated=False)])
        fired = sum(1 for e in events if e.event_type == EventType.STUDENT_LEFT_SEAT)
        assert fired == 1
        events = _feed(analyser, 1, [_state(is_seated=True)])
        events = _feed(analyser, 1, [_state(is_seated=False)])
        assert sum(1 for e in events if e.event_type == EventType.STUDENT_LEFT_SEAT) == 1


class TestRuleBodyTwisting:
    def test_fires_on_repeated_twists(self) -> None:
        analyser = BehaviourAnalyser()
        states: list[dict] = []
        for i in range(BODY_TWIST_COUNT_THRESHOLD + 2):
            orientation = "left" if i % 2 == 0 else "right"
            states.append(_state(body_orientation=orientation))
        events = _feed(analyser, 1, states)
        assert any(e.event_type == EventType.BODY_TWISTING for e in events)

    def test_does_not_fire_when_stable(self) -> None:
        analyser = BehaviourAnalyser()
        states = [_state(body_orientation="forward")] * (BODY_TWIST_COUNT_THRESHOLD + 5)
        events = _feed(analyser, 1, states)
        assert not any(e.event_type == EventType.BODY_TWISTING for e in events)

    def test_stops_firing_after_twists_stop(self) -> None:
        analyser = BehaviourAnalyser()
        states: list[dict] = []
        for i in range(BODY_TWIST_COUNT_THRESHOLD + 2):
            orientation = "left" if i % 2 == 0 else "right"
            states.append(_state(body_orientation=orientation))
        events = _feed(analyser, 1, states)
        assert sum(1 for e in events if e.event_type == EventType.BODY_TWISTING) == 1


class TestAnalyserIntegration:
    def test_multiple_students_independent(self) -> None:
        analyser = BehaviourAnalyser()
        events1 = _feed(analyser, 1, [_state(phone_detected=True)])
        events2 = _feed(analyser, 2, [_state(is_standing=True)])
        assert any(e.event_type == EventType.PHONE_DETECTED for e in events1)
        assert any(e.event_type == EventType.STANDING for e in events2)

    def test_reset_clears_state(self) -> None:
        analyser = BehaviourAnalyser()
        _feed(analyser, 1, [_state(phone_detected=True)])
        analyser.reset()
        events = _feed(analyser, 1, [_state(phone_detected=True)])
        assert any(e.event_type == EventType.PHONE_DETECTED for e in events)

    def test_empty_analyse_returns_no_events(self) -> None:
        analyser = BehaviourAnalyser()
        state = _state()
        state["timestamp"] = time.time()
        events = analyser.analyse(1, state)
        assert len(events) == 0

    def test_can_handle_none_attributes(self) -> None:
        analyser = BehaviourAnalyser()
        state: dict = {
            "head_direction": None,
            "is_standing": False,
            "phone_detected": False,
            "is_seated": True,
            "body_orientation": None,
            "timestamp": time.time(),
        }
        events = analyser.analyse(1, state)
        assert len(events) == 0
