import time

from app.services.behaviour import BehaviourEvent, EventType, Severity
from app.services.scoring import SuspicionScore, SuspicionScorer
from app.services.scoring.models import (
    NORMAL_THRESHOLD,
    OBSERVE_THRESHOLD,
    SCORE_WEIGHTS,
    WARNING_THRESHOLD,
    classify_score,
)
from app.services.scoring.scorer import SCORE_WEIGHTS as SCORER_WEIGHTS


class TestScoreWeights:
    def test_has_all_event_types(self) -> None:
        for et in EventType:
            assert et.value in SCORER_WEIGHTS

    def test_phone_heaviest(self) -> None:
        assert SCORER_WEIGHTS["phone_detected"] == 40.0
        assert SCORER_WEIGHTS["phone_detected"] > SCORER_WEIGHTS["student_left_seat"]

    def test_weights_positive(self) -> None:
        for w in SCORER_WEIGHTS.values():
            assert w > 0


class TestClassifyScore:
    def test_normal(self) -> None:
        assert classify_score(0.0) == "normal"
        assert classify_score(NORMAL_THRESHOLD - 0.1) == "normal"

    def test_observe(self) -> None:
        assert classify_score(NORMAL_THRESHOLD) == "observe"
        assert classify_score(OBSERVE_THRESHOLD - 0.1) == "observe"

    def test_warning(self) -> None:
        assert classify_score(OBSERVE_THRESHOLD) == "warning"
        assert classify_score(WARNING_THRESHOLD - 0.1) == "warning"

    def test_critical(self) -> None:
        assert classify_score(WARNING_THRESHOLD) == "critical"
        assert classify_score(WARNING_THRESHOLD + 50) == "critical"

    def test_negative(self) -> None:
        assert classify_score(-5.0) == "normal"


class TestSuspicionScoreModel:
    def test_to_dict(self) -> None:
        score = SuspicionScore(
            track_id=1,
            total=35.0,
            breakdown={"looking_left": 20.0, "standing": 15.0},
            level="observe",
        )
        d = score.to_dict()
        assert d["track_id"] == 1
        assert d["total"] == 35.0
        assert d["breakdown"] == {"looking_left": 20.0, "standing": 15.0}
        assert d["level"] == "observe"

    def test_defaults(self) -> None:
        score = SuspicionScore(track_id=1, total=0.0)
        assert score.breakdown == {}
        assert score.level == "normal"


class TestSuspicionScorer:
    def _make_event(
        self,
        track_id: int,
        event_type: EventType,
        timestamp: float | None = None,
        confidence: float = 1.0,
    ) -> BehaviourEvent:
        return BehaviourEvent(
            track_id=track_id,
            event_type=event_type,
            severity=Severity.LOW,
            timestamp=timestamp or time.time(),
            confidence=confidence,
        )

    def test_empty_update_returns_zero(self) -> None:
        scorer = SuspicionScorer()
        score = scorer.update(1, [], now=1000.0)
        assert score.total == 0.0
        assert score.breakdown == {}
        assert score.level == "normal"

    def test_single_event_adds_weight(self) -> None:
        scorer = SuspicionScorer()
        ev = self._make_event(1, EventType.PHONE_DETECTED, timestamp=1000.0)
        score = scorer.update(1, [ev], now=1000.0)
        assert score.total == SCORE_WEIGHTS["phone_detected"]
        assert score.breakdown["phone_detected"] == SCORE_WEIGHTS["phone_detected"]
        assert score.level != "normal"

    def test_multiple_events_accumulate(self) -> None:
        scorer = SuspicionScorer()
        ev1 = self._make_event(1, EventType.STANDING, timestamp=1000.0)
        ev2 = self._make_event(1, EventType.PHONE_DETECTED, timestamp=1000.0)
        score = scorer.update(1, [ev1, ev2], now=1000.0)
        expected = SCORE_WEIGHTS["standing"] + SCORE_WEIGHTS["phone_detected"]
        assert score.total == expected

    def test_breakdown_separates_by_type(self) -> None:
        scorer = SuspicionScorer()
        ev1 = self._make_event(1, EventType.STANDING, timestamp=1000.0)
        ev2 = self._make_event(1, EventType.STANDING, timestamp=1000.0)
        ev3 = self._make_event(1, EventType.PHONE_DETECTED, timestamp=1000.0)
        score = scorer.update(1, [ev1, ev2, ev3], now=1000.0)
        assert score.breakdown["standing"] == SCORE_WEIGHTS["standing"] * 2
        assert score.breakdown["phone_detected"] == SCORE_WEIGHTS["phone_detected"]

    def test_decay_reduces_old_events(self) -> None:
        scorer = SuspicionScorer(decay_minutes=1)
        ev = self._make_event(1, EventType.PHONE_DETECTED, timestamp=1000.0)
        score_now = scorer.update(1, [ev], now=1000.0)
        full = score_now.total
        score_later = scorer.update(1, [], now=1000.0 + 60.0)
        assert score_later.total < full
        assert score_later.total > 0

    def test_very_old_event_decays_nearly_zero(self) -> None:
        scorer = SuspicionScorer(decay_minutes=1)
        ev = self._make_event(1, EventType.PHONE_DETECTED, timestamp=1000.0)
        scorer.update(1, [ev], now=1000.0)
        score = scorer.update(1, [], now=1000.0 + 600.0)
        assert score.total < 1.0

    def test_confidence_scales_weight(self) -> None:
        scorer = SuspicionScorer()
        ev = self._make_event(
            1, EventType.PHONE_DETECTED, timestamp=1000.0, confidence=0.5
        )
        score = scorer.update(1, [ev], now=1000.0)
        assert score.total == SCORE_WEIGHTS["phone_detected"] * 0.5

    def test_get_score_returns_none_if_no_events(self) -> None:
        scorer = SuspicionScorer()
        assert scorer.get_score(1, now=1000.0) is None

    def test_get_score_after_update(self) -> None:
        scorer = SuspicionScorer()
        ev = self._make_event(1, EventType.STANDING, timestamp=1000.0)
        scorer.update(1, [ev], now=1000.0)
        score = scorer.get_score(1, now=1000.0)
        assert score is not None
        assert score.total > 0

    def test_reset_track(self) -> None:
        scorer = SuspicionScorer()
        ev = self._make_event(1, EventType.STANDING, timestamp=1000.0)
        scorer.update(1, [ev], now=1000.0)
        scorer.reset(1)
        assert scorer.get_score(1, now=1000.0) is None

    def test_reset_all(self) -> None:
        scorer = SuspicionScorer()
        ev1 = self._make_event(1, EventType.STANDING, timestamp=1000.0)
        ev2 = self._make_event(2, EventType.PHONE_DETECTED, timestamp=1000.0)
        scorer.update(1, [ev1], now=1000.0)
        scorer.update(2, [ev2], now=1000.0)
        scorer.reset()
        assert scorer.get_score(1, now=1000.0) is None
        assert scorer.get_score(2, now=1000.0) is None

    def test_multiple_students_independent_scores(self) -> None:
        scorer = SuspicionScorer()
        ev1 = self._make_event(1, EventType.STANDING, timestamp=1000.0)
        ev2 = self._make_event(2, EventType.PHONE_DETECTED, timestamp=1000.0)
        score1 = scorer.update(1, [ev1], now=1000.0)
        score2 = scorer.update(2, [ev2], now=1000.0)
        assert score1.total == SCORE_WEIGHTS["standing"]
        assert score2.total == SCORE_WEIGHTS["phone_detected"]

    def test_level_transitions(self) -> None:
        scorer = SuspicionScorer()
        now = 1000.0
        # Add events to push score through levels
        ev = self._make_event(1, EventType.PHONE_DETECTED, timestamp=now)
        score = scorer.update(1, [ev], now=now)
        assert score.level == "observe"
        # More high-weight events to push to warning
        for _ in range(2):
            ev = self._make_event(1, EventType.PHONE_DETECTED, timestamp=now)
            score = scorer.update(1, [ev], now=now)
        assert score.level in ("warning", "critical")

    def test_prune_removes_very_old_events(self) -> None:
        scorer = SuspicionScorer(decay_minutes=1)
        ev = self._make_event(1, EventType.STANDING, timestamp=100.0)
        scorer.update(1, [ev], now=100.0)
        score = scorer.update(1, [], now=1000000.0)
        assert score.total == 0.0
        assert all(v == 0.0 for v in score.breakdown.values())
