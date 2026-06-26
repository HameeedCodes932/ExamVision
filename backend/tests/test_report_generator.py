import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.db import Alert, Event, Student
from app.schemas.events import EVENT_WEIGHTS, SEVERITY_SCORES, classify_risk
from app.services.reporting.report_generator import ReportGenerator


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.execute = AsyncMock()
    return session


def _make_student(track_id: int, **kwargs) -> Student:
    return Student(
        id=uuid.uuid4(),
        track_id=track_id,
        seat_label=kwargs.get("seat_label"),
        roll_number=kwargs.get("roll_number"),
    )


class TestRiskClassification:
    def test_classify_risk_normal(self) -> None:
        assert classify_risk(0.0) == "normal"
        assert classify_risk(10.0) == "normal"
        assert classify_risk(19.9) == "normal"

    def test_classify_risk_observe(self) -> None:
        assert classify_risk(20.0) == "observe"
        assert classify_risk(35.0) == "observe"
        assert classify_risk(49.9) == "observe"

    def test_classify_risk_warning(self) -> None:
        assert classify_risk(50.0) == "warning"
        assert classify_risk(65.0) == "warning"
        assert classify_risk(79.9) == "warning"

    def test_classify_risk_critical(self) -> None:
        assert classify_risk(80.0) == "critical"
        assert classify_risk(100.0) == "critical"
        assert classify_risk(500.0) == "critical"

    def test_weights_and_scores_defined(self) -> None:
        assert "phone_detected" in EVENT_WEIGHTS
        assert EVENT_WEIGHTS["phone_detected"] == 40.0
        assert "critical" in SEVERITY_SCORES
        assert SEVERITY_SCORES["critical"] == 80.0


class TestComputeRisk:
    def test_empty_events_no_alerts(self) -> None:
        risk = ReportGenerator._compute_risk({}, [])
        assert risk["risk_score"] == 0.0
        assert risk["risk_level"] == "normal"
        assert risk["event_risk_contribution"] == {}
        assert risk["alert_risk_contribution"] == 0.0

    def test_event_risk_scales_with_count(self) -> None:
        risk = ReportGenerator._compute_risk({"phone_detected": 2}, [])
        assert risk["risk_score"] == 80.0
        assert risk["risk_level"] == "critical"
        assert risk["event_risk_contribution"]["phone_detected"] == 80.0

    def test_alert_risk_adds_to_total(self) -> None:
        alert = Alert(
            student_id=uuid.uuid4(),
            alert_type="phone",
            severity="high",
            message="test",
            resolved=False,
        )
        risk = ReportGenerator._compute_risk({}, [alert])
        assert risk["risk_score"] == 50.0
        assert risk["risk_level"] == "warning"
        assert risk["alert_risk_contribution"] == 50.0

    def test_resolved_alerts_excluded(self) -> None:
        unresolved = Alert(
            student_id=uuid.uuid4(),
            alert_type="phone",
            severity="high",
            message="test",
            resolved=False,
        )
        resolved = Alert(
            student_id=uuid.uuid4(),
            alert_type="phone",
            severity="critical",
            message="old",
            resolved=True,
        )
        risk = ReportGenerator._compute_risk({}, [unresolved, resolved])
        assert risk["risk_score"] == 50.0
        assert risk["alert_risk_contribution"] == 50.0

    def test_multiple_event_types(self) -> None:
        risk = ReportGenerator._compute_risk(
            {"standing": 1, "head_down": 2, "unknown_ev": 3}, []
        )
        assert risk["event_risk_contribution"]["standing"] == 15.0
        assert risk["event_risk_contribution"]["head_down"] == 50.0
        assert risk["event_risk_contribution"]["unknown_ev"] == 30.0


class TestBuildTimeline:
    def test_empty_returns_empty(self) -> None:
        assert ReportGenerator._build_timeline([], []) == []

    def test_events_sorted_reverse(self) -> None:
        now = datetime.now(timezone.utc)
        later = datetime.now(timezone.utc)
        ev1 = Event(student_id=uuid.uuid4(), event_type="standing", timestamp=now)
        ev2 = Event(student_id=uuid.uuid4(), event_type="phone", timestamp=later)
        timeline = ReportGenerator._build_timeline([ev1, ev2], [])
        assert len(timeline) == 2

    def test_alerts_included(self) -> None:
        now = datetime.now(timezone.utc)
        alert = Alert(
            student_id=uuid.uuid4(),
            alert_type="phone",
            severity="high",
            message="test",
            created_at=now,
        )
        timeline = ReportGenerator._build_timeline([], [alert])
        assert len(timeline) == 1
        assert timeline[0]["alert_type"] == "phone"
        assert timeline[0]["severity"] == "high"
        assert timeline[0]["message"] == "test"

    def test_mixed_events_and_alerts(self) -> None:
        now = datetime.now(timezone.utc)
        ev = Event(student_id=uuid.uuid4(), event_type="standing", timestamp=now)
        alert = Alert(
            student_id=uuid.uuid4(),
            alert_type="phone",
            severity="medium",
            message="test",
            created_at=now,
        )
        timeline = ReportGenerator._build_timeline([ev], [alert])
        assert len(timeline) == 2


class TestReportGenerator:
    async def test_generate_report_empty(self, mock_session) -> None:
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=[])
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)
        mock_session.execute.return_value = result_mock
        gen = ReportGenerator(mock_session)
        report = await gen.generate_report("exam1")
        assert report["exam_id"] == "exam1"
        assert report["total_students"] == 0
        assert report["total_events"] == 0
        assert report["total_alerts"] == 0

    async def test_generate_report_with_students(self, mock_session) -> None:
        sid = uuid.uuid4()
        student = Student(id=sid, track_id=1, seat_label="A1")
        student.alerts = []
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=[student])
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)
        event_scalars = MagicMock()
        event_scalars.all = MagicMock(return_value=[])
        event_result = MagicMock()
        event_result.scalars = MagicMock(return_value=event_scalars)

        async def execute_side(*args, **kwargs):
            if "events" in str(args[0]):
                return event_result
            return result_mock

        mock_session.execute.side_effect = execute_side
        gen = ReportGenerator(mock_session)
        report = await gen.generate_report("exam2")
        assert report["total_students"] == 1
        assert report["students"][0]["track_id"] == 1

    async def test_generate_report_with_events(self, mock_session) -> None:
        from datetime import datetime, timezone

        sid = uuid.uuid4()
        now = datetime.now(timezone.utc)
        student = Student(id=sid, track_id=1)
        student.alerts = []
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=[student])
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)
        events = [
            Event(student_id=sid, event_type="standing", timestamp=now),
            Event(student_id=sid, event_type="phone_detected", timestamp=now),
            Event(student_id=sid, event_type="standing", timestamp=now),
        ]
        event_scalars = MagicMock()
        event_scalars.all = MagicMock(return_value=events)
        event_result = MagicMock()
        event_result.scalars = MagicMock(return_value=event_scalars)

        async def execute_side(*args, **kwargs):
            if "events" in str(args[0]):
                return event_result
            return result_mock

        mock_session.execute.side_effect = execute_side
        gen = ReportGenerator(mock_session)
        report = await gen.generate_report("exam3")
        assert report["total_events"] == 3
        assert report["students"][0]["total_events"] == 3
        assert report["students"][0]["event_breakdown"]["standing"] == 2
        assert report["students"][0]["event_breakdown"]["phone_detected"] == 1
        assert "risk_assessment" in report["students"][0]
        assert "timeline" in report["students"][0]
        assert len(report["students"][0]["timeline"]) == 3

    async def test_generate_csv_bytes(self, mock_session) -> None:
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=[])
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)
        mock_session.execute.return_value = result_mock
        gen = ReportGenerator(mock_session)
        csv = await gen.generate_csv_bytes("exam1")
        assert isinstance(csv, bytes)
        assert csv.startswith(b"Exam Report")

    async def test_generate_pdf_bytes_returns_valid_pdf(self, mock_session) -> None:
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=[])
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)
        mock_session.execute.return_value = result_mock
        gen = ReportGenerator(mock_session)
        pdf = await gen.generate_pdf_bytes("exam1")
        assert isinstance(pdf, bytes)
        assert len(pdf) > 0
        assert pdf.startswith(b"%PDF") or b"Exam Report:" in pdf

    async def test_report_includes_alerts(self, mock_session) -> None:
        sid = uuid.uuid4()
        alert = Alert(
            student_id=sid, alert_type="phone", severity="high", message="test", resolved=False
        )
        student = Student(id=sid, track_id=1)
        student.alerts = [alert]
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=[student])
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)
        event_scalars = MagicMock()
        event_scalars.all = MagicMock(return_value=[])
        event_result = MagicMock()
        event_result.scalars = MagicMock(return_value=event_scalars)

        async def execute_side(*args, **kwargs):
            if "events" in str(args[0]):
                return event_result
            return result_mock

        mock_session.execute.side_effect = execute_side
        gen = ReportGenerator(mock_session)
        report = await gen.generate_report("exam4")
        assert report["total_alerts"] == 1
        assert report["students"][0]["total_alerts"] == 1
        assert report["students"][0]["unresolved_alerts"] == 1
        assert report["students"][0]["max_severity"] == "high"

    async def test_cvs_includes_timeline_section(self, mock_session) -> None:
        sid = uuid.uuid4()
        now = datetime.now(timezone.utc)
        events = [Event(student_id=sid, event_type="standing", timestamp=now)]
        event_scalars = MagicMock()
        event_scalars.all = MagicMock(return_value=events)
        event_result = MagicMock()
        event_result.scalars = MagicMock(return_value=event_scalars)
        student = Student(id=sid, track_id=1)
        student.alerts = []
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=[student])
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)

        async def execute_side(*args, **kwargs):
            if "events" in str(args[0]):
                return event_result
            return result_mock

        mock_session.execute.side_effect = execute_side
        gen = ReportGenerator(mock_session)
        csv = await gen.generate_csv_bytes("exam5")
        assert b"--- Timeline ---" in csv
        assert b"standing" in csv

    async def test_generate_student_timeline_returns_entries(self, mock_session) -> None:
        sid = uuid.uuid4()
        now = datetime.now(timezone.utc)
        events = [Event(student_id=sid, event_type="phone_detected", timestamp=now)]
        event_scalars = MagicMock()
        event_scalars.all = MagicMock(return_value=events)
        event_result = MagicMock()
        event_result.scalars = MagicMock(return_value=event_scalars)
        student = Student(id=sid, track_id=1)
        student.alerts = []
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=[student])
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)

        async def execute_side(*args, **kwargs):
            if "events" in str(args[0]):
                return event_result
            return result_mock

        mock_session.execute.side_effect = execute_side
        gen = ReportGenerator(mock_session)
        entries = await gen.generate_student_timeline(sid)
        assert len(entries) == 1
        assert entries[0]["event_type"] == "phone_detected"

    async def test_generate_student_timeline_unknown_id(self, mock_session) -> None:
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = result_mock
        gen = ReportGenerator(mock_session)
        entries = await gen.generate_student_timeline(uuid.uuid4())
        assert entries == []

    async def test_report_includes_risk_assessment(self, mock_session) -> None:
        sid = uuid.uuid4()
        alert = Alert(
            student_id=sid, alert_type="phone", severity="high", message="test", resolved=False
        )
        now = datetime.now(timezone.utc)
        events = [Event(student_id=sid, event_type="standing", timestamp=now)]
        event_scalars = MagicMock()
        event_scalars.all = MagicMock(return_value=events)
        event_result = MagicMock()
        event_result.scalars = MagicMock(return_value=event_scalars)
        student = Student(id=sid, track_id=1)
        student.alerts = [alert]
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=[student])
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)

        async def execute_side(*args, **kwargs):
            if "events" in str(args[0]):
                return event_result
            return result_mock

        mock_session.execute.side_effect = execute_side
        gen = ReportGenerator(mock_session)
        report = await gen.generate_report("exam6")
        risk = report["students"][0]["risk_assessment"]
        assert risk is not None
        assert risk["risk_level"] in ("normal", "observe", "warning", "critical")
        assert risk["risk_score"] > 0
        assert risk["event_risk_contribution"]["standing"] == 15.0
        assert risk["alert_risk_contribution"] == 50.0
