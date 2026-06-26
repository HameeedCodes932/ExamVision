import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db.repository import AlertRepository, EventRepository, StudentRepository
from app.models.db import Alert, Event, Student


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    return session


class TestStudentRepository:
    async def test_get_or_create_returns_existing(self, mock_session):
        existing = Student(track_id=1, id=uuid.uuid4())
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=existing)
        mock_session.execute.return_value = result_mock
        repo = StudentRepository(mock_session)
        student = await repo.get_or_create_by_track_id(1)
        assert student is existing
        mock_session.add.assert_not_called()

    async def test_get_or_create_creates_new(self, mock_session):
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = result_mock
        repo = StudentRepository(mock_session)
        student = await repo.get_or_create_by_track_id(2, seat_label="A1")
        assert student.track_id == 2
        assert student.seat_label == "A1"
        mock_session.add.assert_called_once()

    async def test_get_by_id_returns_student(self, mock_session):
        sid = uuid.uuid4()
        student = Student(id=sid, track_id=1)
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=student)
        mock_session.execute.return_value = result_mock
        repo = StudentRepository(mock_session)
        result = await repo.get_by_id(sid)
        assert result is student

    async def test_get_by_id_returns_none(self, mock_session):
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = result_mock
        repo = StudentRepository(mock_session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    async def test_get_by_track_id(self, mock_session):
        student = Student(track_id=5)
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=student)
        mock_session.execute.return_value = result_mock
        repo = StudentRepository(mock_session)
        result = await repo.get_by_track_id(5)
        assert result is student

    async def test_list_all(self, mock_session):
        students = [Student(track_id=1), Student(track_id=2)]
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=students)
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)
        mock_session.execute.return_value = result_mock
        repo = StudentRepository(mock_session)
        result = await repo.list_all()
        assert result == students


class TestEventRepository:
    async def test_bulk_insert(self, mock_session):
        events = [Event(student_id=uuid.uuid4(), event_type="test")]
        repo = EventRepository(mock_session)
        await repo.bulk_insert(events)
        mock_session.add_all.assert_called_once_with(events)
        mock_session.flush.assert_awaited_once()

    async def test_get_by_student(self, mock_session):
        sid = uuid.uuid4()
        events = [Event(student_id=sid, event_type="test")]
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=events)
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)
        mock_session.execute.return_value = result_mock
        repo = EventRepository(mock_session)
        result = await repo.get_by_student(sid, limit=10, offset=0)
        assert result == events


class TestAlertRepository:
    async def test_create(self, mock_session):
        alert = Alert(student_id=uuid.uuid4(), alert_type="phone", severity="high", message="test")
        repo = AlertRepository(mock_session)
        result = await repo.create(alert)
        mock_session.add.assert_called_once_with(alert)
        mock_session.flush.assert_awaited_once()
        mock_session.refresh.assert_awaited_once_with(alert)
        assert result is alert

    async def test_get_active_by_type_finds(self, mock_session):
        alert = Alert(student_id=uuid.uuid4(), alert_type="phone", severity="high", message="test")
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=alert)
        mock_session.execute.return_value = result_mock
        repo = AlertRepository(mock_session)
        result = await repo.get_active_by_type(uuid.uuid4(), "phone")
        assert result is alert

    async def test_get_active_by_type_none(self, mock_session):
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = result_mock
        repo = AlertRepository(mock_session)
        result = await repo.get_active_by_type(uuid.uuid4(), "phone")
        assert result is None

    async def test_list_alerts(self, mock_session):
        alerts = [Alert(student_id=uuid.uuid4(), alert_type="test", severity="low", message="m")]
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=alerts)
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)
        mock_session.execute.return_value = result_mock
        repo = AlertRepository(mock_session)
        result = await repo.list_alerts(resolved=False)
        assert result == alerts

    async def test_resolve_found(self, mock_session):
        alert = Alert(
            id=uuid.uuid4(),
            student_id=uuid.uuid4(),
            alert_type="test",
            severity="low",
            message="m",
            resolved=False,
        )
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=alert)
        mock_session.execute.return_value = result_mock
        repo = AlertRepository(mock_session)
        result = await repo.resolve(alert.id)
        assert result is alert
        assert alert.resolved is True
        assert alert.resolved_at is not None
        mock_session.flush.assert_awaited_once()
        mock_session.refresh.assert_awaited_once_with(alert)

    async def test_resolve_not_found(self, mock_session):
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = result_mock
        repo = AlertRepository(mock_session)
        result = await repo.resolve(uuid.uuid4())
        assert result is None
