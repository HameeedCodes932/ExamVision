import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.db import Alert, Student
from app.services.logging.alert_manager import AlertManager
from app.services.logging.event_logger import EventLogger


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.add_all = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_session_factory(mock_session):
    factory = MagicMock()
    factory.return_value = mock_session
    factory.__aenter__ = AsyncMock(return_value=mock_session)
    factory.__aexit__ = AsyncMock(return_value=None)
    factory.__call__ = MagicMock(return_value=mock_session)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_session)
    cm.__aexit__ = AsyncMock(return_value=None)
    factory.return_value = cm
    return factory


class TestEventLogger:
    async def test_log_event_buffers_then_flushes(self, mock_session_factory, mock_session):
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = result_mock
        logger = EventLogger(mock_session_factory, batch_size=2)
        assert logger.buffer_size == 0
        await logger.log_event(1, "standing", confidence=1.0, details="test")
        assert logger.buffer_size == 1
        await logger.log_event(1, "phone_detected", confidence=0.9, details="phone")
        assert logger.buffer_size == 0

    async def test_log_event_creates_student(self, mock_session_factory, mock_session):
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = result_mock
        logger = EventLogger(mock_session_factory, batch_size=1)
        await logger.log_event(99, "standing")
        create_call = [
            c for c in mock_session.method_calls if c[0] == "add"
        ]
        added = create_call[0].args[0] if create_call else None
        if added is not None and isinstance(added, Student):
            assert added.track_id == 99
        else:
            assert any(
                isinstance(c[1][0], Student) and c[1][0].track_id == 99
                for c in mock_session.add.call_args_list
            )

    async def test_flush_manually(self, mock_session_factory, mock_session):
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = result_mock
        logger = EventLogger(mock_session_factory, batch_size=100)
        await logger.log_event(1, "test")
        assert logger.buffer_size == 1
        await logger.flush()
        assert logger.buffer_size == 0

    async def test_log_events_bulk(self, mock_session_factory, mock_session):
        result_mock = MagicMock()
        result_mock.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.return_value = result_mock
        logger = EventLogger(mock_session_factory, batch_size=10)
        await logger.log_events_bulk(
            1, [("standing", 1.0, None), ("phone_detected", 0.9, "phone")]
        )
        assert logger.buffer_size == 2

    async def test_flush_empty_no_error(self, mock_session_factory):
        logger = EventLogger(mock_session_factory)
        await logger.flush()
        assert logger.buffer_size == 0


class TestAlertManager:
    async def test_raise_alert_creates_new(self, mock_session_factory, mock_session):
        student_result = MagicMock()
        student_result.scalar_one_or_none = MagicMock(return_value=None)
        alert_result = MagicMock()
        alert_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_session.execute.side_effect = [student_result, alert_result]
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.refresh = AsyncMock()
        manager = AlertManager(mock_session_factory)
        result = await manager.raise_alert(1, "phone", "high", "Phone detected")
        assert result is not None
        assert result.alert_type == "phone"
        assert result.severity == "high"

    async def test_raise_alert_dedup(self, mock_session_factory, mock_session):
        sid = uuid.uuid4()
        existing = Alert(
            student_id=sid, alert_type="phone", severity="high", message="existing"
        )
        student_result = MagicMock()
        student_result.scalar_one_or_none = MagicMock(return_value=None)
        alert_result = MagicMock()
        alert_result.scalar_one_or_none = MagicMock(return_value=existing)
        mock_session.execute.side_effect = [student_result, alert_result]
        manager = AlertManager(mock_session_factory)
        result = await manager.raise_alert(1, "phone", "high", "Phone detected")
        assert result is None

    async def test_list_alerts(self, mock_session_factory, mock_session):
        alerts = [Alert(student_id=uuid.uuid4(), alert_type="test", severity="low", message="m")]
        scalars_mock = MagicMock()
        scalars_mock.all = MagicMock(return_value=alerts)
        result_mock = MagicMock()
        result_mock.scalars = MagicMock(return_value=scalars_mock)
        mock_session.execute.return_value = result_mock
        manager = AlertManager(mock_session_factory)
        result = await manager.list_alerts()
        assert result == alerts

    async def test_resolve(self, mock_session_factory, mock_session):
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
        manager = AlertManager(mock_session_factory)
        result = await manager.resolve(alert.id)
        assert result is alert
        assert alert.resolved is True
