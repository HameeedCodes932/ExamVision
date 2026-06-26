import uuid
from typing import Any

from app.db.repository import AlertRepository, StudentRepository
from app.models.db import Alert


class AlertManager:
    def __init__(self, session_factory: Any) -> None:
        self._session_factory = session_factory

    async def raise_alert(
        self,
        track_id: int,
        alert_type: str,
        severity: str,
        message: str,
    ) -> Alert | None:
        async with self._session_factory() as session:
            student_repo = StudentRepository(session)
            student = await student_repo.get_or_create_by_track_id(track_id)
            alert_repo = AlertRepository(session)
            existing = await alert_repo.get_active_by_type(student.id, alert_type)
            if existing is not None:
                return None
            alert = Alert(
                student_id=student.id,
                alert_type=alert_type,
                severity=severity,
                message=message,
            )
            return await alert_repo.create(alert)

    async def list_alerts(
        self, resolved: bool = False, limit: int = 100, offset: int = 0
    ) -> list[Alert]:
        async with self._session_factory() as session:
            repo = AlertRepository(session)
            return await repo.list_alerts(resolved=resolved, limit=limit, offset=offset)

    async def resolve(self, alert_id: uuid.UUID) -> Alert | None:
        async with self._session_factory() as session:
            repo = AlertRepository(session)
            return await repo.resolve(alert_id)
