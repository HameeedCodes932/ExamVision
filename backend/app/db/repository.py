import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import Alert, Event, Student


class StudentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create_by_track_id(
        self,
        track_id: int,
        seat_label: str | None = None,
        roll_number: str | None = None,
    ) -> Student:
        result = await self._session.execute(
            select(Student).where(Student.track_id == track_id)
        )
        student = result.scalar_one_or_none()
        if student is not None:
            return student
        student = Student(
            track_id=track_id,
            seat_label=seat_label,
            roll_number=roll_number,
        )
        self._session.add(student)
        await self._session.flush()
        return student

    async def get_by_id(self, student_id: uuid.UUID) -> Student | None:
        result = await self._session.execute(
            select(Student).where(Student.id == student_id)
        )
        return result.scalar_one_or_none()

    async def get_by_track_id(self, track_id: int) -> Student | None:
        result = await self._session.execute(
            select(Student).where(Student.track_id == track_id)
        )
        return result.scalar_one_or_none()

    async def list_all(self) -> list[Student]:
        result = await self._session.execute(select(Student))
        return list(result.scalars().all())


class EventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_insert(self, events: list[Event]) -> None:
        self._session.add_all(events)
        await self._session.flush()

    async def get_by_student(
        self,
        student_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Event]:
        result = await self._session.execute(
            select(Event)
            .where(Event.student_id == student_id)
            .order_by(Event.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all())


class AlertRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, alert: Alert) -> Alert:
        self._session.add(alert)
        await self._session.flush()
        await self._session.refresh(alert)
        return alert

    async def get_active_by_type(
        self, student_id: uuid.UUID, alert_type: str, cooldown_seconds: int = 300
    ) -> Alert | None:
        cutoff = datetime.now(timezone.utc).timestamp() - cooldown_seconds
        cutoff_dt = datetime.fromtimestamp(cutoff, tz=timezone.utc)
        result = await self._session.execute(
            select(Alert).where(
                Alert.student_id == student_id,
                Alert.alert_type == alert_type,
                Alert.resolved == False,  # noqa: E712
                Alert.created_at >= cutoff_dt,
            )
        )
        return result.scalar_one_or_none()

    async def list_alerts(
        self, resolved: bool = False, limit: int = 100, offset: int = 0
    ) -> list[Alert]:
        result = await self._session.execute(
            select(Alert)
            .where(Alert.resolved == resolved)  # noqa: E712
            .order_by(Alert.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def resolve(self, alert_id: uuid.UUID) -> Alert | None:
        result = await self._session.execute(
            select(Alert).where(Alert.id == alert_id)
        )
        alert = result.scalar_one_or_none()
        if alert is None:
            return None
        alert.resolved = True
        alert.resolved_at = datetime.now(timezone.utc)
        await self._session.flush()
        await self._session.refresh(alert)
        return alert
