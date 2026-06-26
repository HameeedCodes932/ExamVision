from collections.abc import Sequence
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import EventRepository, StudentRepository
from app.models.db import Event


class EventLogger:
    def __init__(
        self,
        session_factory: Any,
        batch_size: int = 30,
    ) -> None:
        self._session_factory = session_factory
        self._batch_size = batch_size
        self._buffer: list[Event] = []

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    async def log_event(
        self,
        track_id: int,
        event_type: str,
        confidence: float | None = None,
        details: str | None = None,
    ) -> None:
        async with self._session_factory() as session:
            student_repo = StudentRepository(session)
            student = await student_repo.get_or_create_by_track_id(track_id)
            event = Event(
                student_id=student.id,
                event_type=event_type,
                confidence=confidence,
                details=details,
            )
            self._buffer.append(event)
            if len(self._buffer) >= self._batch_size:
                await self._flush(session)

    async def log_events_bulk(
        self,
        track_id: int,
        event_data: Sequence[tuple[str, float | None, str | None]],
    ) -> None:
        async with self._session_factory() as session:
            student_repo = StudentRepository(session)
            student = await student_repo.get_or_create_by_track_id(track_id)
            for event_type, confidence, details in event_data:
                self._buffer.append(
                    Event(
                        student_id=student.id,
                        event_type=event_type,
                        confidence=confidence,
                        details=details,
                    )
                )
            if len(self._buffer) >= self._batch_size:
                await self._flush(session)

    async def flush(self) -> None:
        if not self._buffer:
            return
        async with self._session_factory() as session:
            await self._flush(session)

    async def _flush(self, session: AsyncSession) -> None:
        if not self._buffer:
            return
        events = self._buffer[:]
        self._buffer.clear()
        repo = EventRepository(session)
        await repo.bulk_insert(events)
