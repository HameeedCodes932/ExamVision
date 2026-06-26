import json
import logging
from collections import defaultdict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WSManager:
    def __init__(self) -> None:
        self._channels: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        await websocket.accept()
        self._channels[channel].add(websocket)
        logger.info("WebSocket connected to channel '%s' (%d active)", channel, self.count(channel))

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        self._channels[channel].discard(websocket)
        if not self._channels[channel]:
            del self._channels[channel]
        logger.info(
            "WebSocket disconnected from channel '%s' (%d active)", channel, self.count(channel)
        )

    def count(self, channel: str) -> int:
        return len(self._channels.get(channel, set()))

    def active_channels(self) -> list[str]:
        return list(self._channels.keys())

    async def broadcast(self, channel: str, data: dict) -> None:
        dead: list[WebSocket] = []
        payload = json.dumps(data)
        for ws in self._channels.get(channel, set()):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, channel)

    async def broadcast_event(self, event: dict) -> None:
        await self.broadcast("events", {"type": "event", "data": event})

    async def broadcast_alert(self, alert: dict) -> None:
        await self.broadcast("events", {"type": "alert", "data": alert})


ws_manager = WSManager()
