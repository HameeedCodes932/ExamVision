import pytest

from app.services.ws.manager import WSManager


class _FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self._sent: list[str] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, data: str) -> None:
        self._sent.append(data)

    async def send_json(self, data: dict) -> None:
        self._sent.append(str(data))


class TestWSManager:
    @pytest.mark.asyncio
    async def test_connect_and_count(self) -> None:
        mgr = WSManager()
        ws = _FakeWebSocket()
        await mgr.connect(ws, "events")
        assert mgr.count("events") == 1
        assert ws.accepted

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        mgr = WSManager()
        ws = _FakeWebSocket()
        await mgr.connect(ws, "events")
        mgr.disconnect(ws, "events")
        assert mgr.count("events") == 0
        assert "events" not in mgr._channels

    @pytest.mark.asyncio
    async def test_disconnect_unknown_no_error(self) -> None:
        mgr = WSManager()
        mgr.disconnect(_FakeWebSocket(), "nonexistent")
        assert mgr.count("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self) -> None:
        mgr = WSManager()
        ws1 = _FakeWebSocket()
        ws2 = _FakeWebSocket()
        await mgr.connect(ws1, "events")
        await mgr.connect(ws2, "events")
        await mgr.broadcast("events", {"msg": "hello"})
        assert len(ws1._sent) == 1
        assert len(ws2._sent) == 1
        assert '{"msg": "hello"}' in ws1._sent[0]

    @pytest.mark.asyncio
    async def test_broadcast_skips_dead_connections(self) -> None:
        mgr = WSManager()
        dead_ws = _FakeWebSocket()
        dead_ws.send_text = None  # type: ignore[assignment]

        async def broken_send(_data: str) -> None:
            msg = "broken"
            raise RuntimeError(msg)

        dead_ws.send_text = broken_send  # type: ignore[method-assign]
        ws2 = _FakeWebSocket()
        await mgr.connect(dead_ws, "events")
        await mgr.connect(ws2, "events")
        await mgr.broadcast("events", {"msg": "test"})
        assert mgr.count("events") == 1

    @pytest.mark.asyncio
    async def test_active_channels(self) -> None:
        mgr = WSManager()
        ws = _FakeWebSocket()
        await mgr.connect(ws, "events")
        await mgr.connect(ws, "stream:cam1")
        channels = mgr.active_channels()
        assert "events" in channels
        assert "stream:cam1" in channels

    @pytest.mark.asyncio
    async def test_broadcast_event_and_alert(self) -> None:
        mgr = WSManager()
        ws = _FakeWebSocket()
        await mgr.connect(ws, "events")
        await mgr.broadcast_event({"event_type": "standing"})
        await mgr.broadcast_alert({"alert_type": "phone"})
        assert len(ws._sent) == 2
        assert '"type": "event"' in ws._sent[0]
        assert '"type": "alert"' in ws._sent[1]
