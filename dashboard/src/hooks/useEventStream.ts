import { useCallback, useEffect, useRef, useState } from "react";

import type { WsMessage } from "../types";

interface UseEventStreamOptions {
  enabled?: boolean;
  onEvent?: (msg: WsMessage) => void;
}

export function useEventStream(options: UseEventStreamOptions = {}) {
  const { enabled = true, onEvent } = options;
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (wsRef.current) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const ws = new WebSocket(`${protocol}//${host}/ws/events`);

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      reconnectTimer.current = setTimeout(connect, 3000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        onEvent?.(msg);
      } catch {
        // ignore parse errors
      }
    };
    wsRef.current = ws;
  }, [onEvent]);

  useEffect(() => {
    if (!enabled) return;
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
      wsRef.current = null;
      setConnected(false);
    };
  }, [enabled, connect]);

  return { connected };
}
