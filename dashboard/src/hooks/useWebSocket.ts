import { useCallback, useEffect, useRef, useState } from "react";

interface UseWebSocketOptions {
  onMessage?: (data: ArrayBuffer) => void;
  enabled?: boolean;
}

export function useWebSocket(url: string, options: UseWebSocketOptions = {}) {
  const { onMessage, enabled = true } = options;
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  const connect = useCallback(() => {
    if (wsRef.current) return;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      wsRef.current = null;
    };
    ws.onerror = () => {
      ws.close();
    };
    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer && onMessage) {
        onMessage(event.data);
      }
    };
    wsRef.current = ws;
  }, [url, onMessage]);

  useEffect(() => {
    if (!enabled) return;
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
      setConnected(false);
    };
  }, [enabled, connect]);

  const send = useCallback((data: string) => {
    wsRef.current?.send(data);
  }, []);

  return { connected, send };
}
