import { useEffect, useRef, useState } from "react";

import type { WsMessage } from "../types";
import { useEventStream } from "../hooks/useEventStream";

interface FeedItem {
  id: number;
  timestamp: string;
  message: string;
  severity: "low" | "medium" | "high" | "critical" | "info";
}

const severityColors: Record<FeedItem["severity"], string> = {
  info: "border-l-blue-500 bg-blue-500/10",
  low: "border-l-gray-400 bg-gray-500/10",
  medium: "border-l-yellow-500 bg-yellow-500/10",
  high: "border-l-orange-500 bg-orange-500/10",
  critical: "border-l-red-500 bg-red-500/10",
};

export function EventFeed() {
  const [items, setItems] = useState<FeedItem[]>([]);
  const idRef = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);

  const handleEvent = (msg: WsMessage) => {
    const now = new Date().toLocaleTimeString();
    if (msg.type === "event") {
      setItems((prev) =>
        [
          {
            id: ++idRef.current,
            timestamp: now,
            message: `${msg.data.event_type} — Track ${msg.data.track_id}`,
            severity: "info" as const,
          },
          ...prev,
        ].slice(0, 200),
      );
    } else if (msg.type === "alert") {
      setItems((prev) =>
        [
          {
            id: ++idRef.current,
            timestamp: now,
            message: `${msg.data.alert_type}: ${msg.data.message}`,
            severity: msg.data.severity,
          },
          ...prev,
        ].slice(0, 200),
      );
    }
  };

  useEventStream({ onEvent: handleEvent });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [items]);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 flex flex-col h-full">
      <div className="px-3 py-2 border-b border-gray-700 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
          Live Events
        </h2>
        <button
          onClick={() => setItems([])}
          className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
        >
          Clear
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1 min-h-0">
        {items.map((item) => (
          <div
            key={item.id}
            className={`text-xs px-2 py-1 rounded border-l-2 ${severityColors[item.severity]}`}
          >
            <span className="text-gray-500 mr-2 font-mono">{item.timestamp}</span>
            <span className="text-gray-200">{item.message}</span>
          </div>
        ))}
        {items.length === 0 && (
          <p className="text-xs text-gray-600 italic text-center py-8">
            No events yet. Waiting for stream...
          </p>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
