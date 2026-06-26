import { useCallback, useState } from "react";

import type { AlertOut, Severity } from "../types";
import { usePolling } from "../hooks/usePolling";
import { api } from "../utils/api";

const severityColors: Record<Severity, string> = {
  low: "border-l-blue-500 bg-blue-500/5",
  medium: "border-l-yellow-500 bg-yellow-500/10",
  high: "border-l-orange-500 bg-orange-500/15",
  critical: "border-l-red-500 bg-red-500/20",
};

const severityBadge: Record<Severity, string> = {
  low: "bg-blue-600",
  medium: "bg-yellow-600",
  high: "bg-orange-600",
  critical: "bg-red-600",
};

export function AlertPanel() {
  const [alerts, setAlerts] = useState<AlertOut[]>([]);
  const [showResolved, setShowResolved] = useState(false);
  const [resolving, setResolving] = useState<Set<string>>(new Set());

  const fetchAlerts = useCallback(async () => {
    try {
      const data = await api.listAlerts(false);
      setAlerts(data);
    } catch {
      // server unavailable
    }
  }, []);

  usePolling(fetchAlerts, 3000);

  const handleResolve = async (alertId: string) => {
    setResolving((prev) => new Set(prev).add(alertId));
    try {
      await api.resolveAlert(alertId);
      setAlerts((prev) => prev.map((a) => (a.id === alertId ? { ...a, resolved: true } : a)));
    } catch {
      // ignore
    } finally {
      setResolving((prev) => {
        const next = new Set(prev);
        next.delete(alertId);
        return next;
      });
    }
  };

  const displayed = showResolved ? alerts : alerts.filter((a) => !a.resolved);

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 flex flex-col h-full">
      <div className="px-3 py-2 border-b border-gray-700 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400">
          Alerts
        </h2>
        <label className="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer">
          <input
            type="checkbox"
            checked={showResolved}
            onChange={(e) => setShowResolved(e.target.checked)}
            className="accent-gray-500"
          />
          Show resolved
        </label>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5 min-h-0">
        {displayed.map((alert) => (
          <div
            key={alert.id}
            className={`text-xs px-2 py-1.5 rounded border-l-2 ${
              alert.resolved
                ? "border-l-gray-600 bg-gray-800/30 opacity-60"
                : severityColors[alert.severity]
            }`}
          >
            <div className="flex items-center justify-between mb-0.5">
              <span
                className={`text-[10px] font-bold px-1 py-0.5 rounded text-white ${
                  severityBadge[alert.severity]
                }`}
              >
                {alert.severity}
              </span>
              <span className="text-gray-500 font-mono">
                {new Date(alert.created_at).toLocaleTimeString()}
              </span>
            </div>
            <p className="text-gray-200 font-medium">{alert.alert_type}</p>
            <p className="text-gray-400 mt-0.5">{alert.message}</p>
            {!alert.resolved && (
              <button
                onClick={() => handleResolve(alert.id)}
                disabled={resolving.has(alert.id)}
                className="mt-1 text-[10px] text-blue-400 hover:text-blue-300 disabled:text-gray-600 transition-colors"
              >
                {resolving.has(alert.id) ? "Resolving..." : "Resolve"}
              </button>
            )}
          </div>
        ))}
        {displayed.length === 0 && (
          <p className="text-xs text-gray-600 italic text-center py-8">
            {showResolved ? "No alerts" : "No active alerts"}
          </p>
        )}
      </div>
    </div>
  );
}
