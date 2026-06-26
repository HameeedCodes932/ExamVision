import type { SuspicionLevel } from "../types";

interface StudentCardProps {
  trackId: number;
  seatLabel: string | null;
  rollNumber: string | null;
  score: number;
  level: SuspicionLevel;
  breakdown: Record<string, number>;
  hasAlert: boolean;
  onClick?: () => void;
  selected?: boolean;
}

const levelColors: Record<SuspicionLevel, string> = {
  normal: "border-green-500/40 bg-green-500/5",
  observe: "border-blue-500/40 bg-blue-500/5",
  warning: "border-yellow-500/40 bg-yellow-500/10",
  critical: "border-red-500/60 bg-red-500/10",
};

const levelBadge: Record<SuspicionLevel, string> = {
  normal: "bg-green-600",
  observe: "bg-blue-600",
  warning: "bg-yellow-600",
  critical: "bg-red-600",
};

export function StudentCard({
  trackId,
  seatLabel,
  rollNumber,
  score,
  level,
  breakdown,
  hasAlert,
  onClick,
  selected,
}: StudentCardProps) {
  return (
    <button
      onClick={onClick}
      className={`text-left w-full rounded-lg border px-3 py-2 transition-all duration-200 ${
        levelColors[level]
      } ${selected ? "ring-2 ring-white/30 scale-[1.02]" : "hover:scale-[1.01]"}`}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-semibold text-gray-100">
          {rollNumber ?? `Student #${trackId}`}
        </span>
        <div className="flex items-center gap-1.5">
          {hasAlert && (
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          )}
          <span
            className={`text-[10px] font-bold px-1.5 py-0.5 rounded text-white ${levelBadge[level]}`}
          >
            {level}
          </span>
        </div>
      </div>
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-gray-400">{seatLabel ?? `Seat ${trackId}`}</span>
        <span className="text-lg font-mono font-bold text-gray-100">
          {score.toFixed(1)}
        </span>
      </div>
      {Object.keys(breakdown).length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {Object.entries(breakdown)
            .filter(([, v]) => v > 0)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 3)
            .map(([key, val]) => (
              <span
                key={key}
                className="text-[10px] bg-gray-800 text-gray-400 px-1 rounded"
              >
                {key}: {val.toFixed(1)}
              </span>
            ))}
        </div>
      )}
    </button>
  );
}
