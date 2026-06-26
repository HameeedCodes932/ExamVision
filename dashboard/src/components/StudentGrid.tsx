import { useState } from "react";

import type { SuspicionLevel, SuspicionScoreOut } from "../types";
import { StudentCard } from "./StudentCard";

interface StudentGridProps {
  scores: SuspicionScoreOut[];
  onSelectStudent: (trackId: number) => void;
  selectedTrackId: number | null;
}

export function StudentGrid({ scores, onSelectStudent, selectedTrackId }: StudentGridProps) {
  const [filter, setFilter] = useState<SuspicionLevel | "all">("all");

  const filtered = filter === "all" ? scores : scores.filter((s) => s.level === filter);

  const counts = {
    all: scores.length,
    normal: scores.filter((s) => s.level === "normal").length,
    observe: scores.filter((s) => s.level === "observe").length,
    warning: scores.filter((s) => s.level === "warning").length,
    critical: scores.filter((s) => s.level === "critical").length,
  };

  const hasAlert = (_score: SuspicionScoreOut) => {
    return _score.level === "warning" || _score.level === "critical";
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 flex flex-col h-full">
      <div className="px-3 py-2 border-b border-gray-700">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-gray-400 mb-2">
          Students
        </h2>
        <div className="flex gap-1 flex-wrap">
          {(["all", "normal", "observe", "warning", "critical"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`text-xs px-2 py-0.5 rounded-full transition-colors ${
                filter === f
                  ? "bg-gray-600 text-white"
                  : "bg-gray-800 text-gray-400 hover:text-gray-200"
              }`}
            >
              {f} ({counts[f]})
            </button>
          ))}
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5 min-h-0">
        {filtered.map((s) => (
          <StudentCard
            key={s.track_id}
            trackId={s.track_id}
            seatLabel={null}
            rollNumber={`S${s.track_id}`}
            score={s.total}
            level={s.level}
            breakdown={s.breakdown}
            hasAlert={hasAlert(s)}
            selected={selectedTrackId === s.track_id}
            onClick={() => onSelectStudent(s.track_id)}
          />
        ))}
        {filtered.length === 0 && (
          <p className="text-xs text-gray-600 italic text-center py-8">No students found</p>
        )}
      </div>
    </div>
  );
}
