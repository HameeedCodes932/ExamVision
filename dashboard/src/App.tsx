import { useCallback, useEffect, useState } from "react";

import type { StreamOut, SuspicionScoreOut } from "./types";
import { AlertPanel } from "./components/AlertPanel";
import { DashboardLayout } from "./components/DashboardLayout";
import { EventFeed } from "./components/EventFeed";
import { ExamSessionSelector } from "./components/ExamSessionSelector";
import { LiveVideoCanvas } from "./components/LiveVideoCanvas";
import { StudentGrid } from "./components/StudentGrid";
import { api } from "./utils/api";

function App() {
  const [streams, setStreams] = useState<StreamOut[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [scores, setScores] = useState<SuspicionScoreOut[]>([]);
  const [selectedStudent, setSelectedStudent] = useState<number | null>(null);

  const fetchStreams = useCallback(async () => {
    try {
      const data = await api.listStreams();
      setStreams(data);
      if (!selectedCamera && data.length > 0) {
        setSelectedCamera(data[0].camera_id);
      }
    } catch {
      // server unavailable
    }
  }, [selectedCamera]);

  useEffect(() => {
    fetchStreams();
  }, [fetchStreams]);

  const refreshScores = useCallback(async () => {
    try {
      const students = await api.listStudents();
      const results = await Promise.allSettled(
        students.map((s) => api.getScore(s.track_id)),
      );
      const valid: SuspicionScoreOut[] = [];
      for (const result of results) {
        if (result.status === "fulfilled" && result.value) {
          valid.push(result.value);
        }
      }
      setScores(valid);
    } catch {
      // server unavailable
    }
  }, []);

  useEffect(() => {
    refreshScores();
    const interval = setInterval(refreshScores, 3000);
    return () => clearInterval(interval);
  }, [refreshScores]);

  const handleSelectExam = useCallback((_examId: string) => {
    // exam selection handled by PostExamReportView
  }, []);

  const activeStream = streams.find((s) => s.active);

  return (
    <DashboardLayout
      header={
        <header className="flex items-center justify-between border-b border-gray-800 pb-3">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold tracking-tight text-gray-100">Proctor</h1>
            {streams.length > 0 && (
              <div className="flex gap-1">
                {streams.map((s) => (
                  <button
                    key={s.camera_id}
                    onClick={() => setSelectedCamera(s.camera_id)}
                    className={`text-xs px-2 py-1 rounded transition-colors ${
                      selectedCamera === s.camera_id
                        ? "bg-gray-700 text-white"
                        : "bg-gray-800 text-gray-400 hover:text-gray-200"
                    }`}
                  >
                    {s.camera_id}
                    {s.active && (
                      <span className="ml-1.5 w-1.5 h-1.5 inline-block rounded-full bg-green-500" />
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="flex items-center gap-3 text-sm text-gray-500">
            <span>{scores.length} student{scores.length !== 1 ? "s" : ""}</span>
            <span className="text-xs text-gray-600">
              Polling 3s
            </span>
          </div>
        </header>
      }
      cameraFeed={
        activeStream ? (
          <LiveVideoCanvas
            key={selectedCamera ?? activeStream.camera_id}
            cameraId={selectedCamera ?? activeStream.camera_id}
            quality={80}
            maxFps={15}
          />
        ) : (
          <div className="bg-gray-900 rounded-lg border border-gray-700 flex items-center justify-center min-h-[240px]">
            <p className="text-sm text-gray-600">No active camera streams</p>
          </div>
        )
      }
      studentGrid={
        <StudentGrid
          scores={scores}
          onSelectStudent={setSelectedStudent}
          selectedTrackId={selectedStudent}
        />
      }
      eventFeed={<EventFeed />}
      alertPanel={<AlertPanel />}
      examSession={<ExamSessionSelector onSelectExam={handleSelectExam} />}
    />
  );
}

export default App;
