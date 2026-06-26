import { useState } from "react";

import { PostExamReportView } from "./PostExamReportView";

interface ExamSessionSelectorProps {
  onSelectExam: (examId: string) => void;
}

export function ExamSessionSelector({ onSelectExam }: ExamSessionSelectorProps) {
  const [examId, setExamId] = useState("");
  const [showReport, setShowReport] = useState(false);
  const [lastExamId, setLastExamId] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!examId.trim()) return;
    setLastExamId(examId.trim());
    onSelectExam(examId.trim());
    setShowReport(true);
  };

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={examId}
          onChange={(e) => setExamId(e.target.value)}
          placeholder="Enter exam session ID..."
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 placeholder-gray-600 focus:outline-none focus:border-gray-500"
        />
        <button
          type="submit"
          className="bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm px-4 py-1.5 rounded-lg transition-colors"
        >
          View Report
        </button>
      </form>
      {showReport && lastExamId && <PostExamReportView examId={lastExamId} />}
    </div>
  );
}
