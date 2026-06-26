import { useCallback, useEffect, useState } from "react";

import type { ReportOut } from "../types";
import { api } from "../utils/api";

interface PostExamReportViewProps {
  examId: string;
}

export function PostExamReportView({ examId }: PostExamReportViewProps) {
  const [report, setReport] = useState<ReportOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchReport = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getReport(examId);
      setReport(data);
    } catch {
      setError("Failed to load report. Make sure the exam ID is valid.");
    } finally {
      setLoading(false);
    }
  }, [examId]);

  useEffect(() => {
    fetchReport();
  }, [fetchReport]);

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <p className="text-sm text-gray-400">Loading report...</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <p className="text-sm text-red-400">{error ?? "No data"}</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700">
      <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-200">
            Report: {report.exam_id}
          </h3>
          <p className="text-xs text-gray-500">
            Generated {new Date(report.generated_at).toLocaleString()}
          </p>
        </div>
        <div className="flex gap-2">
          <a
            href={api.getReportCsvUrl(examId)}
            className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 px-3 py-1 rounded transition-colors"
            download
          >
            CSV
          </a>
          <a
            href={api.getReportPdfUrl(examId)}
            className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 px-3 py-1 rounded transition-colors"
            download
          >
            PDF
          </a>
        </div>
      </div>

      <div className="px-4 py-3 border-b border-gray-700 flex gap-6 text-sm">
        <div>
          <span className="text-gray-500">Students</span>
          <p className="text-xl font-bold text-gray-100">{report.total_students}</p>
        </div>
        <div>
          <span className="text-gray-500">Events</span>
          <p className="text-xl font-bold text-gray-100">{report.total_events}</p>
        </div>
        <div>
          <span className="text-gray-500">Alerts</span>
          <p className="text-xl font-bold text-yellow-400">{report.total_alerts}</p>
        </div>
      </div>

      <div className="overflow-x-auto max-h-64 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="bg-gray-900 text-gray-500 uppercase tracking-wider sticky top-0">
            <tr>
              <th className="text-left px-3 py-2">Student</th>
              <th className="text-left px-3 py-2">Seat</th>
              <th className="text-right px-3 py-2">Events</th>
              <th className="text-right px-3 py-2">Alerts</th>
              <th className="text-right px-3 py-2">Unresolved</th>
              <th className="text-left px-3 py-2">Max Severity</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {report.students.map((s) => (
              <tr key={s.student_id} className="hover:bg-gray-700/30">
                <td className="px-3 py-2 text-gray-200 font-medium">
                  {s.roll_number ?? `S${s.track_id}`}
                </td>
                <td className="px-3 py-2 text-gray-400">{s.seat_label ?? "-"}</td>
                <td className="px-3 py-2 text-right text-gray-200">{s.total_events}</td>
                <td className="px-3 py-2 text-right text-gray-200">{s.total_alerts}</td>
                <td className="px-3 py-2 text-right text-yellow-400">{s.unresolved_alerts}</td>
                <td className="px-3 py-2">
                  {s.max_severity ? (
                    <span className="text-[10px] font-bold bg-red-600/20 text-red-400 px-1.5 py-0.5 rounded">
                      {s.max_severity}
                    </span>
                  ) : (
                    <span className="text-gray-600">-</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
