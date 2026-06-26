import type {
  AlertOut,
  ReportOut,
  StreamOut,
  Student,
  StudentWithEvents,
  SuspicionScoreOut,
} from "../types";

const BASE = "/api/v1";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} returned ${res.status}`);
  return res.json();
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`POST ${path} returned ${res.status}`);
  return res.json();
}

export const api = {
  // Streams
  listStreams: () => getJson<StreamOut[]>("/streams"),

  // Students
  listStudents: () => getJson<Student[]>("/students"),
  getStudent: (id: string) => getJson<StudentWithEvents>(`/students/${id}`),
  getStudentHistory: (id: string) =>
    getJson<StudentWithEvents>(`/students/${id}/history`),

  // Events
  logEvent: (
    trackId: number,
    eventType: string,
    confidence?: number,
    details?: string,
  ) =>
    postJson<{ status: string }>(
      `/events/log?track_id=${trackId}&event_type=${encodeURIComponent(eventType)}${
        confidence ? `&confidence=${confidence}` : ""
      }${details ? `&details=${encodeURIComponent(details)}` : ""}`,
    ),

  // Scores
  getScore: (trackId: number) =>
    getJson<SuspicionScoreOut | null>(`/scoring/${trackId}`),

  // Alerts
  listAlerts: (resolved = false) =>
    getJson<AlertOut[]>(`/alerts?resolved=${resolved}`),
  raiseAlert: (trackId: number, alertType: string, severity: string, message: string) =>
    postJson<AlertOut | { status: "duplicate" }>(
      `/alerts/raise?track_id=${trackId}&alert_type=${encodeURIComponent(alertType)}&severity=${severity}&message=${encodeURIComponent(message)}`,
    ),
  resolveAlert: (alertId: string) =>
    postJson<AlertOut>(`/alerts/${alertId}/resolve`),

  // Reports
  getReport: (examId: string) => getJson<ReportOut>(`/reports/${examId}`),
  getReportCsvUrl: (examId: string) => `${BASE}/reports/${examId}/csv`,
  getReportPdfUrl: (examId: string) => `${BASE}/reports/${examId}/pdf`,

  // Health
  health: () => getJson<{ status: string }>("/health"),
};
