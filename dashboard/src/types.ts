// ── Enums ───────────────────────────────────────────────────────────────

export type Severity = "low" | "medium" | "high" | "critical";
export type SourceType = "rtsp" | "usb" | "file";
export type SuspicionLevel = "normal" | "observe" | "warning" | "critical";

// ── Students ────────────────────────────────────────────────────────────

export interface Student {
  id: string;
  track_id: number;
  seat_label: string | null;
  roll_number: string | null;
  created_at: string;
}

export interface StudentWithEvents extends Student {
  events: EventOut[];
  alerts: AlertOut[];
}

// ── Events ──────────────────────────────────────────────────────────────

export interface EventOut {
  id: string;
  student_id: string;
  timestamp: string;
  event_type: string;
  confidence: number | null;
  details: string | null;
}

// ── Alerts ──────────────────────────────────────────────────────────────

export interface AlertOut {
  id: string;
  student_id: string;
  alert_type: string;
  severity: Severity;
  message: string;
  created_at: string;
  resolved_at: string | null;
  resolved: boolean;
}

// ── Scoring ─────────────────────────────────────────────────────────────

export interface SuspicionScoreOut {
  student_id: string;
  track_id: number;
  total: number;
  breakdown: Record<string, number>;
  level: SuspicionLevel;
}

// ── Streams ─────────────────────────────────────────────────────────────

export interface StreamOut {
  camera_id: string;
  uri: string;
  source_type: SourceType;
  active: boolean;
  connected_clients: number;
}

// ── Reports ─────────────────────────────────────────────────────────────

export interface StudentReportOut {
  student_id: string;
  track_id: number;
  seat_label: string | null;
  roll_number: string | null;
  total_events: number;
  event_breakdown: Record<string, number>;
  total_alerts: number;
  unresolved_alerts: number;
  max_severity: string | null;
  suspicion_score: number | null;
  suspicion_level: string | null;
}

export interface ReportOut {
  exam_id: string;
  generated_at: string;
  total_students: number;
  total_events: number;
  total_alerts: number;
  students: StudentReportOut[];
}

// ── WebSocket messages ──────────────────────────────────────────────────

export interface WsEventMessage {
  type: "event";
  data: {
    track_id: number;
    event_type: string;
    confidence: number | null;
    details: string | null;
  };
}

export interface WsAlertMessage {
  type: "alert";
  data: AlertOut;
}

export type WsMessage = WsEventMessage | WsAlertMessage;
