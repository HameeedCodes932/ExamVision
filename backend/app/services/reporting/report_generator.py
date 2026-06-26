import csv
import io
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import EventRepository, StudentRepository
from app.models.db import Alert, Event
from app.schemas.events import (
    EVENT_WEIGHTS,
    SEVERITY_SCORES,
    classify_risk,
)


class ReportGenerator:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _compute_risk(
        event_breakdown: dict[str, int],
        alerts: list[Alert],
    ) -> dict[str, Any]:
        event_risk: dict[str, float] = defaultdict(float)
        for event_type, count in event_breakdown.items():
            weight = EVENT_WEIGHTS.get(event_type, 10.0)
            event_risk[event_type] = round(weight * count, 1)
        total_event_risk = sum(event_risk.values())
        alert_risk = sum(
            SEVERITY_SCORES.get(a.severity, 10.0) for a in alerts if not a.resolved
        )
        total_score = round(total_event_risk + alert_risk, 1)
        return {
            "risk_score": total_score,
            "risk_level": classify_risk(total_score),
            "event_risk_contribution": dict(event_risk),
            "alert_risk_contribution": round(alert_risk, 1),
        }

    @staticmethod
    def _build_timeline(events: list[Event], alerts: list[Alert]) -> list[dict[str, Any]]:
        combined: list[dict[str, Any]] = []
        for ev in events:
            combined.append(
                {
                    "timestamp": ev.timestamp,
                    "event_type": ev.event_type,
                    "confidence": ev.confidence,
                    "details": ev.details,
                    "alert_type": None,
                    "severity": None,
                    "message": None,
                }
            )
        for al in alerts:
            combined.append(
                {
                    "timestamp": al.created_at,
                    "event_type": None,
                    "confidence": None,
                    "details": None,
                    "alert_type": al.alert_type,
                    "severity": al.severity,
                    "message": al.message,
                }
            )
        combined.sort(
            key=lambda x: x["timestamp"] or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return combined

    async def generate_report(self, exam_id: str) -> dict[str, Any]:
        student_repo = StudentRepository(self._session)
        event_repo = EventRepository(self._session)
        students = await student_repo.list_all()
        rows: list[dict[str, Any]] = []
        for student in students:
            events = await event_repo.get_by_student(student.id, limit=10000)
            event_breakdown: dict[str, int] = defaultdict(int)
            for ev in events:
                event_breakdown[ev.event_type] += 1
            total_alerts = len(student.alerts)
            unresolved = sum(1 for a in student.alerts if not a.resolved)
            severities = [a.severity for a in student.alerts]
            max_sev = max(severities) if severities else None
            risk = self._compute_risk(event_breakdown, student.alerts)
            timeline = self._build_timeline(events, student.alerts)
            rows.append(
                {
                    "student_id": student.id,
                    "track_id": student.track_id,
                    "seat_label": student.seat_label,
                    "roll_number": student.roll_number,
                    "total_events": len(events),
                    "event_breakdown": dict(event_breakdown),
                    "total_alerts": total_alerts,
                    "unresolved_alerts": unresolved,
                    "max_severity": max_sev,
                    "suspicion_score": None,
                    "suspicion_level": None,
                    "risk_assessment": risk,
                    "timeline": timeline,
                }
            )
        return {
            "exam_id": exam_id,
            "generated_at": datetime.now(timezone.utc),
            "total_students": len(students),
            "total_events": sum(r["total_events"] for r in rows),
            "total_alerts": sum(r["total_alerts"] for r in rows),
            "students": rows,
        }

    async def generate_student_timeline(
        self, student_id: Any, limit: int = 200
    ) -> list[dict[str, Any]]:
        student_repo = StudentRepository(self._session)
        student = await student_repo.get_by_id(student_id)
        if student is None:
            return []
        event_repo = EventRepository(self._session)
        events = await event_repo.get_by_student(student.id, limit=limit)
        return self._build_timeline(events, student.alerts)

    async def generate_csv_bytes(self, exam_id: str) -> bytes:
        report = await self.generate_report(exam_id)
        buf = io.StringIO()
        writer = csv.writer(buf)

        writer.writerow(["Exam Report", exam_id])
        writer.writerow(["Generated", report["generated_at"].isoformat()])
        writer.writerow([])
        writer.writerow(
            [
                "student_id",
                "track_id",
                "seat_label",
                "roll_number",
                "total_events",
                "event_breakdown",
                "total_alerts",
                "unresolved_alerts",
                "max_severity",
                "risk_score",
                "risk_level",
                "suspicion_score",
                "suspicion_level",
            ]
        )
        for s in report["students"]:
            risk = s.get("risk_assessment") or {}
            writer.writerow(
                [
                    str(s["student_id"]),
                    s["track_id"],
                    s["seat_label"] or "",
                    s["roll_number"] or "",
                    s["total_events"],
                    str(dict(s["event_breakdown"])),
                    s["total_alerts"],
                    s["unresolved_alerts"],
                    s["max_severity"] or "",
                    risk.get("risk_score", ""),
                    risk.get("risk_level", ""),
                    s["suspicion_score"] if s["suspicion_score"] is not None else "",
                    s["suspicion_level"] or "",
                ]
            )

        writer.writerow([])
        writer.writerow(["--- Timeline ---"])
        writer.writerow(["student_id", "timestamp", "type", "detail", "severity"])
        for s in report["students"]:
            for entry in s.get("timeline", []):
                writer.writerow(
                    [
                        str(s["student_id"]),
                        entry["timestamp"].isoformat() if entry.get("timestamp") else "",
                        entry.get("event_type") or entry.get("alert_type") or "",
                        entry.get("details") or entry.get("message") or "",
                        entry.get("severity") or "",
                    ]
                )

        return buf.getvalue().encode("utf-8")

    async def generate_pdf_bytes(self, exam_id: str) -> bytes:
        report = await self.generate_report(exam_id)
        try:
            from reportlab.lib import colors  # type: ignore[import-untyped]
            from reportlab.lib.pagesizes import A4  # type: ignore[import-untyped]
            from reportlab.lib.units import mm  # type: ignore[import-untyped]
            from reportlab.pdfgen import canvas as pdf_canvas  # type: ignore[import-untyped]
        except ImportError:
            return self._pdf_text_fallback(report)

        buf = io.BytesIO()
        c = pdf_canvas.Canvas(buf, pagesize=A4)
        width, height = A4
        margin = 20 * mm
        y = height - margin

        def _draw(text: str, size: int = 10, bold: bool = False) -> None:
            nonlocal y
            if y < margin:
                c.showPage()
                y = height - margin
            c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
            c.drawString(margin, y, text)
            y -= size * 1.8

        # Cover
        c.setFont("Helvetica-Bold", 24)
        c.drawString(margin, y, "Proctor Exam Report")
        y -= 40
        c.setFont("Helvetica", 14)
        c.drawString(margin, y, f"Session: {report['exam_id']}")
        y -= 30
        c.setFont("Helvetica", 10)
        c.drawString(
            margin, y, f"Generated: {report['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}"
        )
        y -= 60

        # Summary
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, y, "Summary")
        y -= 25
        c.setFont("Helvetica", 11)
        for label, val in [
            ("Total Students", str(report["total_students"])),
            ("Total Events", str(report["total_events"])),
            ("Total Alerts", str(report["total_alerts"])),
        ]:
            c.drawString(margin + 10, y, f"{label}:  {val}")
            y -= 18
        y -= 15

        # Student detail pages
        for idx, s in enumerate(report["students"]):
            if y < margin + 80:
                c.showPage()
                y = height - margin
            c.setFont("Helvetica-Bold", 14)
            name = s["roll_number"] or f"Student #{s['track_id']}"
            c.drawString(margin, y, f"{idx + 1}. {name}")
            y -= 22
            c.setFont("Helvetica", 10)
            risk = s.get("risk_assessment") or {}
            alert_info = (
                f"Events: {s['total_events']}  |  Alerts: {s['total_alerts']}"
                f" ({s['unresolved_alerts']} unresolved)"
            )
            risk_info = (
                f"Risk Score: {risk.get('risk_score', '-')}"
                f"  |  Risk Level: {risk.get('risk_level', '-')}"
            )
            lines = [
                f"Seat: {s['seat_label'] or '-'}",
                alert_info,
                risk_info,
                f"Breakdown: {dict(s['event_breakdown'])}",
            ]
            for line in lines:
                c.drawString(margin + 10, y, line)
                y -= 16

            # Timeline table header
            timeline = s.get("timeline", [])
            if timeline:
                y -= 8
                c.setFont("Helvetica-Bold", 9)
                cols = [margin, margin + 110, margin + 200, margin + 280]
                c.drawString(cols[0], y, "Time")
                c.drawString(cols[1], y, "Type")
                c.drawString(cols[2], y, "Detail")
                c.drawString(cols[3], y, "Severity")
                y -= 12
                c.setFont("Helvetica", 8)
                for entry in timeline[:30]:
                    if y < margin + 10:
                        c.showPage()
                        y = height - margin
                        c.setFont("Helvetica-Bold", 9)
                        c.drawString(cols[0], y, "Time")
                        c.drawString(cols[1], y, "Type")
                        c.drawString(cols[2], y, "Detail")
                        c.drawString(cols[3], y, "Severity")
                        y -= 12
                        c.setFont("Helvetica", 8)
                    c.setFillColor(
                        colors.HSL(
                            0 if entry.get("severity") in ("high", "critical") else 120, 0.5, 0.5
                        )
                        if entry.get("severity")
                        else colors.HSL(0, 0, 0.6)
                    )
                    ts = entry["timestamp"].strftime("%H:%M:%S") if entry.get("timestamp") else ""
                    c.drawString(cols[0], y, ts)
                    label = (entry.get("event_type") or entry.get("alert_type") or "")[:20]
                    c.drawString(cols[1], y, label)
                    detail = (entry.get("details") or entry.get("message") or "")[:30]
                    c.drawString(cols[2], y, detail)
                    c.drawString(cols[3], y, entry.get("severity") or "")
                    c.setFillColor(colors.black)
                    y -= 11
            y -= 20

        c.save()
        return buf.getvalue()

    def _pdf_text_fallback(self, report: dict[str, Any]) -> bytes:
        lines = [
            f"Exam Report: {report['exam_id']}",
            f"Generated: {report['generated_at'].isoformat()}",
            "",
            f"Total Students: {report['total_students']}",
            f"Total Events:  {report['total_events']}",
            f"Total Alerts:  {report['total_alerts']}",
            "",
        ]
        for s in report["students"]:
            risk = s.get("risk_assessment") or {}
            lines.append(f"  Student {s['track_id']} ({s['seat_label'] or 'N/A'}):")
            lines.append(
                f"    Events: {s['total_events']} | Alerts: {s['total_alerts']}"
                f" ({s['unresolved_alerts']} unresolved)"
            )
            lines.append(f"    Risk: {risk.get('risk_score', '-')} ({risk.get('risk_level', '-')})")
            lines.append(f"    Breakdown: {s['event_breakdown']}")
            lines.append("")
        return "\n".join(lines).encode("utf-8")
