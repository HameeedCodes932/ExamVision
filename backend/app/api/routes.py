import asyncio
import time
from uuid import UUID

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response
from pydantic import BaseModel

from app.core.config import settings
from app.db import get_session
from app.db.repository import EventRepository, StudentRepository
from app.schemas.events import (
    AlertOut,
    EventOut,
    ReportOut,
    StreamOut,
    StudentOut,
    StudentWithEvents,
    TimelineEntryOut,
)
from app.services.behaviour import BehaviourAnalyser
from app.services.detection import DetectionPipeline
from app.services.face import FaceDetector
from app.services.gaze import GazeEstimator
from app.services.logging import AlertManager, EventLogger
from app.services.phone import PhoneDetector
from app.services.pose import PoseEstimator
from app.services.profiling import FrameProfiler
from app.services.profiling.benchmark import benchmark_to_csv, run_benchmark
from app.services.reporting.report_generator import ReportGenerator
from app.services.scoring import SuspicionScorer
from app.services.video import SourceConfig, StreamManager, VideoIngestionPipeline
from app.services.video.frame_utils import encode_jpeg
from app.services.ws.manager import ws_manager

router = APIRouter()

_pipeline: DetectionPipeline | None = None
_profiler: FrameProfiler | None = None
_stream_manager: StreamManager | None = None
_ingestion: VideoIngestionPipeline | None = None
_pose_estimator: PoseEstimator | None = None
_face_detector: FaceDetector | None = None
_gaze_estimator: GazeEstimator | None = None
_phone_detector: PhoneDetector | None = None
_behaviour_analyser: BehaviourAnalyser | None = None
_suspicion_scorer: SuspicionScorer | None = None
_event_logger: EventLogger | None = None
_alert_manager: AlertManager | None = None
_frame_broadcast_tasks: dict[str, asyncio.Task] = {}


async def get_pipeline() -> DetectionPipeline:
    global _pipeline
    if _pipeline is None:
        prof = await get_profiler()
        _pipeline = DetectionPipeline(profiler=prof)
        await _pipeline.load()
    return _pipeline


async def get_profiler() -> FrameProfiler | None:
    global _profiler
    if not settings.profiling_enabled:
        _profiler = None
        return None
    if _profiler is None:
        _profiler = FrameProfiler(window=500)
    return _profiler


async def get_stream_manager() -> StreamManager:
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager


async def get_ingestion() -> VideoIngestionPipeline:
    global _ingestion
    if _ingestion is None and _pipeline is not None and _stream_manager is not None:
        _ingestion = VideoIngestionPipeline(
            stream_manager=_stream_manager,
            detection_pipeline=_pipeline,
        )
    return _ingestion


async def get_pose_estimator() -> PoseEstimator:
    global _pose_estimator
    if _pose_estimator is None:
        _pose_estimator = PoseEstimator()
        await _pose_estimator.load()
    return _pose_estimator


async def get_face_detector() -> FaceDetector:
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceDetector()
        await _face_detector.load()
    return _face_detector


async def get_gaze_estimator() -> GazeEstimator:
    global _gaze_estimator
    if _gaze_estimator is None:
        _gaze_estimator = GazeEstimator()
        await _gaze_estimator.load()
    return _gaze_estimator


async def get_phone_detector() -> PhoneDetector:
    global _phone_detector
    if _phone_detector is None:
        _phone_detector = PhoneDetector()
        await _phone_detector.load()
    return _phone_detector


async def get_behaviour_analyser() -> BehaviourAnalyser:
    global _behaviour_analyser
    if _behaviour_analyser is None:
        _behaviour_analyser = BehaviourAnalyser()
    return _behaviour_analyser


async def get_suspicion_scorer() -> SuspicionScorer:
    global _suspicion_scorer
    if _suspicion_scorer is None:
        _suspicion_scorer = SuspicionScorer()
    return _suspicion_scorer


async def get_event_logger() -> EventLogger:
    global _event_logger
    if _event_logger is None:
        from app.db.session import async_session_factory

        _event_logger = EventLogger(async_session_factory)
    return _event_logger


async def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        from app.db.session import async_session_factory

        _alert_manager = AlertManager(async_session_factory)
    return _alert_manager


# ── Health ──────────────────────────────────────────────────────────────


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Detection (single image upload) ─────────────────────────────────────


@router.post("/detect")
async def detect_from_upload(file: UploadFile):
    import cv2
    import numpy as np

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    prof = await get_profiler()
    if prof:
        prof.log_frame()
    pipeline = await get_pipeline()
    if prof:
        with prof.timer("detect_api"):
            result = await pipeline.process(frame)
    else:
        result = await pipeline.process(frame)
    return {
        "detections": [
            {
                "bbox": d.bbox.to_list(),
                "confidence": d.confidence,
                "class_id": d.class_id,
                "class_name": d.class_name,
            }
            for d in result.detections
        ],
        "frame_shape": list(result.frame_shape),
        "inference_ms": round(result.inference_ms, 2),
    }


# ── Pose Estimation ────────────────────────────────────────────────────


@router.post("/pose")
async def pose_from_upload(file: UploadFile):
    import cv2
    import numpy as np

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    estimator = await get_pose_estimator()
    results = await estimator.estimate(frame)
    return {
        "people": [
            {
                "bbox": p["bbox"],
                "confidence": p["confidence"],
                "keypoints": [
                    {"x": kp[0], "y": kp[1], "confidence": kp[2]} for kp in p.get("keypoints", [])
                ],
            }
            for p in results
        ],
    }


@router.post("/pose/analyze")
async def pose_analyze(file: UploadFile):
    import cv2
    import numpy as np

    from app.services.pose import (
        compute_skeleton,
        estimate_head_direction,
        is_hand_raised,
        is_standing,
    )

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    estimator = await get_pose_estimator()
    results = await estimator.estimate(frame)

    analyzed = []
    for p in results:
        kps = p.get("keypoints", [])
        analyzed.append(
            {
                "bbox": p["bbox"],
                "confidence": p["confidence"],
                "standing": is_standing(kps),
                "hand_raised_left": is_hand_raised(kps, "left"),
                "hand_raised_right": is_hand_raised(kps, "right"),
                "head_direction": estimate_head_direction(kps),
                "skeleton": compute_skeleton(kps),
            }
        )

    return {"people": analyzed}


# ── Face Detection & Head Pose ──────────────────────────────────────────


@router.post("/face/detect")
async def face_detect(file: UploadFile):
    import cv2
    import numpy as np

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    detector = await get_face_detector()
    result = await detector.detect(frame)
    if result is None:
        return {"face": None}

    return {"face": result.to_dict()}


@router.post("/face/pose")
async def face_pose(file: UploadFile):
    import cv2
    import numpy as np

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    detector = await get_face_detector()
    result = await detector.detect(frame)
    if result is None or result.num_landmarks < 468:
        return {"face": None, "head_pose": None}

    pose = await detector.estimate_head_pose(result.landmarks, frame.shape[:2])
    return {
        "face": result.to_dict(),
        "head_pose": pose.to_dict() if pose else None,
    }


# ── Gaze Estimation ─────────────────────────────────────────────────────


@router.post("/gaze")
async def gaze_estimate(file: UploadFile):
    import cv2
    import numpy as np

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    detector = await get_face_detector()
    face_result = await detector.detect(frame)
    if face_result is None or face_result.num_landmarks < 468:
        return {"face": None, "gaze": None}

    head_pose = await detector.estimate_head_pose(face_result.landmarks, frame.shape[:2])
    head_angles = head_pose.to_dict() if head_pose else None

    face_crop = _crop_roi(frame, face_result.bbox)
    if face_crop.size == 0:
        return {"face": None, "gaze": None}

    estimator = await get_gaze_estimator()
    gaze = await estimator.estimate(face_crop, head_angles)
    if gaze is None:
        return {"face": face_result.to_dict(), "gaze": None}

    return {
        "face": face_result.to_dict(),
        "gaze": gaze.to_dict(),
    }


# ── Phone Detection ─────────────────────────────────────────────────────


@router.post("/phone/detect")
async def phone_detect(file: UploadFile):
    import cv2
    import numpy as np

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    pipeline = await get_pipeline()
    detection_result = await pipeline.process(frame)

    pose_estimator = await get_pose_estimator()
    person_keypoints: list[list[tuple[float, float, float]]] = []
    for det in detection_result.detections:
        bbox = (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2)
        kps_result = await pose_estimator.estimate_person(frame, bbox)
        if kps_result:
            person_keypoints.append(kps_result[0].get("keypoints", []))
        else:
            person_keypoints.append([])

    person_detections = [
        {
            "bbox": d.bbox.to_list(),
            "confidence": d.confidence,
            "class_id": d.class_id,
            "class_name": d.class_name,
            "track_id": d.track_id,
        }
        for d in detection_result.detections
    ]

    detector = await get_phone_detector()
    phones = await detector.process(frame, person_detections, person_keypoints)

    return {
        "phones": [p.to_dict() for p in phones],
        "people": len(person_detections),
    }


@router.post("/phone/detect-only")
async def phone_detect_only(file: UploadFile):
    import cv2
    import numpy as np

    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    detector = await get_phone_detector()
    phones = await detector.detect_phones(frame)

    return {
        "phones": [
            {
                "bbox": p["bbox"],
                "confidence": p["confidence"],
                "class_id": p["class_id"],
                "class_name": p.get("class_name", "cell phone"),
            }
            for p in phones
        ],
    }


# ── Behaviour Analysis ──────────────────────────────────────────────────


class BehaviourStateIn(BaseModel):
    track_id: int
    timestamp: float | None = None
    head_direction: str | None = None
    head_yaw: float | None = None
    head_pitch: float | None = None
    gaze_target: str | None = None
    is_standing: bool = False
    phone_detected: bool = False
    is_seated: bool = True
    body_orientation: str | None = None
    mouth_aspect_ratio: float | None = None


class BehaviourAnalyseIn(BaseModel):
    states: list[BehaviourStateIn]


class BehaviourEventOut(BaseModel):
    track_id: int
    event_type: str
    severity: str
    timestamp: float
    details: str | None = None
    confidence: float = 1.0


class BehaviourAnalyseOut(BaseModel):
    events: list[BehaviourEventOut]


@router.post("/behaviour/analyse", response_model=BehaviourAnalyseOut)
async def behaviour_analyse(body: BehaviourAnalyseIn):
    analyser = await get_behaviour_analyser()
    all_events: list[BehaviourEventOut] = []
    for student_state in body.states:
        state = student_state.model_dump(exclude_none=True)
        track_id = state.pop("track_id")
        events = analyser.analyse(track_id, state)
        for ev in events:
            all_events.append(
                BehaviourEventOut(
                    track_id=ev.track_id,
                    event_type=ev.event_type.value,
                    severity=ev.severity.value,
                    timestamp=ev.timestamp,
                    details=ev.details,
                    confidence=ev.confidence,
                )
            )
    return BehaviourAnalyseOut(events=all_events)


@router.post("/behaviour/reset")
async def behaviour_reset():
    analyser = await get_behaviour_analyser()
    analyser.reset()
    return {"status": "reset"}


# ── Suspicion Scoring ───────────────────────────────────────────────────


class SuspicionScoreOut(BaseModel):
    track_id: int
    total: float
    breakdown: dict[str, float]
    level: str


class ScoringEventIn(BaseModel):
    track_id: int
    event_type: str
    timestamp: float
    confidence: float = 1.0


class ScoringUpdateIn(BaseModel):
    track_id: int
    events: list[ScoringEventIn]


@router.post("/scoring/update", response_model=SuspicionScoreOut)
async def scoring_update(body: ScoringUpdateIn):
    scorer = await get_suspicion_scorer()
    from app.services.behaviour import BehaviourEvent, EventType, Severity

    behaviour_events = []
    for ev in body.events:
        try:
            et = EventType(ev.event_type)
        except ValueError:
            et = EventType.LOOKING_LEFT
        behaviour_events.append(
            BehaviourEvent(
                track_id=ev.track_id,
                event_type=et,
                severity=Severity.LOW,
                timestamp=ev.timestamp,
                confidence=ev.confidence,
            )
        )
    score = scorer.update(body.track_id, behaviour_events)
    return SuspicionScoreOut(
        track_id=score.track_id,
        total=round(score.total, 1),
        breakdown={k: round(v, 1) for k, v in score.breakdown.items()},
        level=score.level,
    )


@router.get("/scoring/{track_id}", response_model=SuspicionScoreOut | None)
async def scoring_get(track_id: int):
    scorer = await get_suspicion_scorer()
    score = scorer.get_score(track_id)
    if score is None:
        return None
    return SuspicionScoreOut(
        track_id=score.track_id,
        total=round(score.total, 1),
        breakdown={k: round(v, 1) for k, v in score.breakdown.items()},
        level=score.level,
    )


@router.post("/scoring/reset")
async def scoring_reset(track_id: int | None = None):
    scorer = await get_suspicion_scorer()
    scorer.reset(track_id)
    return {"status": "reset"}


# ── Stream Management ───────────────────────────────────────────────────


@router.get("/streams", response_model=list[StreamOut])
async def list_streams():
    mgr = await get_stream_manager()
    cameras = mgr.get_all_cameras()
    return [
        StreamOut(
            camera_id=c["camera_id"],
            uri=c["uri"],
            source_type=c["source_type"],
            active=c["active"],
            connected_clients=mgr.subscriber_count(c["camera_id"]),
        )
        for c in cameras
    ]


@router.post("/streams")
async def add_stream(config: SourceConfig):
    mgr = await get_stream_manager()
    await mgr.add_source(config)
    return {"status": "added", "camera_id": config.camera_id}


@router.delete("/streams/{camera_id}")
async def remove_stream(camera_id: str):
    mgr = await get_stream_manager()
    await mgr.remove_source(camera_id)
    return {"status": "removed", "camera_id": camera_id}


@router.post("/streams/{camera_id}/start")
async def start_stream(camera_id: str):
    mgr = await get_stream_manager()
    await mgr.start_source(camera_id)
    return {"status": "started", "camera_id": camera_id}


@router.post("/streams/{camera_id}/stop")
async def stop_stream(camera_id: str):
    mgr = await get_stream_manager()
    await mgr.stop_source(camera_id)
    return {"status": "stopped", "camera_id": camera_id}


# ── Students / Events / Alerts ─────────────────────────────────────────


@router.get("/students", response_model=list[StudentOut])
async def list_students(session=Depends(get_session)):  # noqa: B008
    repo = StudentRepository(session)
    students = await repo.list_all()
    return students


@router.get("/students/{student_id}", response_model=StudentWithEvents)
async def get_student(student_id: UUID, session=Depends(get_session)):  # noqa: B008
    repo = StudentRepository(session)
    student = await repo.get_by_id(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@router.get("/students/{student_id}/history", response_model=StudentWithEvents)
async def get_student_history(student_id: UUID, session=Depends(get_session)):  # noqa: B008
    repo = StudentRepository(session)
    student = await repo.get_by_id(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@router.get("/students/{student_id}/events", response_model=list[EventOut])
async def get_student_events(
    student_id: UUID,
    limit: int = 100,
    offset: int = 0,
    session=Depends(get_session),  # noqa: B008
):
    repo = EventRepository(session)
    events = await repo.get_by_student(student_id, limit=limit, offset=offset)
    return events


@router.post("/events/log")
async def log_event(
    track_id: int,
    event_type: str,
    confidence: float | None = None,
    details: str | None = None,
):
    logger = await get_event_logger()
    await logger.log_event(track_id, event_type, confidence=confidence, details=details)
    await ws_manager.broadcast_event(
        {
            "track_id": track_id,
            "event_type": event_type,
            "confidence": confidence,
            "details": details,
        }
    )
    return {"status": "logged", "buffer_size": logger.buffer_size}


@router.post("/events/flush")
async def flush_events():
    logger = await get_event_logger()
    await logger.flush()
    return {"status": "flushed"}


@router.get("/alerts", response_model=list[AlertOut])
async def list_alerts(resolved: bool = False):
    manager = await get_alert_manager()
    alerts = await manager.list_alerts(resolved=resolved)
    return alerts


@router.post("/alerts/raise")
async def raise_alert(track_id: int, alert_type: str, severity: str, message: str):
    manager = await get_alert_manager()
    alert = await manager.raise_alert(track_id, alert_type, severity, message)
    if alert is None:
        return {"status": "duplicate"}
    out = AlertOut(
        id=alert.id,
        student_id=alert.student_id,
        alert_type=alert.alert_type,
        severity=alert.severity,
        message=alert.message,
        created_at=alert.created_at,
        resolved_at=alert.resolved_at,
        resolved=alert.resolved,
    )
    await ws_manager.broadcast_alert(out.model_dump())
    return out


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: UUID):
    manager = await get_alert_manager()
    alert = await manager.resolve(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return AlertOut(
        id=alert.id,
        student_id=alert.student_id,
        alert_type=alert.alert_type,
        severity=alert.severity,
        message=alert.message,
        created_at=alert.created_at,
        resolved_at=alert.resolved_at,
        resolved=alert.resolved,
    )


# ── Reports ──────────────────────────────────────────────────────────────


@router.get("/reports/{exam_id}", response_model=ReportOut)
async def get_report(exam_id: str, session=Depends(get_session)):  # noqa: B008
    generator = ReportGenerator(session)
    report = await generator.generate_report(exam_id)
    return ReportOut(**report)


@router.get("/reports/{exam_id}/csv")
async def get_report_csv(exam_id: str, session=Depends(get_session)):  # noqa: B008
    generator = ReportGenerator(session)
    csv_bytes = await generator.generate_csv_bytes(exam_id)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="report_{exam_id}.csv"'},
    )


@router.get("/reports/{exam_id}/pdf")
async def get_report_pdf(exam_id: str, session=Depends(get_session)):  # noqa: B008
    generator = ReportGenerator(session)
    pdf_bytes = await generator.generate_pdf_bytes(exam_id)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{exam_id}.pdf"'},
    )


@router.get("/reports/{exam_id}/timeline/{student_id}", response_model=list[TimelineEntryOut])
async def get_report_timeline(
    exam_id: str,
    student_id: UUID,
    limit: int = 200,
    session=Depends(get_session),  # noqa: B008
):
    generator = ReportGenerator(session)
    entries = await generator.generate_student_timeline(student_id, limit=limit)
    return entries


# ── Benchmark / Profiling ────────────────────────────────────────────────


@router.post("/benchmark")
async def benchmark_start(
    num_frames: int = Query(50, ge=10, le=500),
    model_size: str = Query("nano", pattern="^(nano|small|medium)$"),
    backend: str = Query("pytorch", pattern="^(pytorch|onnx)$"),
    quantization: str = Query("none", pattern="^(none|fp16|int8)$"),
):
    report = await run_benchmark(
        num_frames=num_frames,
        model_size=model_size,
        backend=backend,
        quantization=quantization,
    )
    return Response(content=report, media_type="text/plain")


@router.post("/benchmark/csv")
async def benchmark_csv(
    num_frames: int = Query(50, ge=10, le=500),
    model_size: str = Query("nano", pattern="^(nano|small|medium)$"),
    backend: str = Query("pytorch", pattern="^(pytorch|onnx)$"),
    quantization: str = Query("none", pattern="^(none|fp16|int8)$"),
):
    report = await run_benchmark(
        num_frames=num_frames,
        model_size=model_size,
        backend=backend,
        quantization=quantization,
    )
    csv_data = benchmark_to_csv(report)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=benchmark.csv"},
    )


@router.get("/profiler/summary")
async def profiler_summary():
    prof = await get_profiler()
    if prof is None or prof.total_frames == 0:
        return {"profiling_enabled": False, "frames": 0}
    return prof.summary()


@router.post("/profiler/reset")
async def profiler_reset():
    prof = await get_profiler()
    if prof is not None:
        prof.reset()
    return {"status": "reset"}


# ── WebSocket: Live stream frames ───────────────────────────────────────


@router.websocket("/ws/stream/{camera_id}")
async def stream_websocket(
    websocket: WebSocket,
    camera_id: str,
    quality: int = Query(85, ge=10, le=100),
    max_fps: int = Query(30, ge=1, le=60),
    mode: str = Query("binary", pattern="^(binary|json)$"),
):
    mgr = await get_stream_manager()
    await ws_manager.connect(websocket, f"stream:{camera_id}")
    q = await mgr.subscribe(camera_id)
    last_send = 0.0
    skip_interval = 1.0 / max_fps if max_fps > 0 else 0.0
    try:
        while True:
            frame = await q.get()
            now = time.time()
            if now - last_send < skip_interval:
                continue
            last_send = now
            if mode == "json":
                await websocket.send_json(
                    {
                        "camera_id": camera_id,
                        "timestamp": frame.timestamp,
                        "frame_index": frame.frame_index,
                        "shape": [frame.height, frame.width, 3],
                    }
                )
            else:
                jpeg_bytes = encode_jpeg(frame.raw, quality=quality)
                await websocket.send_bytes(jpeg_bytes)
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket, f"stream:{camera_id}")
        mgr.unsubscribe(camera_id, q)


@router.websocket("/ws/events")
async def events_websocket(websocket: WebSocket):
    await ws_manager.connect(websocket, "events")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket, "events")


def _crop_roi(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    return frame[y1:y2, x1:x2]
