from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- App ---
    app_name: str = "Proctor"
    app_version: str = "0.1.0"
    debug: bool = False

    # --- Database / Cache ---
    database_url: str = "postgresql+asyncpg://proctor:proctor@localhost:5432/proctor"
    redis_url: str = "redis://localhost:6379/0"

    # --- Video ---
    video_bufsize: int = 128
    video_target_fps: int = 15
    video_frame_width: int = 1280
    video_frame_height: int = 720
    video_process_every_n: int = 1  # frame skipping: process every Nth frame

    # --- Detection ---
    detection_confidence: float = 0.5
    detection_model_size: str = "nano"
    detection_inference_backend: str = "pytorch"
    detection_quantization: str = "none"
    detection_executor_workers: int = 2

    # --- Pose ---
    pose_confidence: float = 0.5
    pose_model_size: str = "nano"
    pose_inference_backend: str = "pytorch"
    pose_quantization: str = "none"
    pose_executor_workers: int = 2

    # --- Tracking ---
    tracking_max_lost: int = 30
    tracking_iou_threshold: float = 0.3

    # --- Behaviour ---
    behaviour_window_seconds: int = 30

    # --- Scoring ---
    suspicion_decay_minutes: int = 5

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str | None = None

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # --- Model paths ---
    model_dir: Path = Path("models")

    # --- Profiling ---
    profiling_enabled: bool = False

    # --- Thresholds (tunable for real-world) ---
    threshold_standing_height_ratio: float = 0.6
    threshold_hand_raise_wrist_above_shoulder: bool = True
    threshold_head_down_pitch: float = -20.0
    threshold_phone_iou: float = 0.3
    threshold_hand_proximity: float = 100.0
    threshold_gaze_angle: float = 30.0

    model_config = {"env_prefix": "PROCTOR_", "env_file": ".env"}


settings = Settings()
