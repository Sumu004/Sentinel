"""Central config — every value is env-driven so no path/key/table name is hardcoded.

Loophole this closes: the original MVP hardcoded 'models/your-model-for-detection',
'CrimeVideo-Hash' (duplicated in two files), and a literal 'Replace with your API URL'.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    site_id: str = field(default_factory=lambda: os.getenv("SENTINEL_SITE_ID", "dev-site-01"))

    source_kind: str = field(default_factory=lambda: os.getenv("SENTINEL_SOURCE_KIND", "webcam"))
    webcam_index: int = field(default_factory=lambda: int(os.getenv("SENTINEL_WEBCAM_INDEX", "0")))
    rtsp_url: str = field(default_factory=lambda: os.getenv("SENTINEL_RTSP_URL", ""))

    detector_backend: str = field(default_factory=lambda: os.getenv("SENTINEL_DETECTOR_BACKEND", "motion"))
    detector_model_path: str = field(default_factory=lambda: os.getenv("SENTINEL_DETECTOR_MODEL_PATH", ""))
    detect_classes: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            c.strip() for c in os.getenv("SENTINEL_DETECT_CLASSES", "person").split(",") if c.strip()
        )
    )

    event_min_duration_s: float = field(default_factory=lambda: float(os.getenv("SENTINEL_EVENT_MIN_DURATION_S", "3")))
    track_max_age_frames: int = field(default_factory=lambda: int(os.getenv("SENTINEL_TRACK_MAX_AGE_FRAMES", "15")))
    tracker_backend: str = field(default_factory=lambda: os.getenv("SENTINEL_TRACKER_BACKEND", "centroid"))

    capture_fps: int = field(default_factory=lambda: int(os.getenv("SENTINEL_CAPTURE_FPS", "15")))
    pre_event_seconds: float = field(default_factory=lambda: float(os.getenv("SENTINEL_PRE_EVENT_SECONDS", "3")))
    post_event_seconds: float = field(default_factory=lambda: float(os.getenv("SENTINEL_POST_EVENT_SECONDS", "5")))

    outbox_retry_interval_s: float = field(default_factory=lambda: float(os.getenv("SENTINEL_OUTBOX_RETRY_INTERVAL_S", "15")))
    heartbeat_interval_s: float = field(default_factory=lambda: float(os.getenv("SENTINEL_HEARTBEAT_INTERVAL_S", "30")))
    heartbeat_silent_threshold_s: float = field(
        default_factory=lambda: float(os.getenv("SENTINEL_HEARTBEAT_SILENT_THRESHOLD_S", "90"))
    )

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("SENTINEL_DATA_DIR", "./data")))
    clips_dir: Path = field(default_factory=lambda: Path(os.getenv("SENTINEL_DATA_DIR", "./data")) / "clips")
    db_path: Path = field(default_factory=lambda: Path(os.getenv("SENTINEL_DATA_DIR", "./data")) / "sentinel.db")

    evidence_key_path: Path = field(
        default_factory=lambda: Path(os.getenv("SENTINEL_DATA_DIR", "./data")) / "device_key.pem"
    )
    ipfs_enabled: bool = field(default_factory=lambda: _bool("SENTINEL_IPFS_ENABLED", False))
    ipfs_api_url: str = field(default_factory=lambda: os.getenv("SENTINEL_IPFS_API_URL", "http://127.0.0.1:5001"))
    anchor_enabled: bool = field(default_factory=lambda: _bool("SENTINEL_ANCHOR_ENABLED", False))

    storage_backend: str = field(default_factory=lambda: os.getenv("SENTINEL_STORAGE_BACKEND", "sqlite"))
    dynamodb_table: str = field(default_factory=lambda: os.getenv("SENTINEL_DYNAMODB_TABLE", "SentinelEvents"))
    dynamodb_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    api_host: str = field(default_factory=lambda: os.getenv("SENTINEL_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: int(os.getenv("SENTINEL_API_PORT", "8000")))
    api_token: str = field(default_factory=lambda: os.getenv("SENTINEL_API_TOKEN", ""))

    vlm_backend: str = field(default_factory=lambda: os.getenv("SENTINEL_VLM_BACKEND", "none"))
    vlm_endpoint: str = field(default_factory=lambda: os.getenv("SENTINEL_VLM_ENDPOINT", "http://127.0.0.1:11434"))
    vlm_model: str = field(default_factory=lambda: os.getenv("SENTINEL_VLM_MODEL", "qwen2.5vl:3b"))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
