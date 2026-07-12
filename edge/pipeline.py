"""Wires source -> detector -> tracker -> debounce -> record -> evidence -> cloud.

This is the fix for the original MVP's main.py, which referenced an undefined
`object1` variable and crashed on the first detection (NameError) — see
ROADMAP.md Phase 0. Every stage here is the pluggable interface its DECISIONS.md
entry describes, so Phase 2.1 swaps the detector/tracker without touching this
file's structure.

Phase 2.3 adds edge resilience: failed cloud sends queue in edge/outbox.py and
retry periodically instead of being dropped, and a heartbeat tells the backend
this site is alive — see cloud/backend/app.py's /heartbeat and /sites/status.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from config import settings
from edge.cloud_client import send_event_or_queue, send_heartbeat, send_payload
from edge.detector import make_detector
from edge.events import debounce
from edge.outbox import Outbox, retry_pending
from edge.recorder import RingBufferRecorder
from edge.source import make_source
from edge.tracker import CentroidTracker
from evidence.custody import CustodyLog
from evidence.ipfs_client import IPFSError, add_to_ipfs
from evidence.signing import sign_clip

logger = logging.getLogger(__name__)
_custody_log: CustodyLog | None = None


def _get_custody_log() -> CustodyLog:
    global _custody_log
    if _custody_log is None:
        _custody_log = CustodyLog()
    return _custody_log


def run(show_preview: bool = False) -> None:
    settings.ensure_dirs()
    source = make_source()
    detector = make_detector()
    tracker = CentroidTracker()
    recorder = RingBufferRecorder()
    outbox = Outbox()

    last_retry = 0.0
    last_heartbeat = 0.0

    logger.info(
        "Sentinel edge pipeline starting — site=%s source=%s detector=%s",
        settings.site_id,
        settings.source_kind,
        settings.detector_backend,
    )

    try:
        for frame in source.frames():
            _process_frame(frame, detector, tracker, recorder, outbox, show_preview)

            now = time.time()
            if now - last_retry >= settings.outbox_retry_interval_s:
                sent = retry_pending(outbox, send_payload)
                if sent:
                    logger.info("Outbox: resent %d queued event(s)", sent)
                last_retry = now
            if now - last_heartbeat >= settings.heartbeat_interval_s:
                send_heartbeat()
                last_heartbeat = now
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        source.release()
        if show_preview:
            import cv2

            cv2.destroyAllWindows()


def _process_frame(
    frame: np.ndarray,
    detector,
    tracker: CentroidTracker,
    recorder: RingBufferRecorder,
    outbox: Outbox,
    show_preview: bool,
) -> None:
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    new_events = debounce(tracks)

    recorder.push_frame(frame)

    for event in new_events:
        logger.info("Event %s: %s tracked for %.1fs+", event.event_id, event.label, settings.event_min_duration_s)
        recorder.trigger(event.label)
        send_event_or_queue(event, outbox)

    finished_clip = recorder.pop_finished_clip()
    if finished_clip is not None:
        _seal_evidence(finished_clip)

    if show_preview:
        import cv2

        for det in detections:
            x, y, w, h = det.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("sentinel", frame)
        cv2.waitKey(1)


def _seal_evidence(clip_path) -> None:
    custody = _get_custody_log()
    custody.record(clip_path.name, "captured")

    try:
        manifest = sign_clip(clip_path)
        logger.info("Sealed evidence: %s (sha256=%s)", clip_path.name, manifest.sha256[:12])
        custody.record(clip_path.name, "signed")
    except Exception:
        logger.exception("Failed to sign clip %s", clip_path)
        return

    if settings.ipfs_enabled:
        try:
            cid = add_to_ipfs(clip_path)
            logger.info("Pinned %s to IPFS: %s", clip_path.name, cid)
        except IPFSError as exc:
            logger.warning("IPFS upload skipped for %s: %s", clip_path.name, exc)
