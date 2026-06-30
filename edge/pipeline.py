"""Wires source -> detector -> tracker -> debounce -> record -> evidence -> cloud.

This is the fix for the original MVP's main.py, which referenced an undefined
`object1` variable and crashed on the first detection (NameError) — see
ROADMAP.md Phase 0. Every stage here is the pluggable interface its DECISIONS.md
entry describes, so Phase 2.1 swaps the detector/tracker without touching this
file's structure.
"""

from __future__ import annotations

import logging

import numpy as np

from config import settings
from edge.cloud_client import send_event
from edge.detector import make_detector
from edge.events import debounce
from edge.recorder import RingBufferRecorder
from edge.source import make_source
from edge.tracker import CentroidTracker
from evidence.ipfs_client import IPFSError, add_to_ipfs
from evidence.signing import sign_clip

logger = logging.getLogger(__name__)


def run(show_preview: bool = False) -> None:
    settings.ensure_dirs()
    source = make_source()
    detector = make_detector()
    tracker = CentroidTracker()
    recorder = RingBufferRecorder()

    logger.info(
        "Sentinel edge pipeline starting — site=%s source=%s detector=%s",
        settings.site_id,
        settings.source_kind,
        settings.detector_backend,
    )

    try:
        for frame in source.frames():
            _process_frame(frame, detector, tracker, recorder, show_preview)
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
    show_preview: bool,
) -> None:
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    new_events = debounce(tracks)

    recorder.push_frame(frame)

    for event in new_events:
        logger.info("Event %s: %s tracked for %.1fs+", event.event_id, event.label, settings.event_min_duration_s)
        recorder.trigger(event.label)
        send_event(event)

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
    try:
        manifest = sign_clip(clip_path)
        logger.info("Sealed evidence: %s (sha256=%s)", clip_path.name, manifest.sha256[:12])
    except Exception:
        logger.exception("Failed to sign clip %s", clip_path)
        return

    if settings.ipfs_enabled:
        try:
            cid = add_to_ipfs(clip_path)
            logger.info("Pinned %s to IPFS: %s", clip_path.name, cid)
        except IPFSError as exc:
            logger.warning("IPFS upload skipped for %s: %s", clip_path.name, exc)
