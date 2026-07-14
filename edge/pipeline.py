from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from config import settings
from edge.bytetrack_tracker import make_tracker
from edge.cloud_client import send_event_or_queue, send_heartbeat, send_payload
from edge.description_worker import DescriptionWorker
from edge.detector import make_detector
from edge.events import debounce
from edge.live_frame_streamer import LiveFrameStreamer
from edge.outbox import Outbox, retry_pending
from edge.recorder import RingBufferRecorder
from edge.source import make_source
from evidence.custody import CustodyLog
from evidence.ipfs_client import IPFSError, add_to_ipfs
from evidence.signing import sign_clip
from reasoning.context import ContextEngine
from reasoning.describe import TemplateDescriber, make_describer

_ASYNC_VLM_BACKENDS = {"qwen-local", "frontier"}

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
    tracker = make_tracker()
    recorder = RingBufferRecorder()
    outbox = Outbox()
    context = ContextEngine()

    fast_describer = TemplateDescriber()
    description_worker: DescriptionWorker | None = None
    if settings.vlm_backend in _ASYNC_VLM_BACKENDS:
        description_worker = DescriptionWorker(make_describer())
        description_worker.start()

    live_frame_streamer = LiveFrameStreamer()
    live_frame_streamer.start()

    last_retry = 0.0
    last_heartbeat = 0.0

    logger.info(
        "Sentinel edge pipeline starting — site=%s source=%s detector=%s vlm_backend=%s",
        settings.site_id,
        settings.source_kind,
        settings.detector_backend,
        settings.vlm_backend,
    )

    try:
        for frame in source.frames():
            _process_frame(
                frame,
                detector,
                tracker,
                recorder,
                outbox,
                context,
                fast_describer,
                description_worker,
                live_frame_streamer,
                show_preview,
            )

            now = time.time()
            if now - last_retry >= settings.outbox_retry_interval_s:
                sent = retry_pending(outbox, send_payload)
                if sent:
                    logger.info("Outbox: resent %d queued event(s)", sent)
                last_retry = now
            if now - last_heartbeat >= settings.heartbeat_interval_s:
                send_heartbeat()
                last_heartbeat = now
        else:
            logger.warning(
                "Video source stopped yielding frames (camera disconnected or released?) — pipeline exiting."
            )
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        source.release()
        if description_worker is not None:
            description_worker.stop()
        live_frame_streamer.stop()
        if show_preview:
            import cv2

            cv2.destroyAllWindows()


def _process_frame(
    frame: np.ndarray,
    detector,
    tracker,
    recorder: RingBufferRecorder,
    outbox: Outbox,
    context: ContextEngine,
    fast_describer,
    description_worker,
    live_frame_streamer: LiveFrameStreamer,
    show_preview: bool,
) -> None:
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    new_events = debounce(tracks)

    recorder.push_frame(frame)

    for event in new_events:
        suppressed, reason = context.should_suppress(event.label)
        description = fast_describer.describe(
            event.label, settings.event_min_duration_s, context_reason=reason, frame=frame
        )

        if suppressed:
            logger.info("Event %s suppressed by context engine: %s", event.event_id, reason)
            continue

        logger.info(
            "Event %s: %s (severity=%s) — %s", event.event_id, event.label, description.severity, description.text
        )
        recorder.trigger(event.label)
        described_event = dataclasses.replace(event, description=description.text, severity=description.severity)
        send_event_or_queue(described_event, outbox)

        if description_worker is not None:
            description_worker.enqueue(described_event, settings.event_min_duration_s, frame)

    finished_clip = recorder.pop_finished_clip()
    if finished_clip is not None:
        _seal_evidence(finished_clip)

    import cv2

    for det in detections:
        x, y, w, h = det.box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    live_frame_streamer.update(frame)

    if show_preview:
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
