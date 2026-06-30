# Project Sentinel — Roadmap

Phased plan to take the current MVP (single-camera webcam script + IPFS/DynamoDB
evidence logging) to the autonomous, multi-site platform described in
[VISION.md](VISION.md).

Each phase ships something testable. We build a vertical slice early, then deepen
layer by layer.

---

## Phase 0 — Current state (MVP)

What exists today:

- `main.py` — webcam → YOLO-World detection → stitch matched frames to MP4
- `ipfs_convertion.py` — upload clips to a local IPFS node, get a CID
- `pushHash.py` — push `{cid, datetime, stationRecieve}` to DynamoDB
- `lambda_function.py` — scan DynamoDB, mark received, return recent items
- `Control Station/app.py` — Flask dashboard polling the Lambda every 10s

**Known blocking bugs (fix before anything else):**

- `main.py:40` references `object1`, never defined → `NameError` on first detection.
- `ipfs_convertion.py:41` returns `cid` that is only set inside `try` → `UnboundLocalError` on IPFS failure.
- `ipfs_convertion.py:13-18` builds an HTTP POST that is never sent (dead code) — the CLI path is used instead.
- `Control Station/app.py:14` calls `requests.get('Replace with your API URL')` — placeholder fires every 10s.
- `main.py:5` `from moviepy.editor import ...` — removed in moviepy ≥2.0; `requirements.txt` pins nothing.
- `requirements.txt` lists `jsonlib-python3` and `datetime` (stdlib / wrong PyPI package).

---

## Phase 2.0 — Make it real and runnable

**Goal:** a single-site pipeline that actually runs end-to-end on real cameras.

- Fix all Phase 0 blocking bugs.
- Introduce a config layer (env / secrets) — kill every hardcoded model path,
  table name (`CrimeVideo-Hash` is duplicated literally in two files), and the
  `'Replace with your API URL'` placeholder.
- Replace `VideoCapture(0)` with an **RTSP/ONVIF source abstraction** for real IP cameras.
- Add **object tracking + event debouncing** (ByteTrack/Norfair) so "one intruder
  for 8s" is one event, not 240 frames.
- Make the detection class set config-driven, not the literal `['a','b','c','d']`.
- Pin dependencies; containerize the edge component (Docker).

**Deliverable:** one PR — runnable single-camera pipeline with config, real input,
and event debouncing.

---

## Phase 2.1 — Perception core (L1)

**Goal:** the model stack and training pipeline.

- `training/` — data config, fine-tune script, eval with a **false-alarms-per-
  camera-per-day** metric (not just mAP), ONNX/TensorRT export.
- Fine-tune YOLO11 (or RF-DETR) on bootstrap public datasets + domain footage.
- Add YOLO11-pose for fall detection (keypoint-geometry rule/classifier).
- Quantize (INT8) for the chosen edge target.
- Hard-negative mining + night/IR data to drive down false alarms.

**Deliverable:** a fine-tuned, quantized model running real-time on the edge box,
with reproducible training and an honest FP metric.

---

## Phase 2.2 — Evidence-grade hardening (L5)

**Goal:** evidence that holds up in court / insurance.

- **Sign at the edge** — hash + device-key-sign clips before upload.
- **Immutable anchor** — Merkle-tree daily hashes; anchor the root via OpenTimestamps
  (Bitcoin) + S3 Object Lock (WORM). Not AWS QLDB — discontinued 31 Jul 2025. See
  [DECISIONS.md](DECISIONS.md) D6.
- **Chain-of-custody log** — record every access/view/export.
- Fix the `UnboundLocalError` and remove the dead HTTP path while here.
- C2PA-compatible provenance; verifiable redaction (blur + prove unedited).

**Deliverable:** an evidence package with a verifiable integrity badge.

---

## Phase 2.3 — Edge resilience (L4, unattended-site core)

**Goal:** survive the conditions of a remote, unmanned site.

- **Store-and-forward queue** — events buffer locally and sync when connectivity
  returns. Cutting the internet must not erase evidence.
- **Local ring buffer** — continuously record the last N minutes to capture
  *pre-event* footage.
- **Watchdog + heartbeat** — cloud raises an alert when a site goes silent
  (camera unplugged / tampered / powered off). Silence is an alarm.
- Cascade inference (cheap motion/anomaly gate before expensive models).

**Deliverable:** an edge node that keeps working and keeps evidence through
network loss, power events, and tampering.

---

## Phase 2.4 — Reasoning brain (L2)

**Goal:** turn events into understood situations.

- Spatio-temporal fusion across cameras (multi-camera homography, site map).
- VLM-based event interpretation → human-readable alert text.
- Context engine — schedules, geofences, roles, expected vehicles.
- Agentic response — severity routing, escalation decisions, natural-language
  queries over the event store.

**Deliverable:** alerts that read like the VISION.md opening sentence, with a
measurably lower false-alarm rate.

---

## Phase 2.5 — Notifications + multi-tenant platform (L6)

**Goal:** the dashboard and alerting an operator (or owner) actually uses.

- Replace the placeholder Flask polling app with a real backend (FastAPI),
  multi-tenant (org → site → camera → event), with authn/authz. (The current
  `/assign` endpoint has zero auth.)
- Swap 10s polling for WebSocket/SSE push.
- Replace the double `table.scan()` with indexed queries / a GSI on `stationRecieve`.
- Notification engine — multi-channel (push/SMS/WhatsApp/webhook), escalation
  policy, severity routing, cooldowns to kill alert fatigue.
- Live digital-twin site map; evidence viewer with integrity badge.

**Deliverable:** a multi-site dashboard with real-time, escalating notifications.

---

## Phase 2.6 — Learning systems + fleet scale (L3 + L4 hardening)

**Goal:** the self-improving fleet.

- Continual per-site learning (forgetting-aware).
- Federated learning across sites (no raw video leaves the edge).
- Active-learning queue feeding the training pipeline.
- Synthetic data for rare/dangerous events.
- OTA model/firmware updates; IaC (Terraform); observability (metrics, logs,
  per-site uptime SLO).
- Privacy/compliance: configurable retention, face/plate blur, data-residency,
  GDPR/DPDP deletion.

**Deliverable:** a fleet that gets more accurate every week and is operable at scale.

---

## Sequencing summary

| Phase | Layer | Theme |
|-------|-------|-------|
| 2.0 | — | Runnable & honest MVP |
| 2.1 | L1 | Perception core + training |
| 2.2 | L5 | Evidence-grade hardening |
| 2.3 | L4 | Edge resilience |
| 2.4 | L2 | Reasoning brain |
| 2.5 | L6 | Notifications + platform |
| 2.6 | L3 | Learning systems + fleet scale |

Pilot target: a remote-home deployment after 2.2, deepened through later phases.

## Open decisions blocking the model track

- **Edge hardware:** Jetson Orin · Intel NUC + OpenVINO · Pi 5 + Hailo.
- **Model licence:** AGPL YOLO11 · permissive RF-DETR.
- **Pilot vertical:** remote home · warehouse.
