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

## Phase 2.0 — Make it real and runnable — **delivered**

**Goal:** a single-site pipeline that actually runs end-to-end on real cameras.

- [x] Fixed all Phase 0 blocking bugs (rebuilt clean rather than patched — see code).
- [x] Config layer (`config/settings.py`, env-driven) — no hardcoded model path,
  table name, or placeholder URL anywhere in the codebase.
- [x] `edge/source.py` — one `OpenCVSource` class serves both webcam (free, dev)
  and RTSP (real camera) via `SENTINEL_SOURCE_KIND`.
- [x] `edge/tracker.py` + `edge/events.py` — centroid tracking + duration-based
  debouncing, so a sustained track becomes one event, not one per frame.
  (ByteTrack is the Phase 2.1 upgrade — see DECISIONS.md D3 — once a real
  detector exists to feed it.)
- [x] `edge/detector.py` — class set is config-driven
  (`SENTINEL_DETECT_CLASSES`); ships a real, free, zero-download motion
  detector by default plus a `ModelDetector` stub at the exact interface
  Phase 2.1's fine-tuned RF-DETR/YOLO12 will fill.
- [x] `edge/recorder.py` — pre/post-event ring buffer producing real clips.
- [x] `evidence/signing.py` — sign-at-source hashing (fixes the original
  `UnboundLocalError` in `ipfs_convertion.py`); `evidence/ipfs_client.py` and
  `evidence/anchor.py` for the optional, free IPFS/OpenTimestamps path.
- [x] `cloud/backend/` — local FastAPI + SQLite backend (the free substitute
  for AWS API Gateway, D8), with bearer-token auth on write endpoints (the
  original `/assign` had none) and a `DynamoDBStore` behind the same interface
  for later.
- [x] Tested: `pytest tests/` (6/6 passing — tracker, debouncing, SQLite store)
  plus a manual end-to-end run (webcam-equivalent → detect → track → debounce →
  record → sign → verify → POST to backend → query) — see
  [README.md Quickstart](README.md#quickstart-phase-20--tested-runs-today-0-cost).
- [ ] Docker containerization — not yet done, tracked for the start of 2.1.

**Known limitation found during testing:** event duration is measured in
wall-clock time, which is correct for a live camera (each `read()` blocks at
the camera's real frame rate) but means pre-recorded video files play back
faster than real-time unless throttled — relevant for anyone writing
file-based tests against `edge/tracker.py`, not for production use.

**Deliverable:** one PR — runnable single-camera pipeline with config, real input,
and event debouncing.

---

## Phase 2.1 — Perception core (L1) — **closed: first real trained detector deployed**

**Goal:** the model stack and training pipeline.

- [x] `training/` — scope doc, dataset plan (remote-home pilot: person, vehicle,
  package, animal-as-suppression-class), data downloader (Roboflow), fine-tune
  scripts for **both** D1 bake-off candidates (RF-DETR and YOLO12), and an eval
  script with a real, runnable **false-alarms-per-camera-per-day** metric (not
  just mAP) — see [training/README.md](training/README.md).
- [x] `edge/detector.py ModelDetector` implemented and verified running real
  YOLO inference on real images.
- [x] `training/colab_finetune.ipynb` — turnkey GPU notebook: trains, runs the
  held-out-mAP bake-off, exports ONNX.
- [x] **Real GPU fine-tune completed and verified.** YOLO12s trained 45 epochs
  on the real Pascal VOC dataset (Colab T4, backbone frozen per the
  layer-freezing recipe, weights persisted to Google Drive after the first
  attempt was lost to a Colab disconnect). Final: **precision 0.862, recall
  0.836, mAP50 0.900, mAP50-95 0.696.** Convergence was clean — smooth loss
  decrease, no overfitting, the `close_mosaic` tail produced the expected late
  bump. Weights pulled down and saved at `data/models/sentinel_yolo12_voc_v1.pt`
  (gitignored — 33MB binary, not versioned in git; reproducible from the
  notebook + `results.csv`).
- [x] **The fine-tuned model verified end-to-end against a real image**, same
  test used for the plain pretrained baseline — and it **outperformed the
  baseline**: found a 4th, harder (lower-confidence, likely partially occluded)
  person that the plain COCO-pretrained `yolo12n` missed. A real, qualitative
  improvement from fine-tuning, not just parity.
- [ ] **RF-DETR side of the bake-off** — not run. The VOC path in the notebook
  only exercises YOLO12 (RF-DETR needs COCO-format data). One real detector is
  now deployed and verified; a genuine two-way bake-off is a follow-up.
- [ ] **`package` class** — VOC has no package/parcel class. The deployed model
  covers `person`/`vehicle`(`car`,`bus`)/`animal`(`dog`,`cat`) — 3 of 4 pilot
  classes. Package needs the dataset work flagged in `training/README.md`
  (community Roboflow set or self-captured porch footage).
- [ ] Quantize (ONNX→TensorRT) on the chosen edge target (D5) — the ONNX export
  cell exists in the notebook, not yet run against real hardware.
- [ ] RTMPose for fall detection (D2) — not started.
- [ ] Hard-negative mining + night/IR data — needs real pilot footage first.

**Deliverable:** met for 3 of 4 pilot classes. A real, verified, fine-tuned
YOLO12 detector is running in the pipeline today — this is no longer tooling
without a model. Package coverage, the RF-DETR bake-off comparison, and edge
quantization carry forward as scoped follow-up work, not blockers.

---

## Phase 2.x — Testing & scoring (cross-cutting, scaffolded)

- [x] [TESTING.md](TESTING.md) — protocol for testing on unseen real-world live
  footage; three separately-scored layers.
- [x] [eval/scoring.py](eval/scoring.py) — detection rate (IoU P/R/F1 +
  false-alarms/day), description rate (subject/action/severity + hallucination),
  end-to-end system score. Unit-tested (`tests/test_scoring.py`, 10 tests).
- [ ] The labelled unseen-footage test set + live VLM for description scoring —
  needs real pilot recordings and the Phase 2.4 reasoning layer.

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
