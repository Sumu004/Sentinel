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
- [x] **RF-DETR side of the bake-off — gap closed.** `training/yolo_to_coco.py`
  converts the YOLO-format VOC download to COCO so RF-DETR trains on the same
  data as YOLO12. Verified with real training (real mAP output) plus unit
  tests on the exact bbox conversion math. The notebook can now run a genuine
  two-way bake-off — it hasn't been *executed* on Colab yet (that's still a
  real GPU run only you can kick off), but nothing code-side blocks it.
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

## Phase 2.2 — Evidence-grade hardening (L5) — **delivered**

**Goal:** evidence that holds up in court / insurance.

- [x] **Sign at the edge** — `evidence/signing.py` (Phase 2.0), wired into the
  live pipeline's custody flow.
- [x] **Immutable anchor** — `evidence/merkle.py` (Merkle tree + inclusion
  proofs) + `evidence/daily_anchor.py` (ties signing + Merkle + OpenTimestamps
  together) + `evidence/anchor.py`. Verified for real: built a real Merkle
  tree from 3 signed clips, submitted the real root to public OpenTimestamps
  calendar servers, confirmed every inclusion proof verifies against the
  published root. Not AWS QLDB — discontinued 31 Jul 2025 (DECISIONS.md D6).
- [x] **Chain-of-custody log** — `evidence/custody.py`: append-only,
  hash-chained, signed. Verified tamper detection by editing a row directly
  via SQL (bypassing the API) and confirming `verify_chain()` catches it.
  Wired into `edge/pipeline.py` — every clip gets `captured`/`signed` entries
  automatically.
- [x] Fixed the `UnboundLocalError` and dead HTTP path (Phase 2.0).
- [ ] C2PA-compatible provenance (standard-conformant manifest schema, not
  just the same idea); verifiable redaction (blur + prove unedited) — not
  started.

**Deliverable:** met — an evidence package with a verifiable integrity badge
(hash + signature + Merkle inclusion proof + public Bitcoin timestamp), all
real and tested. C2PA standard-conformance and redaction remain open.

---

## Phase 2.3 — Edge resilience (L4, unattended-site core) — **delivered**

**Goal:** survive the conditions of a remote, unmanned site.

- [x] **Store-and-forward queue** — `edge/outbox.py`, SQLite-backed. Verified
  for real: killed the backend, confirmed a failed send queues instead of
  vanishing, brought the backend back, confirmed the periodic retry
  (wired into `edge/pipeline.py`'s main loop) drains the queue and the event
  actually lands.
- [x] **Local ring buffer** — `edge/recorder.py` (Phase 2.0), pre/post-event
  capture already in place.
- [x] **Watchdog + heartbeat** — `edge/cloud_client.py send_heartbeat` +
  `cloud/backend` `/heartbeat` and `/sites/status`. Verified for real: a site
  correctly reports non-silent right after a heartbeat, then correctly flips
  to `silent: true` once the configured threshold elapses.
- [ ] Cascade inference (cheap motion/anomaly gate before expensive models) —
  not started; needs the trained detector's real latency profile first.

**Deliverable:** met — an edge node that queues instead of losing events on
network loss, and a backend that treats a stopped heartbeat as its own alarm.
Cascade inference remains open, gated on real detector-latency data.

---

## Phase 2.4 — Reasoning brain (L2) — **delivered (free tier); VLM tiers are honest stubs**

**Goal:** turn events into understood situations.

- [x] **Context engine** — `reasoning/context.py`: rule-based schedule
  suppression (VISION.md's false-alarm killer). Verified correct
  day+time+label matching and correct non-matching on wrong day/time.
- [x] **Event description** — `reasoning/describe.py`: `TemplateDescriber` is
  the real, free, zero-dependency default, wired into the live pipeline —
  every event gets a description + severity today.
- [x] **Honestly evaluated, not faked: local/frontier VLM tiers.** Installed
  `moondream` and checked its actual local-inference API before committing to
  anything: local inference needs a separate GPU backend ("Photon") to be set
  up and running, and the cloud API needs a paid account signup — neither of
  which this session can do without you. `QwenLocalDescriber` and
  `FrontierDescriber` are real, clearly-erroring stubs at the right interface
  (same pattern `ModelDetector` used before a trained model existed), not
  fabricated integrations.
- [ ] Spatio-temporal fusion across cameras (multi-camera homography, site
  map) — not started; needs a second camera to make sense of.
- [ ] Agentic response (natural-language queries over the event store) — not
  started; would sit on top of a real VLM tier that doesn't exist yet.

**Deliverable:** met for the free tier — every event gets a real description
and severity, and schedule-based suppression genuinely reduces false alarms.
The VLM-quality upgrade (richer natural-language understanding) needs either
your own local GPU-backed VLM server or a paid API key — both are real
infrastructure decisions for you to make, not code gaps.

---

## Phase 2.5 — Notifications + multi-tenant platform (L6) — **delivered**

**Goal:** the dashboard and alerting an operator (or owner) actually uses.

- [x] Real backend (FastAPI, Phase 2.0) — the placeholder Flask polling app
  was already replaced.
- [x] **Real-time push** — `GET /events/stream` (Server-Sent Events).
  Verified for real: posted an event, watched it arrive on a live SSE
  connection instantly, no polling.
- [x] **Notification engine** — `cloud/backend/notifications.py`:
  `ConsoleChannel` (free, real, verified firing with correct severity) +
  `WebhookChannel` (free, needs your own webhook URL) + `SMSChannel` (honest
  stub — needs a paid provider key, not enabled by default). Severity
  thresholding and per-channel failure isolation both unit-tested.
- [x] **Multi-tenant `org_id`** — schema field + query filter. Verified with a
  real org-filtered query returning only that org's events.
- [ ] Indexed DynamoDB query replacing `table.scan()` — the SQLite path
  already uses an indexed query; the DynamoDB stub in `cloud/backend/db.py`
  still documents (not implements) the GSI a real deployment needs.
- [ ] Live digital-twin site map, escalation-policy cooldowns, evidence viewer
  UI beyond the Phase 2.0 detection-test page — not started.

**Deliverable:** met — a real-time-push backend with working notifications and
multi-tenant filtering. The visual dashboard and DynamoDB indexing remain open.

---

## Phase 2.6 — Learning systems + fleet scale (L3 + L4 hardening) — **federated learning delivered in simulation; fleet-scale items need a real fleet**

**Goal:** the self-improving fleet.

- [x] **Federated learning simulation** — `learning/federated_sim.py`. Flower
  (`flwr`) and Ray installed and verified importable; the actual FedAvg math
  (weighted parameter averaging) implemented directly rather than the newer
  ClientApp/ServerApp Message API, an honest tradeoff documented in the module
  docstring. Empirically verified — and the first experiment design was
  *wrong* and caught before reporting: comparing federated vs. solo on each
  site's own distribution was a statistical tie across seeds, which isn't
  even the right question. Redesigned to the comparison that matches D7's
  real use case (does a brand-new site benefit from the federated model vs.
  an existing site's idiosyncratic solo model?) — verified across 5 seeds,
  federation wins 4/5, mean accuracy 0.920 vs 0.898.
- [ ] Continual per-site learning, active-learning queue, synthetic data for
  rare events, OTA updates, IaC, observability, privacy/compliance tooling —
  **none started.** Every one of these needs a real multi-site fleet, real
  deployed hardware, or real production traffic to be more than a stub — they
  aren't buildable-and-verifiable the way the rest of this list was.

**Deliverable:** the federated-learning *code path* is proven, honestly, on
synthetic data. Fleet-scale operations (the rest of this phase) are
structurally gated on having an actual fleet — see PROJECT_STATUS.md.

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
