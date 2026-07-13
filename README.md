# Sentinel

> From a detection script to an autonomous guard for places nobody is watching.

Sentinel is an edge-first AI monitoring platform for remote, unattended sites —
homes, warehouses, substations, construction yards. It **detects** threats at the
edge in under two seconds, **understands** them in context, **acts** on its own
(notify, escalate, trigger), and seals **court-grade, tamper-evident evidence**.

A detector answers *"is there a gun in frame 4?"* Sentinel answers:

> *"An unfamiliar person entered the east yard at 2am, moved to avoid the camera,
> lingered by the generator with no scheduled work. The clip is sealed and
> verifiable. I have already called the owner and the guard service."*

## Why unattended sites

When no human is watching the feed, the priorities invert:

- **Reliability beats features** — the box is the only thing watching; it cannot
  silently die.
- **Notification is the product** — there is no operator; the alert *is* the value.
- **Evidence integrity is the moat** — "here is a video" is worthless for a claim
  or the police; "here is a video proven untampered" is the sellable thing.
- **Silence is an alarm** — a camera unplugged or a site gone quiet is itself a threat.

## Architecture — six layers

| Layer | Name | What it does |
|-------|------|--------------|
| L1 | Perception | A fleet of models — detection, tracking, re-ID, pose/action, anomaly, audio, fire/smoke |
| L2 | Reasoning brain | Fuses events into *situations*; VLM interpretation; context engine; agentic response |
| L3 | Learning systems | Continual per-site learning, federated across the fleet, active learning, synthetic data |
| L4 | Edge engineering | Real-time multi-camera inference, quantization, cascade gating, offline-first, OTA |
| L5 | Evidence chain | Sign-at-source, immutable Merkle anchor, chain-of-custody, verifiable redaction |
| L6 | Platform | Multi-tenant SaaS, digital-twin awareness, notifications, integrations, compliance |

Full detail in [VISION.md](VISION.md). Phased delivery plan in [ROADMAP.md](ROADMAP.md).
Researched technology choices (with licensing and end-of-life loopholes closed) in
[DECISIONS.md](DECISIONS.md).

## Status

**Phases 2.0 through 2.6 are delivered and tested — 95/95 tests passing.** A
real, GPU-fine-tuned detector (YOLO12s, mAP50-95 0.696) runs live in a
pipeline that also signs and Merkle-anchors evidence, survives network loss
via a store-and-forward outbox, flags a silent camera as its own alarm,
generates real-time descriptions with schedule-based false-alarm suppression,
pushes events over SSE with working notifications, and has a federated
learning path verified to generalize to new sites better than a solo model.
The rest of L1's model fleet (`perception/`) — fall detection, re-ID, anomaly,
audio, fire/smoke — now has real, free, tested implementations too, each with
a documented upgrade path to the heavier DECISIONS.md-chosen model.

See **[PROJECT_STATUS.md](PROJECT_STATUS.md)** for the precise line between
what's real/tested, what's an honest stub waiting on a real infrastructure
decision (a VLM server, a paid API key, edge hardware), and what's
structurally gated on having an actual deployed fleet. [ROADMAP.md](ROADMAP.md)
has the full phase-by-phase detail.

## Cost to build right now: $0

Every component has a free, self-hosted substitute for development — same
interfaces, no purchase, no cloud bill. Hardware and paid tiers are an **upgrade**,
applied later without a redesign. Full mapping in [DECISIONS.md](DECISIONS.md) D8.

| Target (later) | Build with now (free) |
|---|---|
| Jetson edge hardware | Your own CPU, or free Colab/Kaggle GPU for training |
| IP cameras | Laptop webcam |
| Frontier VLM escalation tier | Disabled — Qwen2.5-VL self-hosted handles everything |
| Lambda + DynamoDB | **Keep as-is** — both are always-free (permanent, not time-limited) at dev scale |
| API Gateway | Local FastAPI route or direct `boto3` invoke — its free tier is 12-months-only, then bills from request one |
| S3 + Object Lock | Local filesystem or self-hosted MinIO |
| OpenTimestamps / IPFS | Already free — used as-is |

Set a $1 AWS Billing budget alarm regardless — costs nothing, catches surprises.

## Technology choices (researched — see [DECISIONS.md](DECISIONS.md))

Chosen on **capability first** — best model per tier, license deferred to
commercialization. Depends on **no discontinued service**:

- **Detection:** bake-off between RF-DETR and YOLO12 on the chosen edge box; Co-DETR/DINO-class for max-accuracy cloud re-analysis
- **Pose/falls:** RTMPose (edge real-time) + ViTPose (max-accuracy cloud)
- **Tracking / re-ID:** ByteTrack + OSNet
- **Reasoning VLM:** frontier API (Claude/GPT-4o-class) for top-tier reasoning; Qwen2.5-VL 3B/7B on-edge for offline/low-latency
- **Federated learning:** Flower
- **Edge hardware:** Jetson Orin NX 16GB reference, Orin Nano 8GB cost tier, Pi 5 + Hailo-8 entry tier
- **Evidence:** C2PA + OpenTimestamps + S3 Object Lock (WORM) — **not** AWS QLDB (discontinued 31 Jul 2025)

Permissive runner-ups (RF-DETR, Qwen2.5-VL) stay benchmarked and wired in for a
mechanical swap if/when the project is productized.

## Still open (product-owner call)

- **Pilot vertical:** remote home (recommended) vs. warehouse
- **Public-chain anchoring** allowed, or self-hosted immutable store for air-gapped clients?

## Quickstart (Phase 2.0 — tested, runs today, $0 cost)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# terminal 1 — local backend (replaces AWS API Gateway; see DECISIONS.md D8)
uvicorn cloud.backend.app:app --port 8000

# terminal 2 — edge pipeline against your webcam
python -m edge.main
```

This runs the real pipeline end to end: webcam → motion-based detection
(zero-download default; Phase 2.1 swaps in the trained RF-DETR/YOLO12 model) →
centroid tracking → event debouncing → a pre/post-event clip recorded to
`./data/clips/` → the clip hashed and signed (`evidence/signing.py`) → the
event posted to the local backend → queryable at `GET /events`.

Run the tests (no camera or GPU required):

```bash
python -m pytest tests/ -v
```

## Phase 2.1 — perception core

`training/` has fine-tune scripts for both D1 bake-off candidates (RF-DETR,
YOLO12), a `yolo_to_coco.py` converter that closes the RF-DETR gap on the VOC
path, a dataset plan for the remote-home pilot, and a real
false-alarms-per-camera-per-day eval — see
[training/README.md](training/README.md) for the full scope and what's been
verified to actually run.

```bash
pip install -r training/requirements-training.txt
python -m training.train_yolo12 --data <data.yaml> --output-dir ./data/models/yolo12
python -m training.train_rfdetr --dataset-dir <coco-dir> --output-dir ./data/models/rfdetr
```

For the real GPU run, open [training/colab_finetune.ipynb](training/colab_finetune.ipynb)
in Google Colab (free T4) — it trains both candidates, runs the bake-off, and
exports weights. Drop the resulting `best.pt` in and the pipeline uses it:

```bash
SENTINEL_DETECTOR_BACKEND=model SENTINEL_DETECTOR_MODEL_PATH=./data/models/best.pt python -m edge.main
```

The model-backed detector (`edge/detector.py ModelDetector`) is implemented and
verified running real YOLO inference — a COCO-pretrained `yolo12n.pt` already
detects `person`/`vehicle`/`animal`; the Colab fine-tune adds `package` and
domain adaptation.

## Phases 2.2–2.6 — evidence, resilience, reasoning, platform, learning

All delivered and tested this session — see [PROJECT_STATUS.md](PROJECT_STATUS.md)
for the verification performed on each:

- **Evidence (2.2)** — `evidence/merkle.py` + `evidence/custody.py` +
  `evidence/daily_anchor.py`: Merkle-anchored daily hashes (real OpenTimestamps
  submissions), tamper-evident chain-of-custody.
- **Edge resilience (2.3)** — `edge/outbox.py`: store-and-forward so a network
  outage queues events instead of losing them; `/heartbeat` + `/sites/status`:
  a silent camera is its own alarm.
- **Reasoning (2.4)** — `reasoning/context.py` + `reasoning/describe.py`:
  schedule-based false-alarm suppression; real event descriptions via both a
  free template tier and a real local Qwen2.5-VL tier (Ollama), verified
  producing genuine scene descriptions from a real image; a paid frontier VLM
  tier remains an honest stub pending an API key. `edge/description_worker.py`
  keeps the ~11s VLM call off the real-time frame loop — alerts fire
  instantly on the template description, then get enriched asynchronously.
- **Platform (2.5)** — `GET /events/stream` (SSE, replacing polling),
  `cloud/backend/notifications.py` (pluggable alert channels), multi-tenant
  `org_id`.
- **Learning (2.6)** — `learning/federated_sim.py`: FedAvg proven to help a
  brand-new site generalize better than an existing site's solo model.

There is one test entry point, matching the real deployment shape: the live
webcam pipeline (`python -m edge.main`) against the local backend — see the
Quickstart above. A single live dashboard at `/` (`cloud/backend/static/dashboard.html`)
ties everything together in one page: a **live annotated tracking view**
(real detection boxes streamed from the actual pipeline via
`edge/live_frame_streamer.py` + `GET /live-frame/stream`, MJPEG), a raw
browser webcam preview (visual only, not the detection source), the
real-time event feed with descriptions/severity over SSE, and a raw event
log. Verified end-to-end in a real browser: a posted event and a
description-enrichment PATCH both appeared live with no refresh, and a
pushed annotated frame rendered and moved correctly in the tracking panel.
The earlier standalone image-upload test UI was removed in favor of this
single path.

## Testing & scoring

[TESTING.md](TESTING.md) is the protocol for testing on **unseen, real-world,
live footage**, with three separately-scored layers (code in
[eval/scoring.py](eval/scoring.py), unit-tested):

- **Detection rate** (L1) — precision/recall/F1 per class + false-alarms-per-camera-per-day
- **Description rate** (L2) — subject/action/severity accuracy + hallucination rate
- **System score** — end-to-end: right alert, right severity, sealed evidence, in time

See [ROADMAP.md](ROADMAP.md) for the full phase breakdown.
