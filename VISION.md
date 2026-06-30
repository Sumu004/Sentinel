# Project Sentinel — Vision

> From a detection script to an autonomous guard for places nobody is watching.

## The one sentence

An edge-first AI that watches remote, unattended sites — homes, warehouses,
substations, construction yards — **understands** what it sees, **acts** on its
own, and seals **court-grade, tamper-evident evidence**.

A detector answers *"is there a gun in frame 4?"* That is 20% of the problem.
Sentinel answers:

> *"An unfamiliar person entered the east yard at 2am, moved to avoid the
> camera, lingered by the generator with no scheduled work. Confidence 0.87.
> The 30-second clip is sealed and verifiable. I have already called the owner
> and the guard service."*

That is a **perception + reasoning + action** system. Everything in this
document builds toward that sentence.

## Why unattended sites

When no human is watching the feed, the priorities invert relative to a staffed
control room:

- **Reliability beats features** — if the box is the only thing watching, it
  cannot silently die. Watchdog, store-and-forward, offline buffering.
- **Notification is the product** — there is no operator at a wall of monitors;
  the alert *is* the value delivered.
- **Evidence integrity is the moat** — "here is a video" is worthless for an
  insurance claim, an OSHA/HSE incident, or the police. "Here is a video with a
  cryptographic timestamp proving it was not edited" is the sellable thing.

## Design principles

1. **Edge-first for speed and safety.** Inference runs on-site, so detection
   latency does not depend on the uplink. Raw video never leaves the site unless
   an event fires — only signed, hashed evidence clips go up. Less bandwidth,
   less privacy exposure, less attack surface.
2. **Silence is an alarm.** A camera unplugged, a lens covered, a site gone
   quiet — for an unattended location, the absence of signal is itself a threat.
3. **Understand, don't just detect.** False alarms are the product killer. Every
   bad 3am page trains the customer to ignore the system. The reasoning and
   context layers exist to drive the false-alarm rate toward zero.
4. **Provable by construction.** Evidence is hashed and signed at the moment of
   capture, anchored immutably, and every access is logged.

## The six layers

| Layer | Name | What it does |
|-------|------|--------------|
| L1 | Perception | A fleet of models — detection, tracking, re-ID, pose/action, anomaly, audio, fire/smoke |
| L2 | Reasoning brain | Fuses events across cameras and time into *situations*; VLM interpretation; context engine; agentic response |
| L3 | Learning systems | Continual per-site learning, federated across the fleet, active learning, synthetic data |
| L4 | Edge engineering | Real-time multi-camera inference, quantization, cascade gating, offline-first, OTA |
| L5 | Evidence chain | Sign-at-source, immutable Merkle anchor, chain-of-custody, verifiable redaction |
| L6 | Platform | Multi-tenant SaaS, digital-twin situational awareness, notifications, integrations, compliance |

### L1 — Perception stack

Not one model — a stack, each tuned to one job:

- **Detection** — fine-tuned **RF-DETR (Apache-2.0)** by default; YOLO11 only under a
  purchased enterprise licence (YOLO is AGPL). Closed-set, fully quantizable, low
  false-alarm rate. Open-vocabulary models (YOLO-World) are for prototyping and as a
  *teacher* for few-shot onboarding, not the runtime. See [DECISIONS.md](DECISIONS.md) D1.
- **Tracking + re-identification** — ByteTrack (MIT) + OSNet (MIT). The
  same person across cameras and across days.
- **Pose & action recognition** — **RTMPose (Apache-2.0)** + a temporal model (SlowFast /
  VideoMAE). Falls, climbing, fighting, tampering, concealment — *behaviour*,
  not objects. Falls are pose-based, never a bounding-box class. (RTMPose, not
  YOLO11-pose, which is also AGPL — see [DECISIONS.md](DECISIONS.md) D2.)
- **Anomaly detection (unsupervised)** — catches the threats nobody labelled:
  *"this has never happened here before."* The core value for guarding the unknown.
- **Audio event detection** — YAMNet / PANNs. Glass break, gunshot, scream.
  Works in the dark and around corners.
- **Fire / smoke / flood** — dedicated temporal model; benefits from flicker cues.

### L2 — Reasoning brain

- **Spatio-temporal fusion** — multi-camera homography so one intruder walking
  cam-1 → cam-3 is a single track on a site map, not three alerts.
- **VLM interpretation** — a vision-language model writes the human-readable
  description of each event clip.
- **Context engine** — knows the site's *normal*: schedules, geofences, roles,
  expected vehicles. An alert fires only when reality contradicts expectation.
  The single biggest false-alarm killer.
- **Agentic response** — an LLM agent decides severity, who to notify, whether
  to escalate or trigger a siren/light, and answers natural-language queries
  over the event store (*"show everyone who approached the dock after 10pm"*).

### L3 — Learning systems

- **Continual / online learning** — each site adapts to its own cameras without
  catastrophic forgetting.
- **Federated learning** — sites improve the global model without sending raw
  video to the cloud. Privacy-preserving by design.
- **Active learning** — the system surfaces its most uncertain clips for human
  labelling, maximising accuracy per annotation hour.
- **Synthetic data** — generate rare/dangerous scenarios (weapons, fire, night
  intrusion) you cannot collect enough of in the wild.
- **Few-shot site onboarding** — a customer points at a thing and the system
  learns it from a handful of examples.

The flywheel: `edge clips → active-learning queue → label → continual retrain →
federated aggregate → OTA push`. Smarter at every site, every week, automatically.

### L4 — Edge engineering

- Multi-camera real-time inference on one box; model sharing, batched inference,
  dynamic FPS (drop to 1fps when quiet, burst on motion).
- INT8 quantization + TensorRT / Hailo / OpenVINO per hardware target; distillation
  for the smallest boxes.
- **Cascade inference** — a cheap motion/anomaly trigger gates the expensive
  models, so idle time costs almost nothing (battery/solar-friendly).
- Offline-first: store-and-forward queue, ring buffer with pre-event footage,
  local event log. Cutting the internet must not blind or erase it.
- Tamper resistance: unplug, lens-cover, spray, reposition each raise their own
  alarm.
- Watchdog + heartbeat + OTA model/firmware updates across a fleet.

### L5 — Evidence chain

- **Sign at source** — hash + device-key-sign the clip on the camera the instant
  it is recorded.
- **Immutable anchor** — Merkle-tree the day's hashes; anchor the root via a public
  timestamp (OpenTimestamps/Bitcoin) and store originals in WORM (S3 Object Lock).
  Tampering with any clip breaks the tree. (Note: AWS QLDB was discontinued
  31 Jul 2025 — do not build on it. See [DECISIONS.md](DECISIONS.md) D6.)
- **Chain of custody** — every view, export, and access logged and signed.
- **Content provenance** — C2PA-compatible so evidence interoperates with courts,
  insurers, journalists.
- **Verifiable redaction** — blur faces for privacy while proving the rest is
  unedited.

### L6 — Platform

- Multi-tenant SaaS: org → site → zone → camera → event, with RBAC and audit.
- Live digital-twin site map — tracks on a floorplan, not a grid of feeds. One
  operator watches hundreds of sites.
- Real-time push (WebSocket/SSE), mobile app, escalation policies, two-way
  control (siren/lights/intercom/lock relays).
- Integrations: alarm panels, access control, ONVIF cameras, guard dispatch,
  insurance APIs, Slack/Teams/WhatsApp.
- No-code rule builder; marketplace of vertical model packs.
- Compliance engine: GDPR/DPDP retention & deletion, data-residency, consent.

## Target markets

- **Remote & vacation homes** — an autonomous guard that knows family from a
  stranger, calls you, seals evidence.
- **Warehouses & logistics (off-hours)** — intrusion, theft, PPE, fire across
  hundreds of sites on one dashboard.
- **Energy & utilities** — unmanned substations, pipelines, towers.
- **Construction sites** — theft (a huge problem), safety compliance, insurance
  evidence.
- **Retail chains & banks** — weapon, theft, after-hours, fleet-wide.
- **Critical infrastructure & defence** — on-prem, air-gapped, perimeter + behaviour.

## The moat

1. **Data flywheel** — a federated, continually-learning fleet competitors cannot
   replicate without the deployments.
2. **Evidence integrity** — provable, court-grade chain of custody.
3. **Reasoning layer** — understanding situations, not detecting objects, drives a
   false-alarm rate nobody else hits.
4. **Offline-first edge** — works where cloud-only competitors go blind.

## Sequencing

Build a believable vertical slice early (a remote-home pilot), then deepen layer
by layer. Full phase breakdown in [ROADMAP.md](ROADMAP.md).

`Perception core (L1) → Evidence chain (L5) → Reasoning/VLM (L2) →
Platform (L6) → Learning systems (L3) → fleet hardening (L4)`

## Open decisions

| Decision | Options | Impacts |
|----------|---------|---------|
| Edge hardware | Jetson Orin · Intel NUC + OpenVINO · Pi 5 + Hailo | Model size, quantization path |
| Model licence | AGPL YOLO11 · permissive RF-DETR | Whole codebase licensing |
| Primary vertical for pilot | Remote home · warehouse | Model classes, retention rules |
