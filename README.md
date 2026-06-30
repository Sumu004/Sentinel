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

## Status

Early-stage. Currently seeding vision and roadmap; Phase 2.0 (runnable
single-site pipeline) is the first buildable milestone.

## Open decisions

- **Edge hardware:** Jetson Orin · Intel NUC + OpenVINO · Pi 5 + Hailo
- **Model licence:** AGPL YOLO11 · permissive RF-DETR
- **Pilot vertical:** remote home · warehouse
