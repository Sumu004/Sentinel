# Sentinel — Vision

An edge AI guard for places nobody is watching — remote homes, warehouses,
substations, construction yards.

A plain detector answers "is there a person in frame 4?" Sentinel answers:
"An unfamiliar person entered the yard at 2am, lingered by the generator,
left when the light came on. Clip sealed and verifiable. Owner and guard
service already notified."

## Why this is different

- No one's watching the feed live, so the alert **is** the product.
- A camera going silent (unplugged, covered, dead) is itself a threat.
- Evidence has to be provable — "here's a video" is worthless without proof
  it wasn't edited.
- False alarms kill trust fast, so understanding context matters as much as
  detecting objects.

## The six layers

| Layer | Name | What it does |
|---|---|---|
| L1 | Perception | Detection, tracking, re-ID, pose/fall, anomaly, audio, fire/smoke |
| L2 | Reasoning | Turns raw events into understood situations, VLM descriptions, context rules |
| L3 | Learning | Per-site and federated learning across the fleet |
| L4 | Edge | Real-time multi-camera inference, offline-first, quantization |
| L5 | Evidence | Sign-at-source, Merkle-anchored, tamper-evident chain of custody |
| L6 | Platform | Multi-tenant dashboard, notifications, integrations |

## Target markets

Remote homes, warehouses (off-hours), utilities/substations, construction
sites, retail, critical infrastructure.

## Build order

Perception → evidence → reasoning → platform → learning → fleet hardening.
Full phase plan in [ROADMAP.md](ROADMAP.md). Tech choices in
[DECISIONS.md](DECISIONS.md).
