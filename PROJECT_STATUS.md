# Sentinel — Project status

A single, honest answer to "is the project done?" A six-layer architecture
(VISION.md) spanning fleet learning, custom edge hardware, and paid reasoning
APIs is not a thing that finishes in one build session — but everything that
*can* be built and verified without a real deployment, real hardware, or a
paid account has been. This document draws the line precisely: what's real
and tested, what's an honest stub waiting on your infrastructure decision, and
what structurally cannot exist without a real fleet.

Full phase-by-phase detail is in [ROADMAP.md](ROADMAP.md); this is the
summary and the "what would it take" answer for what's left.

## Test suite: 44/44 passing

10 test files, covering tracking/debouncing, storage, evidence (signing,
Merkle proofs, chain-of-custody tamper detection), the outbox, heartbeats,
reasoning (context engine, description), notifications, federated learning,
and the YOLO→COCO converter. Every module below that says "tested" has a
corresponding test in `tests/`.

## Delivered and verified this project

| Phase | What | Real verification performed |
|---|---|---|
| **2.0** Runnable pipeline | Source→detect→track→debounce→record→sign→POST, local FastAPI+SQLite backend | Full pipeline run against synthetic video with real-time pacing; event reached the backend and a signed clip was produced |
| **2.1** Perception core | Real GPU-fine-tuned YOLO12s (Pascal VOC), `ModelDetector` wired into the pipeline, RF-DETR bake-off path | Trained for real on Colab T4 (mAP50-95 0.696); the fine-tuned model beat the plain pretrained baseline on a real test image; RF-DETR path closed with a tested YOLO→COCO converter |
| **2.2** Evidence hardening | Merkle-tree daily anchor, chain-of-custody log, OpenTimestamps | Real submission to public Bitcoin-calendar servers; tamper-detection verified by editing a custody-log row directly via SQL and confirming the chain check catches it |
| **2.3** Edge resilience | Store-and-forward outbox, heartbeat/silent-site detection | Killed the backend mid-run, confirmed events queue instead of vanishing, confirmed retry drains the queue on reconnect; confirmed a site flips to `silent: true` after its heartbeat threshold elapses |
| **2.4** Reasoning (free tier) | Rule-based context engine (schedule suppression), template event descriptions | Verified correct suppression on matching day/time/label, correct non-suppression otherwise; descriptions verified for normal, suppressed, and low-severity cases |
| **2.5** Platform | Real-time SSE push, notification engine, multi-tenant `org_id` | Posted an event, watched it arrive on a live SSE connection instantly; confirmed console alert fired with correct severity; confirmed org-filtered query isolation |
| **2.6** Learning | Federated learning simulation (FedAvg) | Empirically verified across 5 seeds that a federated model generalizes better to a brand-new site than an existing site's solo model (0.920 vs 0.898 mean accuracy) — and the *first* experiment design was wrong and was caught before being reported, not just accepted |
| Testing UI | Light-mode image-upload detection tester | Verified end-to-end: real fine-tuned model, real annotated output, rendered and confirmed in a live preview |

## Honest stubs — the interface is real, the backing infrastructure isn't

These are not gaps in the code; they're places where the correct next step
requires a decision or an account only you can make/create. Each raises a
clear error explaining exactly what's missing, rather than silently doing
nothing or faking a result.

| Component | What exists | What it's waiting on |
|---|---|---|
| `reasoning/describe.py QwenLocalDescriber` | Full interface, wired into the pipeline | A running local VLM server (Ollama/vLLM serving Qwen2.5-VL) — real GPU infrastructure to stand up |
| `reasoning/describe.py FrontierDescriber` | Full interface | A paid frontier API key (Claude/GPT-4o-class) — a deliberate, consent-gated spend decision |
| `cloud/backend/notifications.py SMSChannel` | Full interface | A paid SMS provider account (Twilio et al.) |
| ONNX→TensorRT quantization | Export cell exists in the Colab notebook | Real Jetson/edge hardware to quantize against — can't be tested on a dev laptop |
| `package` detection class | Full dataset plan, class-suppression logic ready | Either a decent Roboflow dataset (checked — the options are weak) or your own captured porch footage |

## What's structurally gated on a real fleet — can't be simulated further

Phase 2.6's remaining items (continual per-site learning, active-learning
queue, synthetic rare-event data, OTA updates, IaC, observability,
privacy/compliance tooling) all require **an actual deployed fleet** —
multiple real sites generating real traffic. There is no honest way to
"build and verify" fleet operations with zero fleet. The federated-learning
*mechanism* is proven (above); operating it at scale is a different problem
that starts the day a second physical site goes live.

Similarly: **spatio-temporal fusion across cameras** (Phase 2.4) needs a
second camera to make sense of, and **cascade inference** (Phase 2.3) needs a
real detector's real latency profile on real edge hardware to tune correctly
— both are legitimately blocked on infrastructure that doesn't exist yet, not
on unwritten code.

## What only you can do next

1. **Run the Colab bake-off for real.** The notebook now trains both YOLO12
   and RF-DETR on the same data (VOC by default, or your own Roboflow
   dataset) — running it is a real GPU job only you can kick off.
2. **Decide the VLM tier.** Set up a local Ollama/vLLM server for
   Qwen2.5-VL, or provide a frontier API key — either unlocks Phase 2.4's
   richer reasoning tier.
3. **Get `package` data.** Either browse Roboflow Universe for the least-bad
   option, or capture ~20 minutes of your own porch footage — the more
   reliable path per the earlier dataset research.
4. **Pick edge hardware** (Jetson Orin NX / Nano, Pi 5 + Hailo — DECISIONS.md
   D5) to unlock real quantization and latency-tuned cascade inference.
5. **Deploy to a real site.** Everything gated on "a real fleet" starts here.

## Bottom line

Every layer of the architecture that can exist as tested, working code today,
does. What's left is real infrastructure and real-world data collection —
work that requires you, hardware, or a deployed site, not more code from this
session.
