# Sentinel — Roadmap

Started from a broken single-camera MVP script. Building it up in phases,
each one shipping something that runs.

| Phase | Layer | What | Status |
|---|---|---|---|
| 0 | — | Original MVP | Broken — fixed in 2.0 |
| 2.0 | — | Runnable single-camera pipeline (source → detect → track → record → sign → backend) | Done |
| 2.1 | L1 | Perception core — trained detector (YOLO12), RF-DETR bake-off path, training scripts | Mostly done — detector trained and live; `package` class and quantization still open |
| 2.2 | L5 | Evidence hardening — Merkle anchor, chain of custody, OpenTimestamps | Done |
| 2.3 | L4 | Edge resilience — store-and-forward outbox, heartbeat/silent-site detection | Done |
| 2.4 | L2 | Reasoning — context rules, event descriptions (template + local VLM) | Done for the free tier; frontier VLM is a stub pending an API key |
| 2.5 | L6 | Platform — live dashboard, SSE push, notifications, multi-tenant | Done |
| 2.6 | L3 | Learning — federated learning simulation | Simulation done; real fleet-scale learning needs an actual fleet |

L1 perception fleet (pose/fall, re-ID, anomaly, audio, fire/smoke) — all
shipped with free, tested defaults; heavier models are a documented upgrade
path once there's a reason to swap.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the detailed breakdown of
what's tested vs. what's blocked on real hardware/data/an account.

## Next up (needs you, not more code)

1. Run the Colab bake-off (RF-DETR vs YOLO12) for real
2. Pick a VLM tier — local Qwen setup or a frontier API key
3. Get `package` class training data
4. Pick edge hardware
5. Deploy to a real site
