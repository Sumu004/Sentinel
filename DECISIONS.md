# Sentinel — Tech Decisions

Short version: pick the best tool for each job, keep everything free to
build and test, swap to paid/hardware only when actually deploying.

| # | Area | Choice | Why (short) |
|---|---|---|---|
| D1 | Detection | RF-DETR + YOLO12 (bake-off, best wins) | Both near the top of current benchmarks; run both, keep the winner |
| D2 | Pose / falls | RTMPose (edge), ViTPose (cloud) | Best accuracy/latency tradeoff for each tier |
| D3 | Tracking + re-ID | ByteTrack + OSNet | MIT-licensed, lightweight, edge-proven |
| D4 | Reasoning VLM | Qwen2.5-VL (edge, free) + a frontier API (cloud, paid, escalation only) | Works offline for free; frontier tier only when quality really matters |
| D5 | Edge hardware | Jetson Orin NX (reference), Orin Nano (cheaper), Pi 5 + Hailo (cheapest) | Same software stack across all three |
| D6 | Evidence anchor | C2PA-style signing + OpenTimestamps + S3 Object Lock | Free, no vendor lock-in, holds up as tamper-evident proof. Not AWS QLDB — discontinued 2025 |
| D7 | Federated learning | Flower | Leading FL framework, keeps raw data on-site |
| D8 | Cost | Everything has a free substitute for dev (see below) | Ship for $0, upgrade to paid/hardware later without a rewrite |

## Free substitutes used today (D8)

| Real target | Free stand-in now |
|---|---|
| Jetson edge hardware | Your own CPU / free Colab or Kaggle GPU |
| IP cameras | Laptop webcam |
| Frontier VLM | Qwen2.5-VL, self-hosted |
| API Gateway | Local FastAPI |
| S3 + Object Lock | Local filesystem / MinIO |
| Lambda + DynamoDB | Kept as-is — always free at this scale |
| OpenTimestamps / IPFS | Already free |

Set a $1 AWS billing alarm regardless — costs nothing, catches surprises.

## Still open

- Pilot vertical: remote home vs. warehouse
- Public-chain anchoring vs. self-hosted store for air-gapped clients
- Edge hardware pick, model license (revisit at commercialization)
