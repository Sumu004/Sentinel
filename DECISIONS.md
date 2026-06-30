# Sentinel — Technical Decision Record

Researched recommendations for the open decisions in [VISION.md](VISION.md) and
[ROADMAP.md](ROADMAP.md). Each decision states the choice, the reasoning, the
alternatives weighed, and — importantly — the **loophole it closes**. Sourced from
current (2025–2026) benchmarks and vendor docs; see Sources at the end.

Guiding constraint from the product owner: **pick the best-performing technology
available — license is secondary, release/commercialization is a later concern.**
So every choice below is ranked on capability first. License is recorded as an
*attribute* (it matters the day we commercialize), not a gate. The one loophole
that still binds regardless of license posture:

- **Vendor end-of-life** — building the evidence moat on a discontinued service
  (e.g. AWS QLDB) is an existential risk no license stance can excuse. Still closed
  in D6.

> Performance-first note: where the best model is AGPL (Ultralytics YOLO) or a paid
> API (frontier VLMs), we use it now and treat licensing as a commercialization-time
> task — swap to the permissive runner-up only if/when we productize. Both options
> are kept side by side so the swap is mechanical.

---

## D1 — Detection model: **best per tier via a bake-off** — RF-DETR and YOLO12 both in

**Choice:** run a measured bake-off (accuracy + latency) on the *chosen edge box*
and ship whichever wins per tier. Lead candidates: **RF-DETR (transformer, NMS-free)**
and **YOLO12 (attention-enhanced CNN)**. Add a heavy cloud re-scorer for max accuracy.

**Why (capability-first):**
- The two are genuinely neck-and-neck and each wins different cases: RF-DETR
  medium reaches **54.7 mAP@50:95** (matching YOLO11x) and runs real-time
  (nano ~431 FPS / 2.3 ms, medium ~221 FPS / 4.5 ms on T4+TensorRT); YOLO12 clusters
  in the mid-latency/high-accuracy region and is often faster at the small end with
  the mature Ultralytics tooling. Picking on a benchmark beats picking on theory.
- **Best accuracy, full stop (cloud re-analysis):** for offline forensic re-scoring
  of flagged clips, run a heavyweight detector (e.g. **Co-DETR / DINO**-class, which
  top the COCO leaderboard) where latency doesn't matter. Edge stays real-time;
  cloud gets the last few points of recall.
- RF-DETR being NMS-free still simplifies the edge pipeline — a tiebreaker, not a gate.

**License note (secondary):** YOLO12 is AGPL-3.0, RF-DETR is Apache-2.0. Irrelevant
now; at commercialization either buy the Ultralytics enterprise licence or fall back
to the (already-benchmarked) RF-DETR. Kept side by side so the swap is mechanical.

**Rejected:** YOLO-NAS — project effectively unmaintained after the NVIDIA acquisition.

---

## D2 — Pose / fall detection: **RTMPose at the edge, ViTPose for max accuracy**

**Choice:** **RTMPose** (OpenMMLab) as the real-time edge pose model; **ViTPose**
for the highest-accuracy cloud re-analysis of flagged events. Falls derived from
keypoint geometry (torso angle, vertical velocity, time-on-ground), never a class.

**Why (capability-first):** RTMPose has the best accuracy/latency trade-off for
real-time edge keypoints and exports cleanly to ONNX/TensorRT. ViTPose is the
accuracy ceiling for offline verification where latency is free. (Both happen to
be Apache-2.0; YOLO11-pose is the AGPL alternative and is the weaker choice on
merit anyway, so license doesn't even enter into it here.)

---

## D3 — Tracking, re-ID, anomaly: permissive stack

| Job | Choice | Licence | Note |
|-----|--------|---------|------|
| Multi-object tracking | ByteTrack | MIT | Lightweight, edge-proven |
| Re-identification | OSNet (via Torchreid) | MIT | SOTA-lightweight; 2025 edge work pairs YOLO-detect on-device + OSNet re-ID on a hub |
| Video anomaly (unsupervised) | Autoencoder/prediction baseline → VLM-assisted | permissive | 2025 SOTA is memory-driven, zero-shot, real-time (e.g. "Flashback") and increasingly VLM-assisted |

Re-ID and heavy anomaly models can run on a **site hub**, not every camera —
2025 edge designs offload detection to the camera and feature extraction/identity
to a local hub. This keeps the per-camera box cheap.

---

## D4 — Reasoning VLM (L2): **frontier API for best reasoning, Qwen2.5-VL on-edge**

**Choice:** use a **frontier hosted VLM (Claude / GPT-4o-class)** for the cloud
reasoning + natural-language-query tier where quality is paramount; run
**Qwen2.5-VL-3B/7B** on the site hub for low-latency, offline-capable interpretation.
SmolVLM / Moondream for the smallest boxes.

**Why (capability-first):** for the "understands the place" layer, a frontier model
gives the best situational reasoning and clean natural-language alerts/queries —
worth the per-call cost for the moments that matter. Qwen2.5-VL (Apache-2.0, strong
CoT + video understanding) handles the edge tier so the system still works offline
and isn't paying an API for every quiet frame. VLMs also power training-free anomaly
detection (D3), so the edge model does double duty.

**Architecture note:** keep the edge VLM as the always-on path and the frontier API
as an escalation tier (called only on high-severity / ambiguous events). Best
reasoning where it counts, no cloud dependency for basic operation.

---

## D5 — Edge hardware: **tiered**, with Jetson Orin NX 16GB as the reference target

**Choice:** three tiers so the same software targets cheap and capable sites:

| Tier | Hardware | Compute | Real-time capacity | Use |
|------|----------|---------|--------------------|-----|
| **Reference / multi-camera** | **Jetson Orin NX 16GB** | up to 117 TOPS (Super Mode), 102 GB/s | full stack incl. 3B VLM, several cameras | warehouses, multi-cam sites |
| Cost / single-dual camera | Jetson Orin Nano 8GB | up to 67 TOPS (Super Mode), 68 GB/s | detector + tracker + pose | remote homes |
| Ultra-low-cost | Raspberry Pi 5 + Hailo-8 | 26 TOPS | ~77 FPS YOLOv8s, single camera | simple perimeter |

**Why Orin NX as reference:** it runs the *whole* perception stack plus a small
VLM with CUDA/TensorRT maturity, shares a pinout with Orin Nano (one carrier board
serves both tiers), and sits in a 10–25 W envelope viable for solar/battery remote
sites. AGX Orin (up to 275 TOPS) is reserved for heavy multi-camera hubs only —
overkill and over-budget for a single site. The Pi 5 + Hailo tier is real and
useful (26 TOPS, ~77 FPS YOLOv8s) but the Hailo toolchain is more restrictive and
won't host a VLM — so it's detection-only edge.

**Loophole closed:** "fast and safer" is met without locking to one box — the
shared CUDA stack means Nano and NX run identical code, and the Hailo tier keeps a
low-cost entry without forking the whole codebase.

---

## D6 — Evidence anchor: **C2PA + OpenTimestamps + S3 Object Lock**, NOT AWS QLDB

**Choice:** sign clips at source with **C2PA 2.x content credentials** (SHA-256 +
X.509 device cert), store originals in **S3 with Object Lock (WORM / compliance
mode)**, and anchor the daily Merkle root to a public, verifiable timestamp via
**OpenTimestamps (Bitcoin)** and/or a transparency log (Sigstore Rekor). Optional
IPFS/Filecoin pin for decentralised availability.

**Why — this closes the biggest loophole in the original plan:**
- **AWS QLDB is discontinued.** Support **ended 31 July 2025** — quietly, with no
  migration path beyond "use Aurora PostgreSQL," which does *not* keep an immutable
  record (history must be generated as external audit data). Building the evidence
  moat on QLDB would have been building on a dead service.
- **C2PA is the industry standard** for tamper-evident media provenance (Adobe,
  Arm, BBC, Intel, Microsoft; spec 2.x). Hash-at-capture + signed manifest means
  any later splice, audio edit, or deepfake injection breaks the hash — immediately
  and verifiably. It explicitly does *not* require a blockchain.
- **Public anchoring without vendor lock-in:** OpenTimestamps gives a free,
  Bitcoin-backed, independently verifiable timestamp of the Merkle root. S3 Object
  Lock gives WORM storage that even the account owner cannot alter or delete within
  the retention window — defensible for courts and insurers.

**Alternatives weighed:** Azure SQL Ledger (good, but Azure lock-in); ImmuDB /
ScalarDL / Dolt (self-hostable immutable stores — keep as the on-prem/air-gapped
option where no public chain is allowed).

**Loophole closed:** no discontinued service, no single-cloud lock-in, and an
integrity guarantee that holds even against an insider with cloud-account access.

---

## D7 — Federated & continual learning (L3): **Flower (Apache-2.0)**

**Choice:** Flower for cross-site federated learning.

**Why:** Flower is the leading FL framework (Apache-2.0), runs across cloud, mobile,
and edge, reaches 90–95% of centralised accuracy while raw video never leaves the
site, and aligns with GDPR / EU AI Act constraints — which is exactly the privacy
posture the unattended-surveillance use case demands. Start with FedAvg; the known
cost is communication overhead and a learning curve beyond default strategies.

---

## Net posture (performance-first)

The stack is chosen on **capability per tier**: best edge real-time model + best
heavyweight model for offline re-analysis, with a frontier VLM for top-tier
reasoning. License is recorded per component but does not gate any choice now —
where the best option is AGPL (YOLO12) or a paid API (frontier VLM), we use it and
keep the benchmarked permissive runner-up (RF-DETR, Qwen2.5-VL) wired in for a
mechanical swap if/when we commercialize. The only hard constraint that survives is
**no discontinued/dead-end service** in the evidence chain (D6).

## Decisions still genuinely open (need product-owner input)

- **Pilot vertical** — remote home vs. warehouse (changes detection classes,
  retention rules). Recommended: remote home first (smallest hardware, clearest
  buyer, fastest pilot).
- **Public-chain anchoring allowed?** Fine for commercial/insurance; for
  defence/air-gapped clients swap OpenTimestamps for a self-hosted ImmuDB/ScalarDL.
- **Commercialization-time licensing** — deferred. When/if we productize, revisit
  the AGPL components (YOLO12) and paid APIs (frontier VLM); the permissive
  alternatives are already benchmarked and wired in.

## Sources

- [Jetson Orin module comparison (ProventusNova)](https://proventusnova.com/blog/jetson-orin-agx-vs-orin-nx-vs-orin-nano/) · [Forecr Jetson spec comparison](https://www.forecr.io/blogs/embedded-systems/nvidia-jetson-comparison)
- [Raspberry Pi 5 + Hailo-8/8L benchmarks (Seeed Studio)](https://wiki.seeedstudio.com/benchmark_on_rpi5_and_cm4_running_yolov8s_with_rpi_ai_kit/) · [Edge AI comparison (Geeky Gadgets)](https://www.geeky-gadgets.com/ai-edge-sbc-hardware-comparison/)
- [RF-DETR vs YOLO comparison (Roboflow playground)](https://playground.roboflow.com/models/compare/rf-detr-vs-yolo11) · [RF-DETR vs YOLOv12 study (arXiv 2504.13099)](https://arxiv.org/pdf/2504.13099)
- [Ultralytics licensing](https://www.ultralytics.com/license) · [Enterprise pricing discussion](https://github.com/orgs/ultralytics/discussions/7440)
- [AWS discontinues QLDB (InfoQ)](https://www.infoq.com/news/2024/07/aws-kill-qldb/) · [QLDB alternatives (DoltHub)](https://www.dolthub.com/blog/2024-08-12-qldb-deprecated-alternatives/)
- [C2PA chain of custody (OpenFox)](https://www.openfox.com/news/how-blockchain-secures-chain-of-custody-in-an-era-of-ai-deepfakes/) · [Blockchain timestamping 2025 (OriginStamp)](https://originstamp.com/en/blog/reader/blockchain-timestamping-2025-data-integrity)
- [SmolVLM / small VLMs 2025 (arXiv 2504.05299)](https://arxiv.org/html/2504.05299v1) · [Qwen2.5-VL](https://qwen.ai/blog?id=qwen2.5-vl)
- [Flower federated learning (arXiv 2007.14390)](https://arxiv.org/pdf/2007.14390) · [Felicis on Flower](https://www.felicis.com/blog/investing-in-flower)
- [Real-time person re-ID at the edge (Springer 2025)](https://link.springer.com/article/10.1007/s10044-025-01492-z)
- [Video anomaly detection via VLMs (arXiv 2510.02155)](https://arxiv.org/html/2510.02155v1)
