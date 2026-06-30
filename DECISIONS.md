# Sentinel — Technical Decision Record

Researched recommendations for the open decisions in [VISION.md](VISION.md) and
[ROADMAP.md](ROADMAP.md). Each decision states the choice, the reasoning, the
alternatives weighed, and — importantly — the **loophole it closes**. Sourced from
current (2025–2026) benchmarks and vendor docs; see Sources at the end.

Guiding constraint from the product owner: *use the best resource available, and
leave no loopholes.* For a commercial product sold to industry, two loopholes
dominate and drive most choices below:

1. **Licensing** — an AGPL component forces open-sourcing the whole product or
   buying a per-seat commercial licence. This must be avoided by default.
2. **Vendor end-of-life** — building the evidence moat on a discontinued service
   (e.g. AWS QLDB) is an existential risk.

---

## D1 — Detection model: **RF-DETR (Apache-2.0)** as default; YOLO11 only under enterprise licence

**Choice:** RF-DETR is the default detector. Ship YOLO11 only if an Ultralytics
Enterprise Licence is purchased.

**Why:**
- **Licensing is the whole point.** Ultralytics YOLO (v8/11/12) is **AGPL-3.0**.
  Using it in a closed, commercial product means either open-sourcing all of
  Sentinel or buying an enterprise licence — community reports put a single-developer
  quote around **$5,000/year**, scaling with deployment. RF-DETR (Roboflow, released
  March 2025) is **Apache-2.0** — free for commercial, closed-source use, forever.
- **It's not a compromise on quality.** RF-DETR medium reaches **54.7 mAP@50:95**
  on COCO — matching YOLO11x — while running real-time (RF-DETR nano ~431 FPS /
  2.3 ms, medium ~221 FPS / 4.5 ms on a T4 with TensorRT). It's a transformer
  detector (DETR family), so it's NMS-free, which simplifies the edge pipeline.

**Alternatives weighed:**
- *YOLO11/YOLO12* — excellent and slightly faster at the small end, but AGPL.
  Keep as an option for research or if the enterprise licence is bought.
- *YOLO-NAS (Deci)* — permissive-ish but the company was acquired by NVIDIA and
  the project is effectively unmaintained. Rejected.

**Loophole closed:** no AGPL contamination in the shipped product.

---

## D2 — Pose / fall detection: **RTMPose (Apache-2.0)**, not YOLO11-pose

**Choice:** RTMPose (from OpenMMLab / MMPose) for keypoint estimation; falls and
other postures derived by a rule/classifier on keypoint geometry.

**Why:** This is the **hidden licensing loophole** — even teams that switch the
detector to RF-DETR often keep `YOLO11-pose`, which is *also AGPL*, re-contaminating
the product. RTMPose is Apache-2.0, designed for real-time edge inference, and
exports cleanly to ONNX/TensorRT. Falls must be pose-based (keypoint geometry:
torso angle, vertical velocity, time-on-ground), never a bounding-box class.

**Loophole closed:** the pose path is permissive too — not just the detector.

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

## D4 — Reasoning VLM (L2): **Qwen2.5-VL (Apache-2.0)** — 3B at edge, 7B in cloud

**Choice:** Qwen2.5-VL-3B for on-hub event interpretation; Qwen2.5-VL-7B in the
cloud for richer reasoning and natural-language queries. Evaluate SmolVLM /
Moondream for the smallest boxes.

**Why:** Qwen2.5-VL (3B/7B/72B) is Apache-2.0, has strong chain-of-thought and
video-understanding ability, and the 3B fits an Orin-class device. SmolVLM and
Moondream are the efficiency-first fallbacks for constrained hardware. VLMs are
also the emerging engine for training-free anomaly detection (D3), so this model
does double duty.

**Loophole closed:** the reasoning layer is permissive and self-hostable — no
per-call dependency on a closed API for the core decision path (a cloud API like
Claude can still be an optional premium tier, but the product works without it).

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

## Net licence posture

Every component in the **default shipping configuration is permissive**
(Apache-2.0 / MIT): RF-DETR, RTMPose, ByteTrack, OSNet, Qwen2.5-VL, Flower. The
only AGPL options (YOLO11/12) are opt-in behind a purchased enterprise licence.
The evidence chain depends on **no discontinued or single-vendor-locked service**.

## Decisions still genuinely open (need product-owner input)

- **Pilot vertical** — remote home vs. warehouse (changes detection classes,
  retention rules). Recommended: remote home first (smallest hardware, clearest
  buyer, fastest pilot).
- **Public-chain anchoring allowed?** Fine for commercial/insurance; for
  defence/air-gapped clients swap OpenTimestamps for a self-hosted ImmuDB/ScalarDL.
- **Enterprise YOLO licence** — only worth buying if a measured accuracy/latency
  gap on the *chosen edge box* justifies it over RF-DETR. Default: don't.

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
