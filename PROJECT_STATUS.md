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

## Test suite: 95/95 passing

13 test files, covering tracking/debouncing, storage (SQLite + a mocked-boto3
DynamoDB suite), evidence (signing, Merkle proofs, chain-of-custody tamper
detection), the outbox, heartbeats, reasoning (context engine, template +
Qwen-local description), notifications, federated learning, the YOLO→COCO
converter, and the L1 perception fleet (fall detection, re-ID, anomaly, audio,
fire/smoke). Every module below that says "tested" has a corresponding test
in `tests/`.

## Delivered and verified this project

| Phase | What | Real verification performed |
|---|---|---|
| **2.0** Runnable pipeline | Source→detect→track→debounce→record→sign→POST, local FastAPI+SQLite backend | Full pipeline run against synthetic video with real-time pacing; event reached the backend and a signed clip was produced |
| **2.1** Perception core | Real GPU-fine-tuned YOLO12s (Pascal VOC), `ModelDetector` wired into the pipeline, RF-DETR bake-off path | Trained for real on Colab T4 twice — `sentinel_yolo12_voc_v1.pt` (precision 0.862, recall 0.836, mAP50 0.900, mAP50-95 0.696) and `v2.pt` (0.858 / 0.835 / 0.896 / 0.689). **v1 wins on every metric** (narrowly — normal run-to-run variance, not a real gap) so it's the active detector, both kept on disk. RF-DETR's own training run OOM'd on the free T4 (`batch_size=8` too high — fixed to `batch_size=2, grad_accum_steps=8` in the notebook) and then proved too slow to finish in this phase — bake-off paused, not abandoned, YOLO12 is the pragmatic pick for now |
| **2.2** Evidence hardening | Merkle-tree daily anchor, chain-of-custody log, OpenTimestamps | Real submission to public Bitcoin-calendar servers; tamper-detection verified by editing a custody-log row directly via SQL and confirming the chain check catches it |
| **2.3** Edge resilience | Store-and-forward outbox, heartbeat/silent-site detection | Killed the backend mid-run, confirmed events queue instead of vanishing, confirmed retry drains the queue on reconnect; confirmed a site flips to `silent: true` after its heartbeat threshold elapses |
| **2.4** Reasoning (free tier) | Rule-based context engine (schedule suppression), template event descriptions | Verified correct suppression on matching day/time/label, correct non-suppression otherwise; descriptions verified for normal, suppressed, and low-severity cases |
| **2.5** Platform | Real-time SSE push, notification engine, multi-tenant `org_id` | Posted an event, watched it arrive on a live SSE connection instantly; confirmed console alert fired with correct severity; confirmed org-filtered query isolation |
| **2.6** Learning | Federated learning simulation (FedAvg) | Empirically verified across 5 seeds that a federated model generalizes better to a brand-new site than an existing site's solo model (0.920 vs 0.898 mean accuracy) — and the *first* experiment design was wrong and was caught before being reported, not just accepted |
| **L1 perception fleet gap-fill** | `perception/`: fall detection (pose heuristic), re-ID (color-histogram gallery), anomaly (per-label statistical z-score), audio (RMS loud-sound), fire/smoke (HSV heuristic) | All five have unit tests (`tests/test_perception.py`, 9 tests) covering the real logic paths — pose analysis, re-ID gallery matching, anomaly warm-up/flagging, audio thresholding, fire/smoke color detection |
| **Storage backend parity** | `DynamoDBStore` now has the same test rigor as `SQLiteStore` | 4 new tests (`tests/test_dynamodb_store.py`) mock `boto3.resource` to verify `save`/`list_recent`/`assign` call the right operations with the right shapes — previously untested despite being a full implementation |
| **Reasoning (2.4) — qwen-local closed** | `reasoning/describe.py QwenLocalDescriber` now does real Qwen2.5-VL inference via a local Ollama server, `Describer` interface extended to carry the event frame through `edge/pipeline.py` | Ran for real against `ollama serve` + `qwen2.5vl:3b` on a real image: produced a genuine multi-detail scene description ("two people standing on a sidewalk next to a blue electric bus... one person is holding a phone") — not a template fill-in |
| **VLM latency optimization** | Downscale frame to 512px before encoding (biggest lever — fewer visual tokens into the vision encoder), cap generation at 80 tokens (`num_predict`), shrink context window (`num_ctx=2048`), `keep_alive: 30m` to avoid Ollama's default 5-minute idle-unload forcing a reload on the next call | Measured before/after on the same real test image: **8.5s → 1.1s warm-state (~7.7x)**, no visible quality loss. Also benchmarked `gemma4:12b` (11.9B params, also vision-capable) as an alternative — 110.9s, ~13x slower with a *less* detailed description than qwen2.5vl:3b, confirming the smaller model is the right choice here, not a compromise |
| **Async description enrichment** | `edge/description_worker.py`: the frame loop always alerts immediately using the free `TemplateDescriber` (sub-second, real-time); a configured slow VLM backend (qwen-local/frontier) runs on a background thread and PATCHes a richer description in once ready, via the new `PATCH /events/{id}/description` endpoint | Caught before shipping: qwen-local's real ~11s latency would have stalled `edge/pipeline.py`'s frame loop for hundreds of frames per event if called inline. Verified end-to-end against a real running backend: event created with the fast description, PATCH landed, GET reflected the enriched text |
| **Single test entry point** | Removed the standalone image-upload test UI (`cloud/backend/vision.py`, `static/index.html`, `/detect` and `/` routes) — a duplicate, lower-fidelity test surface next to the real thing | The live webcam pipeline (`python -m edge.main` against the local backend) is now the one test path, matching the real deployment shape instead of a separate static-image demo |
| **Unified live dashboard** | New `/` route serving `cloud/backend/static/dashboard.html`: live event feed (label, severity, description) over SSE, all on one page | Verified end-to-end in a real browser preview: posted a real event over `POST /events` — appeared live with no refresh; PATCHed a description via `/events/{id}/description` — the same card updated in place (severity LOW, enriched text). No console errors |
| **Dashboard simplified + clear-recordings action** | Removed the redundant browser-webcam-preview panel and the raw JSON log dump — kept only controls, live tracking, and events. New `POST /recordings/clear` deletes clip files (and manifests) from `settings.clips_dir` and clears the pipeline log buffer; refuses while the pipeline is running (409) so it can't delete a clip mid-write. Deliberately does not touch the events DB or the evidence chain-of-custody log | Verified for real: created real dummy files in the clips dir, called the endpoint, confirmed `files_deleted` count and an empty directory afterward. Also found and fixed a real bug while testing: restarting the backend orphaned a running `edge.main` subprocess (a `subprocess.Popen` child doesn't die with its parent) — confirmed via `ps aux`, fixed with a FastAPI `lifespan` shutdown handler that terminates the pipeline child, tested that the deprecation-free `lifespan` pattern works via the full suite |
| **Live annotated tracking view** | `edge/live_frame_streamer.py` (background thread, decoupled from the frame loop like `description_worker.py`) pushes detection-boxes-drawn frames to `PUT /live-frame`; dashboard's tracking panel consumes `GET /live-frame/stream` (MJPEG) | Found and fixed a real bug during verification: the MJPEG generator was missing a `Content-Length` header per part, so the browser couldn't tell where a frame's bytes ended — confirmed via `img.naturalWidth === 0` despite a 200 response. After the fix: pushed 15 synthetic frames with a moving box, confirmed `naturalWidth`/`naturalHeight` populate and the box visibly renders and moves in a live browser screenshot |
| **Real ByteTrack (D3 gap closed)** | `edge/bytetrack_tracker.py`: `SENTINEL_TRACKER_BACKEND=bytetrack` swaps in the actual ByteTrack algorithm via `supervision` (Roboflow, MIT, pip-installable) behind the exact same `update() -> list[Track]` interface as `CentroidTracker` — a real drop-in, not a parallel path | The original blocker (ByteTrack only ever shipped as an un-packaged research repo) is resolved: `supervision` ships a maintained reimplementation. 7 tests pass, including multi-label tracking and stale-track eviction. Honest finding, not overclaimed: a synthetic occlusion/crossing-paths stress test showed both trackers behaving in complex, non-obvious ways — proving which one handles real crossing/occlusion better needs real footage, not hand-crafted coordinates. `CentroidTracker` remains the default; `bytetrack` is opt-in via config |
| **Real OSNet (D3 gap closed)** | `perception/reid.py OSNetReID`: real ImageNet-pretrained OSNet embeddings via `torchreid`, behind the exact same `ReIDEmbedder` interface `HistogramReID`/`Gallery` already use | The actual blocker wasn't packaging — `pip install torchreid` succeeds — it's an undeclared dependency: `import torchreid` crashes with `ModuleNotFoundError: No module named 'gdown'` because torchreid's own dataset code imports it without listing it. `pip install torchreid gdown` closes the gap for real. Verified: downloaded real pretrained weights, extracted real 512-d embeddings, confirmed identical crops score ~1.0 cosine similarity and different-colored crops score meaningfully lower (0.44 on a real bus.jpg photo). 5 new tests, all against the real model, nothing mocked |

## Honest stubs — the interface is real, the backing infrastructure isn't

These are not gaps in the code; they're places where the correct next step
requires a decision or an account only you can make/create. Each raises a
clear error explaining exactly what's missing, rather than silently doing
nothing or faking a result.

| Component | What exists | What it's waiting on |
|---|---|---|
| `reasoning/describe.py FrontierDescriber` | Full interface | A paid frontier API key (Claude/GPT-4o-class) — a deliberate, consent-gated spend decision |
| `cloud/backend/notifications.py SMSChannel` | Full interface | A paid SMS provider account (Twilio et al.) |
| ONNX→TensorRT quantization | Export cell exists in the Colab notebook | Real Jetson/edge hardware to quantize against — can't be tested on a dev laptop |
| `package` detection class | Full dataset plan, class-suppression logic ready | Either a decent Roboflow dataset (checked — the options are weak) or your own captured porch footage |
| `perception/pose.py YoloPoseEstimator` | Real keypoint inference (Ultralytics pose models), interface + fall heuristic fully tested | Not RTMPose (DECISIONS.md D3 target) — that needs the mmpose stack; upgrade is a backend swap behind `PoseEstimator`, no interface change |
| `perception/reid.py HistogramReID` | Real, free, tested cosine-similarity re-ID | Superseded by `OSNetReID` (real OSNet, see above) for anyone who wants the deep-embedding upgrade; kept as the zero-download default |
| `perception/audio.py RMSLoudSoundDetector` | Real, free, tested loud-sound flag | Not a sound *classifier* (glass break vs. gunshot vs. shout) — needs labelled audio data or a paid API to train/call against, same gap shape as `package` images |
| `perception/fire_smoke.py HSVFireSmokeDetector` | Real, free, tested color-heuristic detector | Not a trained fire/smoke model — will false-positive on fire-colored objects; upgrade is a fine-tuned detector on a public fire dataset behind the same interface |
| C2PA (evidence signing) | `evidence/signing.py` does real hash+sign+manifest, by its own docstring "not a certified C2PA implementation" | Full C2PA toolchain needs X.509 device certs and the standard manifest schema — a certification/tooling decision, not a code gap |

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
