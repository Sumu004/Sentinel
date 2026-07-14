# Sentinel — Project status

Test suite: **95/95 passing** (`python -m pytest tests/ -v`).

## Done and tested

- Full pipeline: webcam/RTSP → detect → track → debounce → record → sign → send to backend
- Real trained detector (YOLO12s on Pascal VOC) live in the pipeline
- Evidence chain: signing, Merkle anchor, OpenTimestamps, chain-of-custody
- Edge resilience: store-and-forward outbox, heartbeat/silent-site detection
- Reasoning: rule-based context suppression, template + local Qwen2.5-VL descriptions
- Platform: live dashboard, SSE event stream, notifications, multi-tenant orgs
- Federated learning proven in simulation
- L1 perception fleet: fall detection, re-ID (histogram + real OSNet), anomaly, audio, fire/smoke
- Real ByteTrack tracker (via `supervision`), opt-in alongside the default tracker

## Honest stubs — waiting on a decision, not a code gap

| Component | Needs |
|---|---|
| Frontier VLM descriptions | A paid API key |
| SMS notifications | A paid SMS provider account |
| ONNX → TensorRT quantization | Real edge hardware to quantize against |
| `package` detection class | Better training data or your own captured footage |
| C2PA-certified signing | Full X.509/manifest tooling (current signing is real, just not certified) |

## Blocked on a real fleet

Continual per-site learning, active learning, OTA updates, and fleet-scale
observability all need real deployed sites generating real traffic — can't
be simulated further.

## What's next

1. Run the RF-DETR vs YOLO12 bake-off for real (Colab)
2. Decide the VLM tier (local GPU server or a frontier API key)
3. Get `package` training data
4. Pick edge hardware
5. Deploy to a real site

See [ROADMAP.md](ROADMAP.md) for phase details.
