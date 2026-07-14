# Sentinel

An edge AI system that watches unattended places — homes, warehouses,
substations, construction yards — detects what's happening, tracks it,
records sealed evidence, and sends an alert.

## Demo

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# terminal 1 — backend
uvicorn cloud.backend.app:app --port 8000

# terminal 2 — pipeline (uses your webcam)
python -m edge.main
```

Open `http://localhost:8000` for the live dashboard: tracking feed, events,
severity, logs.

Run the tests (no camera needed):

```bash
python -m pytest tests/ -v
```

## What it does

Webcam or IP camera → detect → track → debounce → record a clip → sign and
hash it → send an event to the backend → show it live on the dashboard.

Everything runs free, locally, no paid API keys, no special hardware.

## Docs

- [VISION.md](VISION.md) — what this is and why
- [ROADMAP.md](ROADMAP.md) — build phases
- [DECISIONS.md](DECISIONS.md) — model/tech choices
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — what's done, what's not
- [TESTING.md](TESTING.md) — how it's scored
- [training/README.md](training/README.md) — training the detector

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
