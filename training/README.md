# Phase 2.1 — Perception core: dataset, recipe, bake-off

Scope for the fine-tuned detector targeted in [DECISIONS.md](../DECISIONS.md) D1.
Everything below has been verified to actually run (see "What's been tested").

## Pilot vertical: remote home

Per [VISION.md](../VISION.md) and the product owner's call, the pilot is a remote
home. That fixes the class set:

| Class | Why | Source |
|---|---|---|
| `person` | The core class — intruder vs. familiar | COCO subset, + your own footage |
| `vehicle` | Unfamiliar car in the driveway | COCO subset (`car`, `truck`) |
| `package` | Porch-theft detection — delivered, then taken | [Packages dataset (Roboflow public)](https://public.roboflow.com/object-detection/packages-dataset) |
| `animal` | **Suppression class, not a threat** — pets/wildlife are the #1 false-alarm source for home cameras; the model must learn to *not* alarm on them | COCO subset (`dog`, `cat`) |

### Night/low-light coverage (closes the "TBD" gap from the first pass)

A home camera spends most of its threat-relevant hours in the dark — a model
only trained on COCO's mostly-daylight images will degrade exactly when it
matters. Two real, sourced datasets to blend in:

- **[ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)** —
  7,363 images across 10 low-light conditions, 12 classes (PASCAL-VOC-like,
  includes `People`). License is BSD-3 **with a catch**: the maintainers ask
  commercial users to contact them directly — not a blocking license like
  SH17's CC-NC, but a real caveat worth the same "flag before commercializing"
  treatment as D1's licensing notes.
- **[NightOwls](https://www.nightowls-dataset.org/)** — pedestrians at night,
  purpose-built for low-light person detection in urban/outdoor scenes.

## Architecture — research findings (this session)

Re-checked the D1 bake-off against the current SOTA landscape rather than
assuming RF-DETR/YOLO12 were still the right two:

- **RF-DETR remains the strongest single pick** — first open model to break 60
  mAP on COCO (found via neural architecture search, not hand-design), and it
  beats both **D-FINE** and **LW-DETR** on RF100-VL, a *generalizability*
  benchmark — closer to what fine-tuning on a small home-camera dataset
  actually needs than raw COCO accuracy.
- **D-FINE was seriously considered as a third candidate** (Apache-2.0, up to
  59.3% AP after Objects365 pretraining, 124 FPS at 54% AP on a T4 — genuinely
  excellent numbers). Checked its actual install path before adding it to the
  scaffold: **it is not pip-installable** — fine-tuning means cloning the repo,
  editing YAML configs, and launching via `torchrun` (its own examples assume
  4 GPUs). That's real friction against the D8 "free, fast iteration on a
  single Colab GPU" constraint that RF-DETR/YOLO12 don't have. Decision: D-FINE
  moves to the **cloud re-analysis** role (see D1) where the one-time setup
  cost is worth paying for the accuracy ceiling; it's out of the edge bake-off.
- **No YOLOv13 exists** as of this research — the current Ultralytics line
  tops out at YOLO12. Worth recording so nobody chases it later.

Fire/smoke, weapons, and PPE (D1/D3 candidates) are real classes for the
warehouse/industrial vertical — deferred until that pilot starts, so Phase 2.1
doesn't train a 17-class model nobody asked for yet. Sources, once needed:
[D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset) (20k+ images,
YOLO format) for fire/smoke; Roboflow Universe weapon-detection sets (e.g.
[gun-and-knife-detection](https://universe.roboflow.com/mahad-ahmed/gun-and-knife-detection),
8,451 images) for weapons;
[SH17](https://universe.roboflow.com/ppehaak/sh17-hmkpl) (8,099 images, 17 PPE
classes) for warehouse PPE — **note SH17 is CC BY-NC-SA 4.0, non-commercial**.
That's stricter than the AGPL question in D1: it blocks commercial use of the
dataset/derived weights outright, not just requiring open-sourcing. Fine for
research now (per "best resource, license secondary, release deferred"); flag
for replacement with a permissively-licensed PPE set before any commercial PPE
product ships.

## Recipe — confirmed working, both candidates from D1

### RF-DETR

```bash
pip install "rfdetr[train,loggers]"
```

```python
from rfdetr import RFDETRBase

model = RFDETRBase()
model.train(
    dataset_dir="<path>",   # COCO format: train/, valid/, test/ subdirs,
                             # each with _annotations.coco.json
    epochs=50,
    batch_size=4,            # 8GB+ VRAM recommended; drop to 1-2 on smaller GPUs
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="<path>",
)
```

Confirmed: `RFDETRBase()` downloads a 355MB pretrained checkpoint on first use
(cached at `~/.roboflow/models/`, one-time, free). The bare `rfdetr` package
imports fine but **training requires the `[train,loggers]` extra** — without
it, `model.train()` fails with `ModuleNotFoundError: pytorch_lightning`, not an
obvious error from the base install. A 1-epoch smoke run on a 6-image synthetic
dataset completed in under 3 minutes (mostly the checkpoint download) and
produced real mAP metrics — the training loop itself is correct and fast on CPU
for tiny data; a real GPU is what's needed for real dataset sizes.

### YOLO12

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("yolo12n.yaml")        # or yolo12s.yaml / yolo12m.yaml for more capacity
results = model.train(
    data="data.yaml",                # YOLO format: train/val image+label dirs, nc, names
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,                        # GPU index, or "cpu"
)
```

Confirmed: `yolo12n.yaml` (the architecture, no pretrained weights) loads and
trains cleanly — verified with a 1-epoch run on synthetic data. Ultralytics
8.4.x ships YOLO12 natively, no extra install beyond the base package.
**For the real fine-tune, start from `yolo12s.pt` (pretrained), not the bare
`.yaml`** — the smoke test used the architecture-only file specifically to
avoid a weight download while testing the training loop; real fine-tuning
should transfer from COCO weights like RF-DETR does.

## Fine-tuning strategy — layer freezing and augmentation

Both candidates fine-tune from COCO-pretrained weights (RF-DETR does this by
default; YOLO12 needs `yolo12s.pt` explicitly, per above). Two concrete,
research-backed refinements over a default fine-tune:

- **Freeze the backbone first, unfreeze gradually.** Early backbone layers
  learn generic features (edges, textures) that transfer regardless of
  domain; the home-camera-specific signal lives in the neck/head. Recipe:
  start with the backbone frozen (`ultralytics` supports this via the
  `freeze` training arg, e.g. `freeze=10` for the first 10 layers), train a
  few epochs, then unfreeze and continue at a lower LR. This is the standard
  fix for catastrophic forgetting when fine-tuning on a dataset orders of
  magnitude smaller than COCO — exactly our situation (a few hundred home-camera
  images vs. COCO's 118k).
- **CCTV-realistic augmentation**, on top of the libraries' defaults: motion
  blur, low-light/contrast shifts (paired with the ExDark/NightOwls data
  above), and compression-artifact simulation — closer to what an actual
  IP camera feed looks like than clean photos.

## Inference-time: SAHI for small/distant objects

**Confirmed installed and working this session** (`pip install -U sahi`,
imports cleanly, supports both RF-DETR and Ultralytics models natively). A
fixed wide-FOV home camera will often see a person as a small, distant blob —
exactly the failure mode standard detection struggles with. SAHI tiles the
frame, runs detection per-tile, and stitches results — measured gains on
small-object benchmarks (VisDrone/xView) are **+5 to +15 AP** depending on the
detector. This is an inference-time technique, not a training change — apply
it on top of whichever model wins the bake-off, no retraining required.

## Bake-off methodology (D1)

Train both on the same remote-home dataset and the same split, then compare:

1. **mAP@50:95** on the held-out test set — standard accuracy.
2. **False-alarms-per-camera-per-day** — the metric that actually matters for
   an unattended site (see [VISION.md](../VISION.md) design principle 3). Run
   the trained model continuously against several hours of real, event-free
   footage from the pilot camera and count spurious detections per 24h. This
   needs real deployment data, so it's a Phase 2.1→2.2 bridge metric, not a
   one-shot benchmark number — `eval.py` below computes the held-out mAP
   directly and stubs the false-alarm counter for when footage exists.
3. **Latency on the target edge box** (D5) — export both to ONNX/TensorRT and
   measure on the actual hardware once chosen, not a generic GPU number.

Whichever wins 2/3 ships. Keep the other's training config — committed
alongside the winner — so re-running the bake-off after the dataset grows is a
rerun, not a redesign.

## Compute (D8 — free)

Neither script requires a GPU to *run* (both were smoke-tested on CPU), but
real training needs one for real epoch counts on real dataset sizes:

- **Free GPU:** Google Colab (free T4) or Kaggle (free T4, 30 hrs/week) — copy
  `training/` in, mount the dataset, run either script.
- **Local CPU:** fine for iterating on the pipeline/code, not for a model
  you'd actually deploy — RF-DETR's docs recommend 8GB+ VRAM, and YOLO12 on CPU
  is slow at real image counts (640px, thousands of images).

## What's been tested (this session, real runs — not claimed, verified)

- `pip install rfdetr` and `pip install "rfdetr[train,loggers]"` — clean installs.
- `pip install ultralytics` — clean install, YOLO12 confirmed present (v8.4.83).
- RF-DETR: downloaded the real 355MB pretrained checkpoint, trained 1 epoch on
  a 6-image synthetic COCO dataset, produced real mAP/precision/recall numbers.
- YOLO12: trained 1 epoch on a 6-image synthetic YOLO dataset on CPU, completed
  in under a second, produced real loss/mAP numbers.
- `pip install -U sahi` — clean install, `AutoDetectionModel`/`get_sliced_prediction`
  import cleanly (v0.12.1). Not yet run against a trained model — needs one to exist first.
- Checked D-FINE's actual install path (not just its paper) before excluding it
  from the edge bake-off — confirmed no pip package, `torchrun`-based training.
- **Not tested:** training on a real dataset (Roboflow downloads), GPU training,
  ONNX/TensorRT export, the false-alarm metric (needs real footage), SAHI against
  a real trained model. These are the actual Phase 2.1 work — this session
  proved the tooling and the architecture choice are sound, not that a
  deployable model exists yet.

## Files in this directory

- `prepare_data.py` — downloads a Roboflow Universe dataset by project/version
  in both COCO and YOLO formats (the two formats the two candidates need).
- `train_rfdetr.py`, `train_yolo12.py` — thin, config-driven wrappers around
  the confirmed-working calls above.
- `eval.py` — held-out mAP (delegates to each library's own validator) plus the
  false-alarm-rate stub described above.
