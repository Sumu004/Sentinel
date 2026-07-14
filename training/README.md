# Training — dataset, recipe, bake-off

## Pilot classes (remote home)

| Class | Why |
|---|---|
| `person` | Core class — intruder vs. familiar |
| `vehicle` | Unfamiliar car in the driveway |
| `package` | Porch-theft detection |
| `animal` | Suppression class — pets shouldn't trigger alarms |

## Setup

```bash
pip install -r training/requirements-training.txt
```

## Train

```bash
python -m training.train_yolo12 --data <data.yaml> --output-dir ./data/models/yolo12
python -m training.train_rfdetr --dataset-dir <coco-dir> --output-dir ./data/models/rfdetr
```

Or open [colab_finetune.ipynb](colab_finetune.ipynb) in Google Colab (free T4)
to train both and run the bake-off in one go.

Drop the resulting weights in and run it:

```bash
SENTINEL_DETECTOR_BACKEND=model SENTINEL_DETECTOR_MODEL_PATH=./data/models/best.pt python -m edge.main
```

## Bake-off

Train both models on the same data, compare mAP@50:95 and false-alarms/day,
ship whichever wins. `training/eval.py` computes both.

## Files

- `prepare_data.py` — downloads a Roboflow dataset in COCO + YOLO format
- `yolo_to_coco.py` — converts YOLO-format data to COCO (for RF-DETR)
- `train_rfdetr.py`, `train_yolo12.py` — training scripts
- `eval.py` — mAP and false-alarm scoring

## Status

YOLO12 is trained and live in the pipeline. RF-DETR trains on the same data
but the full bake-off run is still pending. `package` class needs better
data — see [PROJECT_STATUS.md](../PROJECT_STATUS.md).
