"""Fine-tune YOLO12 (DECISIONS.md D1). Confirmed-working call shape — see
training/README.md "What's been tested" for the verified smoke run.

Defaults to the pretrained checkpoint (yolo12s.pt), not the bare architecture
(.yaml) — the smoke test used .yaml only to skip a weight download while
verifying the training loop. Real fine-tuning should transfer from COCO
weights, same as RF-DETR does by default.

Usage:
    pip install ultralytics
    python -m training.train_yolo12 --data ./data/datasets/remote-home/yolo/data.yaml --output-dir ./data/models/yolo12
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train(
    data_yaml: Path,
    model_arch: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    output_dir: Path,
    freeze: int | None,
) -> None:
    from ultralytics import YOLO

    model = YOLO(model_arch)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_dir),
        freeze=freeze,  # None = no freezing; see training/README.md layer-freezing recipe
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO-format data.yaml")
    parser.add_argument("--model", default="yolo12s.pt", help="Pretrained checkpoint (.pt) or bare architecture (.yaml)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    parser.add_argument("--output-dir", type=Path, default=Path("./data/models/yolo12"))
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze the first N layers (e.g. 10) for the initial epochs — see training/README.md",
    )
    args = parser.parse_args()

    train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.device, args.output_dir, args.freeze)


if __name__ == "__main__":
    main()
