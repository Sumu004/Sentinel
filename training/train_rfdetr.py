"""Fine-tune RF-DETR (DECISIONS.md D1). Confirmed-working call shape — see
training/README.md "What's been tested" for the verified smoke run.

Usage:
    pip install "rfdetr[train,loggers]"
    python -m training.train_rfdetr --dataset-dir ./data/datasets/remote-home/coco --output-dir ./data/models/rfdetr
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train(dataset_dir: Path, output_dir: Path, epochs: int, batch_size: int, grad_accum_steps: int, lr: float) -> None:
    from rfdetr import RFDETRBase

    model = RFDETRBase()
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        output_dir=str(output_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="COCO-format dataset with train/valid/test subdirs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4, help="8GB+ VRAM recommended; use 1-2 on smaller GPUs")
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(args.dataset_dir, args.output_dir, args.epochs, args.batch_size, args.grad_accum_steps, args.lr)


if __name__ == "__main__":
    main()
