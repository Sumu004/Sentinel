import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.yolo_to_coco import convert_dataset


def _make_yolo_dataset(root: Path) -> Path:
    for split in ["train2007", "val2007"]:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    # one labelled image in train, one unlabelled (background) image in val
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "images" / "train2007" / "a.jpg"), img)
    (root / "labels" / "train2007" / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n")

    cv2.imwrite(str(root / "images" / "val2007" / "b.jpg"), img)
    # no label file for b.jpg — background image, should convert with 0 annotations

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.dump({"path": str(root), "train": "images/train2007", "val": "images/val2007", "nc": 1, "names": ["person"]})
    )
    return data_yaml


def test_convert_dataset_produces_valid_coco(tmp_path: Path):
    data_yaml = _make_yolo_dataset(tmp_path / "yolo_ds")
    out_dir = tmp_path / "coco_ds"

    counts = convert_dataset(data_yaml, out_dir)

    assert counts["train"] == 1
    assert counts["valid"] == 1

    train_coco = json.loads((out_dir / "train" / "_annotations.coco.json").read_text())
    assert len(train_coco["images"]) == 1
    assert len(train_coco["annotations"]) == 1
    assert train_coco["categories"] == [{"id": 1, "name": "person", "supercategory": "none"}]

    ann = train_coco["annotations"][0]
    # cx=0.5, cy=0.5, w=0.2, h=0.4 on a 100x100 image -> x_min=40, y_min=30, w=20, h=40
    assert ann["bbox"] == [40.0, 30.0, 20.0, 40.0]
    assert ann["category_id"] == 1  # COCO category ids are 1-indexed


def test_convert_dataset_handles_background_images(tmp_path: Path):
    data_yaml = _make_yolo_dataset(tmp_path / "yolo_ds")
    out_dir = tmp_path / "coco_ds"

    convert_dataset(data_yaml, out_dir)

    valid_coco = json.loads((out_dir / "valid" / "_annotations.coco.json").read_text())
    assert len(valid_coco["images"]) == 1
    assert len(valid_coco["annotations"]) == 0  # background image, no boxes
