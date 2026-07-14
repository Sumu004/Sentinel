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

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "images" / "train2007" / "a.jpg"), img)
    (root / "labels" / "train2007" / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n")

    cv2.imwrite(str(root / "images" / "val2007" / "b.jpg"), img)

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
    assert ann["bbox"] == [40.0, 30.0, 20.0, 40.0]
    assert ann["category_id"] == 1


def test_convert_dataset_handles_background_images(tmp_path: Path):
    data_yaml = _make_yolo_dataset(tmp_path / "yolo_ds")
    out_dir = tmp_path / "coco_ds"

    convert_dataset(data_yaml, out_dir)

    valid_coco = json.loads((out_dir / "valid" / "_annotations.coco.json").read_text())
    assert len(valid_coco["images"]) == 1
    assert len(valid_coco["annotations"]) == 0


def _make_voc_shaped_dataset(root: Path) -> Path:
    for split in ["train2007", "train2012", "val2007"]:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "images" / "train2007" / "000005.jpg"), img)
    (root / "labels" / "train2007" / "000005.txt").write_text("0 0.5 0.5 0.2 0.4\n")
    cv2.imwrite(str(root / "images" / "train2012" / "000005.jpg"), img)
    (root / "labels" / "train2012" / "000005.txt").write_text("0 0.25 0.25 0.1 0.1\n")

    cv2.imwrite(str(root / "images" / "val2007" / "b.jpg"), img)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.dump(
            {
                "path": str(root),
                "train": ["images/train2007", "images/train2012"],
                "val": ["images/val2007"],
                "nc": 1,
                "names": ["person"],
            }
        )
    )
    return data_yaml


def test_convert_dataset_accepts_string_out_dir(tmp_path: Path):
    data_yaml = _make_voc_shaped_dataset(tmp_path / "voc_ds")
    out_dir_str = str(tmp_path / "coco_ds_str")

    counts = convert_dataset(data_yaml, out_dir_str)

    assert counts["train"] == 2
    assert (Path(out_dir_str) / "train" / "_annotations.coco.json").exists()


def test_convert_dataset_merges_list_of_dirs_like_real_voc_yaml(tmp_path: Path):
    data_yaml = _make_voc_shaped_dataset(tmp_path / "voc_ds")
    out_dir = tmp_path / "coco_ds"

    counts = convert_dataset(data_yaml, out_dir)

    assert counts["train"] == 2
    assert counts["valid"] == 1

    train_coco = json.loads((out_dir / "train" / "_annotations.coco.json").read_text())
    assert len(train_coco["images"]) == 2
    assert len(train_coco["annotations"]) == 2
    file_names = {img["file_name"] for img in train_coco["images"]}
    assert file_names == {"train2007_000005.jpg", "train2012_000005.jpg"}
    assert (out_dir / "train" / "train2007_000005.jpg").exists()
    assert (out_dir / "train" / "train2012_000005.jpg").exists()


def test_convert_dataset_resolves_relative_path_against_dataset_root_not_yaml_location(tmp_path: Path):
    dataset_root = tmp_path / "actual_datasets_dir"
    (dataset_root / "VOC" / "images" / "train2007").mkdir(parents=True)
    (dataset_root / "VOC" / "labels" / "train2007").mkdir(parents=True)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(dataset_root / "VOC" / "images" / "train2007" / "a.jpg"), img)
    (dataset_root / "VOC" / "labels" / "train2007" / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n")

    yaml_elsewhere_dir = tmp_path / "somewhere" / "else" / "entirely"
    yaml_elsewhere_dir.mkdir(parents=True)
    data_yaml = yaml_elsewhere_dir / "VOC.yaml"
    data_yaml.write_text(
        yaml.dump({"path": "VOC", "train": "images/train2007", "nc": 1, "names": ["person"]})
    )

    out_dir = tmp_path / "coco_ds"
    counts = convert_dataset(data_yaml, out_dir, dataset_root=dataset_root)

    assert counts == {"train": 1}
