from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml


def _read_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        class_id = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        x_min = (cx - w / 2) * img_w
        y_min = (cy - h / 2) * img_h
        boxes.append({"category_id": class_id, "bbox": [x_min, y_min, w * img_w, h * img_h]})
    return boxes


def convert_split(images_dirs: Path | list[Path], labels_dirs: Path | list[Path], out_dir: Path, class_names: list[str]) -> int:
    if isinstance(images_dirs, Path):
        images_dirs = [images_dirs]
    if isinstance(labels_dirs, Path):
        labels_dirs = [labels_dirs]

    out_dir.mkdir(parents=True, exist_ok=True)
    images_meta = []
    annotations = []
    img_id = 0
    ann_id = 1

    for images_dir, labels_dir in zip(images_dirs, labels_dirs):
        image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            dest_name = f"{images_dir.name}_{img_path.name}"
            images_meta.append({"id": img_id, "file_name": dest_name, "width": w, "height": h})

            label_path = labels_dir / (img_path.stem + ".txt")
            for box in _read_yolo_labels(label_path, w, h):
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": box["category_id"] + 1,
                        "bbox": box["bbox"],
                        "area": box["bbox"][2] * box["bbox"][3],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

            dest = out_dir / dest_name
            if not dest.exists():
                dest.write_bytes(img_path.read_bytes())
            img_id += 1

    categories = [{"id": i + 1, "name": name, "supercategory": "none"} for i, name in enumerate(class_names)]
    coco = {"images": images_meta, "annotations": annotations, "categories": categories}
    (out_dir / "_annotations.coco.json").write_text(json.dumps(coco))
    return len(images_meta)


def convert_dataset(yolo_data_yaml: Path, out_dir: Path, dataset_root: Path | None = None) -> dict[str, int]:
    yolo_data_yaml = Path(yolo_data_yaml)
    out_dir = Path(out_dir)
    config = yaml.safe_load(yolo_data_yaml.read_text())
    if dataset_root is not None:
        base = dataset_root / config.get("path", ".")
    else:
        base = Path(config.get("path", yolo_data_yaml.parent))
        if not base.is_absolute():
            base = yolo_data_yaml.parent / base
    class_names = config["names"] if isinstance(config["names"], list) else list(config["names"].values())

    split_map = {"train": "train", "val": "valid", "test": "test"}
    counts = {}
    for yolo_split_key, coco_split_name in split_map.items():
        rel = config.get(yolo_split_key)
        if not rel:
            continue
        rel_list = rel if isinstance(rel, list) else [rel]

        images_dirs, labels_dirs = [], []
        for r in rel_list:
            images_dir = base / r
            if not images_dir.exists():
                continue
            images_dirs.append(images_dir)
            labels_dirs.append(Path(str(images_dir).replace("/images/", "/labels/", 1)))
        if not images_dirs:
            continue

        counts[coco_split_name] = convert_split(images_dirs, labels_dirs, out_dir / coco_split_name, class_names)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="YOLO-format data.yaml")
    parser.add_argument("--out", type=Path, required=True, help="Output COCO-format directory")
    args = parser.parse_args()

    counts = convert_dataset(args.data, args.out)
    for split, n in counts.items():
        print(f"{split}: {n} images -> {args.out / split}")


if __name__ == "__main__":
    main()
