from __future__ import annotations

import argparse
import os
from pathlib import Path


def download(workspace: str, project: str, version: int, out_dir: Path) -> None:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. Sign up free at roboflow.com, grab an API "
            "key from your account settings, and `export ROBOFLOW_API_KEY=...`."
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ds = proj.version(version)

    coco_dir = out_dir / "coco"
    yolo_dir = out_dir / "yolo"
    coco_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {workspace}/{project} v{version} as COCO -> {coco_dir}")
    ds.download("coco", location=str(coco_dir))

    print(f"Downloading {workspace}/{project} v{version} as YOLOv9 -> {yolo_dir}")
    ds.download("yolov9", location=str(yolo_dir))

    print("Done. RF-DETR: point --dataset-dir at the coco/ output.")
    print("YOLO12: point --data at <yolo_dir>/data.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    download(args.workspace, args.project, args.version, args.out)


if __name__ == "__main__":
    main()
