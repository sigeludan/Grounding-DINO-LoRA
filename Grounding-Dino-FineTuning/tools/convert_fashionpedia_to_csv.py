#!/usr/bin/env python3
"""
Convert Fashionpedia COCO-style JSON to Grounding-Dino-FineTuning CSV format.

CSV columns (required by groundingdino/datasets/dataset.py):
  image_name, label_name, bbox_x, bbox_y, bbox_width, bbox_height

Usage:
  python tools/convert_fashionpedia_to_csv.py \\
    --ann /root/autodl-tmp/fashionpedia/annotations/instances_attributes_train2020.json \\
    --img-dir /root/autodl-tmp/fashionpedia/images/train \\
    --out /root/autodl-tmp/fashionpedia/gdino_csv/train_annotations.csv \\
    --max-images 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Set


def primary_label(category_name: str) -> str:
    """Match inference prompt strategy: use first synonym, lowercase."""
    return category_name.split(",")[0].strip().lower()


def convert(
    ann_path: str,
    img_dir: str,
    out_csv: str,
    max_images: int | None = None,
    seed: int = 42,
) -> dict:
    with open(ann_path, "r") as f:
        data = json.load(f)

    cat_id_to_name: Dict[int, str] = {c["id"]: c["name"] for c in data["categories"]}
    img_id_to_info = {im["id"]: im for im in data["images"]}

    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[ann["image_id"]].append(ann)

    img_ids = [i for i in anns_by_image if anns_by_image[i]]
    rng = random.Random(seed)
    rng.shuffle(img_ids)
    if max_images is not None:
        img_ids = img_ids[:max_images]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    rows: List[dict] = []
    missing_images: Set[str] = set()
    skipped_boxes = 0

    for img_id in img_ids:
        img_info = img_id_to_info[img_id]
        img_name = img_info["file_name"]
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            missing_images.add(img_name)
            continue

        for ann in anns_by_image[img_id]:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                skipped_boxes += 1
                continue
            label = primary_label(cat_id_to_name[ann["category_id"]])
            rows.append(
                {
                    "image_name": img_name,
                    "label_name": label,
                    "bbox_x": int(round(x)),
                    "bbox_y": int(round(y)),
                    "bbox_width": int(round(w)),
                    "bbox_height": int(round(h)),
                }
            )

    fieldnames = [
        "image_name",
        "label_name",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats = {
        "ann_path": ann_path,
        "img_dir": img_dir,
        "out_csv": out_csv,
        "num_images": len(img_ids),
        "num_instances": len(rows),
        "missing_images": len(missing_images),
        "skipped_boxes": skipped_boxes,
        "unique_labels": sorted({r["label_name"] for r in rows}),
    }
    meta_path = out_csv.replace(".csv", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True)
    p.add_argument("--img-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stats = convert(
        args.ann,
        args.img_dir,
        args.out,
        max_images=args.max_images,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
