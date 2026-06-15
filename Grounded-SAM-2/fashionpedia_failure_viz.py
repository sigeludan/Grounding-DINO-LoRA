#!/usr/bin/env python3
"""
Failure / success case visualization: GT vs Zero-shot vs LoRA (side-by-side).

Selects instances from paired per_instance.json by category and Δ mask IoU,
re-runs GDINO+SAM2 for both checkpoints, saves comparison panels.

Usage:
  python fashionpedia_failure_viz.py \
    --categories neckline pocket jacket \
    --top-k 5 \
    --output-dir outputs/failure_viz_th015
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision.ops import box_convert
from tqdm import tqdm

from prompt_templates import primary_category_name
from fashionpedia_eval import (
    DEFAULT_ANN,
    DEFAULT_GDINO_CFG,
    DEFAULT_GDINO_CKPT,
    DEFAULT_IMG_DIR,
    DEFAULT_SAM2_CFG,
    DEFAULT_SAM2_CKPT,
    build_prompt,
    draw_comparison,
    mask_iou,
    pick_best_box,
    xywh_to_xyxy,
)
from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LORA_CKPT = (
    "/root/autodl-tmp/Grounding-Dino-FineTuning/weights/fashionpedia_lora/"
    "20260614_2321/merged_swint_ogc.pth"
)
DEFAULT_ZS_JSON = PROJECT_ROOT / "outputs/fashionpedia_zero_shot_synonyms_th015_best/per_instance.json"
DEFAULT_LORA_JSON = PROJECT_ROOT / "outputs/fashionpedia_lora_v2_synonyms_th015_best/per_instance.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fashionpedia failure/success case visualization")
    p.add_argument("--ann-file", default=DEFAULT_ANN)
    p.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    p.add_argument("--zs-json", type=Path, default=DEFAULT_ZS_JSON)
    p.add_argument("--lora-json", type=Path, default=DEFAULT_LORA_JSON)
    p.add_argument("--zs-checkpoint", default=DEFAULT_GDINO_CKPT)
    p.add_argument("--lora-checkpoint", default=DEFAULT_LORA_CKPT)
    p.add_argument("--categories", nargs="+", default=["neckline", "pocket", "jacket"])
    p.add_argument("--top-k", type=int, default=5, help="Cases per category")
    p.add_argument(
        "--mode",
        choices=["win", "loss", "both"],
        default=None,
        help="win=LoRA best vs ZS, loss=LoRA worst vs ZS; default per-category",
    )
    p.add_argument("--box-threshold", type=float, default=0.15)
    p.add_argument("--text-threshold", type=float, default=0.15)
    p.add_argument("--prompt-strategy", default="synonyms")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/failure_viz_th015")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_pairs(zs_path: Path, lora_path: Path) -> Dict[int, dict]:
    zs = {r["ann_id"]: r for r in json.load(open(zs_path))}
    lora = {r["ann_id"]: r for r in json.load(open(lora_path))}
    out = {}
    for ann_id in zs:
        if ann_id not in lora:
            continue
        zr, lr = zs[ann_id], lora[ann_id]
        if zr["category"] != lr["category"]:
            continue
        out[ann_id] = {
            "ann_id": ann_id,
            "category": zr["category"],
            "file_name": zr["file_name"],
            "image_id": zr["image_id"],
            "zs_mask_iou": zr["mask_iou"],
            "lora_mask_iou": lr["mask_iou"],
            "delta": lr["mask_iou"] - zr["mask_iou"],
        }
    return out


def select_ann_ids(
    pairs: Dict[int, dict],
    category: str,
    top_k: int,
    mode: str,
) -> List[dict]:
    items = [v for v in pairs.values() if v["category"] == category]
    if mode == "win":
        items.sort(key=lambda x: x["delta"], reverse=True)
    elif mode == "loss":
        items.sort(key=lambda x: x["delta"])
    else:
        raise ValueError(mode)
    return items[:top_k]


def default_mode_for_category(category: str) -> str:
    # neckline: show LoRA wins; pocket/jacket: show LoRA losses
    return "win" if category == "neckline" else "loss"


def run_one_model(
    gdino_model,
    sam2_predictor,
    image_rgb: np.ndarray,
    image_tensor: torch.Tensor,
    prompt: str,
    gt_xyxy: np.ndarray,
    gt_mask: np.ndarray,
    box_th: float,
    text_th: float,
    device: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
    h, w = image_rgb.shape[:2]
    boxes, confidences, _ = predict(
        model=gdino_model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_th,
        text_threshold=text_th,
        device=device,
    )
    if boxes.shape[0] > 0:
        boxes = boxes * torch.tensor([w, h, w, h])
        pred_xyxy_all = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    else:
        pred_xyxy_all = np.zeros((0, 4), dtype=np.float32)

    best_idx, box_iou = pick_best_box(pred_xyxy_all, gt_xyxy)
    pred_xyxy = None
    pred_mask = None
    miou = 0.0

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        sam2_predictor.set_image(image_rgb)
        if best_idx is not None:
            pred_xyxy = pred_xyxy_all[best_idx]
            masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=pred_xyxy,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            pred_mask = masks[0].astype(bool)
            miou = mask_iou(pred_mask, gt_mask)

    return pred_xyxy, pred_mask, miou, box_iou


def make_gt_panel(image_bgr: np.ndarray, gt_xyxy: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    vis = image_bgr.copy()
    gx1, gy1, gx2, gy2 = gt_xyxy.astype(int)
    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 200, 0), 2)
    overlay = vis.copy()
    overlay[gt_mask.astype(bool)] = (
        overlay[gt_mask.astype(bool)] * 0.45 + np.array([0, 200, 0]) * 0.55
    ).astype(np.uint8)
    vis = cv2.addWeighted(overlay, 0.75, vis, 0.25, 0)
    cv2.putText(vis, "GT", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


def make_pred_panel(
    image_bgr: np.ndarray,
    gt_xyxy: np.ndarray,
    gt_mask: np.ndarray,
    pred_xyxy: Optional[np.ndarray],
    pred_mask: Optional[np.ndarray],
    title: str,
    miou: float,
    box_iou: float,
) -> np.ndarray:
    return draw_comparison(
        image_bgr,
        gt_xyxy,
        pred_xyxy,
        gt_mask,
        pred_mask,
        f"{title} | box={box_iou:.2f} mask={miou:.2f}",
    )


def stitch_panels(panels: List[np.ndarray]) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        if p.shape[0] != h:
            scale = h / p.shape[0]
            p = cv2.resize(p, (int(p.shape[1] * scale), h))
        resized.append(p)
    return np.hstack(resized)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.zs_json, args.lora_json)
    coco = COCO(args.ann_file)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}
    attr_id_to_name = {a["id"]: a["name"] for a in coco.dataset.get("attributes", [])}

    selections: List[dict] = []
    for cat in args.categories:
        mode = args.mode or default_mode_for_category(cat)
        for item in select_ann_ids(pairs, cat, args.top_k, mode):
            item["select_mode"] = mode
            selections.append(item)

    manifest = []
    print(f"Selected {len(selections)} cases -> {out_dir}")

    print("Loading SAM2...")
    sam2_model = build_sam2(DEFAULT_SAM2_CFG, DEFAULT_SAM2_CKPT, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("Loading Zero-shot GDINO...")
    zs_model = load_model(DEFAULT_GDINO_CFG, args.zs_checkpoint, device=args.device)

    print("Loading LoRA GDINO...")
    lora_model = load_model(DEFAULT_GDINO_CFG, args.lora_checkpoint, device=args.device)

    ann_id_to_ann = {}
    img_id_to_info = {}
    for ann_id in {s["ann_id"] for s in selections}:
        ann = coco.loadAnns(ann_id)[0]
        ann_id_to_ann[ann_id] = ann
        img_id_to_info[ann["image_id"]] = coco.loadImgs(ann["image_id"])[0]

    for item in tqdm(selections, desc="Rendering"):
        ann = ann_id_to_ann[item["ann_id"]]
        img_info = img_id_to_info[ann["image_id"]]
        img_path = Path(args.img_dir) / img_info["file_name"]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue

        category_name = item["category"]
        prompt = build_prompt(
            category_name,
            ann.get("attribute_ids", []),
            attr_id_to_name,
            "category",
            args.prompt_strategy,
        )
        gt_xyxy = xywh_to_xyxy(ann["bbox"])
        gt_mask = coco.annToMask(ann).astype(bool)
        image_rgb, image_tensor = load_image(str(img_path))

        zs_xyxy, zs_mask, zs_miou, zs_box = run_one_model(
            zs_model, sam2_predictor, image_rgb, image_tensor, prompt,
            gt_xyxy, gt_mask, args.box_threshold, args.text_threshold, args.device,
        )
        lora_xyxy, lora_mask, lora_miou, lora_box = run_one_model(
            lora_model, sam2_predictor, image_rgb, image_tensor, prompt,
            gt_xyxy, gt_mask, args.box_threshold, args.text_threshold, args.device,
        )

        cat_slug = primary_category_name(category_name).replace(" ", "_")
        delta = lora_miou - zs_miou
        fname = (
            f"{item['select_mode']}_{cat_slug}_ann{item['ann_id']}_"
            f"d{delta:+.2f}_zs{zs_miou:.2f}_lora{lora_miou:.2f}.jpg"
        )

        panels = [
            make_gt_panel(image_bgr, gt_xyxy, gt_mask),
            make_pred_panel(image_bgr, gt_xyxy, gt_mask, zs_xyxy, zs_mask, "Zero-shot", zs_miou, zs_box),
            make_pred_panel(image_bgr, gt_xyxy, gt_mask, lora_xyxy, lora_mask, "LoRA v2", lora_miou, lora_box),
        ]
        grid = stitch_panels(panels)
        cv2.imwrite(str(out_dir / fname), grid)

        entry = {
            **item,
            "prompt": prompt,
            "file_name": img_info["file_name"],
            "viz_path": str(out_dir / fname),
            "zs_miou_rerun": zs_miou,
            "lora_miou_rerun": lora_miou,
            "delta_rerun": delta,
        }
        manifest.append(entry)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Markdown gallery index
    lines = [
        "# Failure / Success Case Gallery",
        "",
        f"Threshold: box={args.box_threshold}, text={args.text_threshold} | prompt={args.prompt_strategy}",
        "",
        "| Category | Mode | ann_id | Δ mIoU | ZS | LoRA | Preview |",
        "|----------|------|-------:|-------:|---:|-----:|---------|",
    ]
    for e in manifest:
        rel = Path(e["viz_path"]).name
        lines.append(
            f"| {e['category']} | {e['select_mode']} | {e['ann_id']} | "
            f"{e['delta_rerun']:+.3f} | {e['zs_miou_rerun']:.3f} | {e['lora_miou_rerun']:.3f} | `{rel}` |"
        )
    (out_dir / "GALLERY.md").write_text("\n".join(lines) + "\n")

    print(f"Saved {len(manifest)} panels -> {out_dir}")
    print(f"Index -> {out_dir / 'GALLERY.md'}")


if __name__ == "__main__":
    main()
