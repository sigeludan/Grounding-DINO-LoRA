#!/usr/bin/env python3
"""
Compare structured prompt strategies on the same Fashionpedia val subset.

Usage:
  export TRANSFORMERS_OFFLINE=1
  python fashionpedia_prompt_eval.py --max-images 50
  python fashionpedia_prompt_eval.py --strategies baseline descriptive multi_term
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision.ops import box_convert
from tqdm import tqdm

from fashionpedia_eval import (
    DEFAULT_ANN,
    DEFAULT_GDINO_CFG,
    DEFAULT_GDINO_CKPT,
    DEFAULT_IMG_DIR,
    DEFAULT_SAM2_CFG,
    DEFAULT_SAM2_CKPT,
    build_samples,
    draw_comparison,
    mask_dice,
    mask_iou,
    pick_best_box,
    xywh_to_xyxy,
)
from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
from prompt_templates import (
    PROMPT_STRATEGIES,
    all_strategies,
    build_structured_prompt,
    primary_category_name,
)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="Fashionpedia prompt strategy comparison")
    p.add_argument("--ann-file", default=DEFAULT_ANN)
    p.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/fashionpedia_prompt_compare"))
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategies", nargs="*", default=None, help="default: all")
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--save-vis", type=int, default=10, help="per-strategy vis count for best/worst")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def run_one_instance(
    gdino_model,
    sam2_predictor,
    image_rgb,
    image_tensor,
    prompt: str,
    gt_xyxy: np.ndarray,
    gt_mask: np.ndarray,
    device: str,
    box_threshold: float,
    text_threshold: float,
) -> dict:
    h, w = image_rgb.shape[:2]
    boxes, confidences, phrases = predict(
        model=gdino_model,
        image=image_tensor,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    if boxes.shape[0] > 0:
        boxes = boxes * torch.tensor([w, h, w, h])
        pred_xyxy_all = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        conf_np = confidences.numpy()
    else:
        pred_xyxy_all = np.zeros((0, 4), dtype=np.float32)
        conf_np = np.array([])

    best_idx, best_box_iou = pick_best_box(pred_xyxy_all, gt_xyxy)
    pred_mask = None
    pred_xyxy = None
    miou = mdice = 0.0
    pred_conf = 0.0
    pred_phrase = ""

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        sam2_predictor.set_image(image_rgb)
        if best_idx is not None:
            pred_xyxy = pred_xyxy_all[best_idx]
            pred_conf = float(conf_np[best_idx])
            pred_phrase = phrases[best_idx] if best_idx < len(phrases) else ""
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
            mdice = mask_dice(pred_mask, gt_mask)

    return {
        "prompt": prompt,
        "num_detections": int(pred_xyxy_all.shape[0]),
        "box_iou": best_box_iou,
        "mask_iou": miou,
        "mask_dice": mdice,
        "pred_confidence": pred_conf,
        "pred_phrase": pred_phrase,
        "pred_bbox_xyxy": pred_xyxy.tolist() if pred_xyxy is not None else None,
        "_pred_xyxy": pred_xyxy,
        "_pred_mask": pred_mask,
    }


def summarize(records: List[dict]) -> dict:
    if not records:
        return {}
    box_ious = [r["box_iou"] for r in records]
    mask_ious = [r["mask_iou"] for r in records]
    return {
        "num_instances": len(records),
        "detection_rate": float(np.mean([r["num_detections"] > 0 for r in records])),
        "mean_box_iou": float(np.mean(box_ious)),
        "mean_mask_iou": float(np.mean(mask_ious)),
        "mean_mask_dice": float(np.mean([r["mask_dice"] for r in records])),
        "box_iou@0.5": float(np.mean([x >= 0.5 for x in box_ious])),
        "mask_iou@0.5": float(np.mean([x >= 0.5 for x in mask_ious])),
    }


def main() -> None:
    args = parse_args()
    strategies = args.strategies or all_strategies()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_root = out_dir / "visualizations"
    vis_root.mkdir(parents=True, exist_ok=True)

    for s in strategies:
        if s not in PROMPT_STRATEGIES:
            raise ValueError(f"Unknown strategy {s}")

    print(f"Strategies: {strategies}")
    coco = COCO(args.ann_file)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    samples = build_samples(coco, args.max_images, args.seed)
    print(f"Instances: {len(samples)} (seed={args.seed})")

    print("Loading models...")
    sam2_model = build_sam2(DEFAULT_SAM2_CFG, DEFAULT_SAM2_CKPT, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    gdino_model = load_model(DEFAULT_GDINO_CFG, DEFAULT_GDINO_CKPT, device=args.device)

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # strategy -> list of per-instance records (without internal vis fields)
    all_results: Dict[str, List[dict]] = {s: [] for s in strategies}
    # paired records for head-to-head analysis
    paired: List[dict] = []

    for idx, (img_info, ann) in enumerate(tqdm(samples, desc="Evaluating")):
        img_path = os.path.join(args.img_dir, img_info["file_name"])
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        category_name = cat_id_to_name[ann["category_id"]]
        gt_xyxy = xywh_to_xyxy(ann["bbox"])
        gt_mask = coco.annToMask(ann).astype(bool)
        image_rgb, image_tensor = load_image(img_path)

        pair_rec = {
            "index": idx,
            "image_id": img_info["id"],
            "ann_id": ann["id"],
            "file_name": img_info["file_name"],
            "category": category_name,
            "strategies": {},
        }

        for strategy in strategies:
            prompt = build_structured_prompt(category_name, strategy)
            res = run_one_instance(
                gdino_model,
                sam2_predictor,
                image_rgb,
                image_tensor,
                prompt,
                gt_xyxy,
                gt_mask,
                args.device,
                args.box_threshold,
                args.text_threshold,
            )

            if idx < args.save_vis and res["_pred_mask"] is not None:
                vis_dir = vis_root / strategy
                vis_dir.mkdir(parents=True, exist_ok=True)
                title = f"{strategy} | {primary_category_name(category_name)} | mIoU {res['mask_iou']:.2f}"
                cv2.imwrite(
                    str(vis_dir / f"{idx:04d}_miou{res['mask_iou']:.2f}.jpg"),
                    draw_comparison(
                        image_bgr, gt_xyxy, res["_pred_xyxy"], gt_mask, res["_pred_mask"], title
                    ),
                )

            clean = {k: v for k, v in res.items() if not k.startswith("_")}
            all_results[strategy].append(clean)
            pair_rec["strategies"][strategy] = clean

        paired.append(pair_rec)

    # Summaries
    summary = {
        "config": {
            "ann_file": args.ann_file,
            "img_dir": args.img_dir,
            "max_images": args.max_images,
            "seed": args.seed,
            "strategies": strategies,
        },
        "strategy_descriptions": PROMPT_STRATEGIES,
        "per_strategy": {s: summarize(all_results[s]) for s in strategies},
    }

    # Rank by mean_mask_iou
    ranking = sorted(
        strategies,
        key=lambda s: summary["per_strategy"][s].get("mean_mask_iou", 0),
        reverse=True,
    )
    summary["ranking_by_mask_iou"] = ranking
    summary["best_strategy"] = ranking[0] if ranking else None

    # Head-to-head vs baseline
    if "baseline" in strategies:
        base_mious = [r["mask_iou"] for r in all_results["baseline"]]
        for s in strategies:
            if s == "baseline":
                continue
            deltas = [r["mask_iou"] - b for r, b in zip(all_results[s], base_mious)]
            summary["per_strategy"][s]["vs_baseline_mask_iou"] = float(np.mean(deltas))

    with open(out_dir / "per_instance_paired.json", "w") as f:
        json.dump(paired, f, indent=2)
    with open(out_dir / "per_strategy.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown table for quick reading
    lines = [
        "# Prompt Strategy Comparison\n",
        f"Instances: {len(samples)} | seed: {args.seed}\n\n",
        "| Strategy | Description | Det.Rate | Box mIoU | **Mask mIoU** | vs baseline |",
        "|----------|-------------|----------|----------|---------------|-------------|",
    ]
    for s in ranking:
        st = summary["per_strategy"][s]
        vs = st.get("vs_baseline_mask_iou", 0.0)
        vs_str = "—" if s == "baseline" else f"{vs:+.3f}"
        lines.append(
            f"| {s} | {PROMPT_STRATEGIES[s]} | {st['detection_rate']:.2f} | "
            f"{st['mean_box_iou']:.3f} | **{st['mean_mask_iou']:.3f}** | {vs_str} |"
        )
    lines.append(f"\n**Best:** `{summary['best_strategy']}`\n")
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"{'Strategy':<14} {'Mask mIoU':>10} {'Box mIoU':>10} {'Det%':>8}")
    print("-" * 60)
    for s in ranking:
        st = summary["per_strategy"][s]
        print(
            f"{s:<14} {st['mean_mask_iou']:>10.3f} {st['mean_box_iou']:>10.3f} "
            f"{st['detection_rate']*100:>7.1f}%"
        )
    print("=" * 60)
    print(f"Best strategy: {summary['best_strategy']}")
    print(f"Results -> {out_dir}")


if __name__ == "__main__":
    main()
