#!/usr/bin/env python3
"""
Joint threshold sweep for zero-shot vs LoRA on a fixed val subset.

Caches GDINO forward pass per sample, then sweeps box/text thresholds
without re-running the full detector backbone each time.

Usage:
  cd /root/autodl-tmp/Grounded-SAM-2
  export TRANSFORMERS_OFFLINE=1
  python fashionpedia_threshold_sweep.py --subset-size 200
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    build_prompt,
    build_samples,
    mask_dice,
    mask_iou,
    pick_best_box,
    xywh_to_xyxy,
)
from grounding_dino.groundingdino.util.inference import load_image, load_model, preprocess_caption
from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LORA_V2_CKPT = (
    "/root/autodl-tmp/Grounding-Dino-FineTuning/weights/fashionpedia_lora/"
    "20260614_2321/merged_swint_ogc.pth"
)
DEFAULT_LORA_R16_CKPT = (
    "/root/autodl-tmp/Grounding-Dino-FineTuning/weights/fashionpedia_lora/"
    "20260615_1330/merged_swint_ogc.pth"
)


@dataclass
class SweepConfig:
    ann_file: str = DEFAULT_ANN
    img_dir: str = DEFAULT_IMG_DIR
    subset_size: int = 200
    seed: int = 42
    prompt_strategy: str = "synonyms"
    device: str = "cuda"
    box_thresholds: Tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
    text_thresholds: Tuple[float, ...] = (0.15, 0.20, 0.25, 0.30)
    output_dir: Optional[str] = None
    gdino_checkpoint: Optional[str] = None
    model_name: str = "lora_r16"


def parse_args() -> SweepConfig:
    p = argparse.ArgumentParser(description="Fashionpedia threshold sweep (zero-shot vs LoRA)")
    p.add_argument("--ann-file", default=DEFAULT_ANN)
    p.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    p.add_argument("--subset-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-strategy", default="synonyms")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: outputs/threshold_sweep_YYYYMMDD_HHMM",
    )
    p.add_argument(
        "--gdino-checkpoint",
        default=None,
        help="Sweep only this checkpoint (skip default zero-shot/lora_v2 pair)",
    )
    p.add_argument(
        "--model-name",
        default="lora_r16",
        help="Output subfolder name when using --gdino-checkpoint",
    )
    args = p.parse_args()
    out = args.output_dir or str(
        PROJECT_ROOT / "outputs" / f"threshold_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    return SweepConfig(
        ann_file=args.ann_file,
        img_dir=args.img_dir,
        subset_size=args.subset_size,
        seed=args.seed,
        prompt_strategy=args.prompt_strategy,
        device=args.device,
        output_dir=out,
        gdino_checkpoint=args.gdino_checkpoint,
        model_name=args.model_name,
    )


def gdino_forward(model, image_tensor: torch.Tensor, caption: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    caption = preprocess_caption(caption=caption)
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    return logits, boxes


def filter_predictions(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    caption: str,
    model,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    mask = logits.max(dim=1)[0] > box_threshold
    sel_logits = logits[mask]
    sel_boxes = boxes[mask]
    tokenized = model.tokenizer(caption)
    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, model.tokenizer).replace(".", "")
        for logit in sel_logits
    ]
    conf = sel_logits.max(dim=1)[0] if sel_logits.numel() else torch.tensor([])
    return sel_boxes, conf, phrases


def evaluate_thresholds_on_cached(
    cached_samples: List[dict],
    sam2_predictor: SAM2ImagePredictor,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> dict:
    box_ious: List[float] = []
    mask_ious: List[float] = []
    mask_dices: List[float] = []
    det_hits = 0

    for item in cached_samples:
        boxes, confidences, _ = filter_predictions(
            item["logits"],
            item["boxes"],
            item["prompt"],
            item["gdino_model"],
            box_threshold,
            text_threshold,
        )
        h, w = item["h"], item["w"]
        gt_xyxy = item["gt_xyxy"]
        gt_mask = item["gt_mask"]

        if boxes.shape[0] > 0:
            boxes = boxes * torch.tensor([w, h, w, h])
            pred_xyxy_all = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        else:
            pred_xyxy_all = np.zeros((0, 4), dtype=np.float32)

        best_idx, best_box_iou = pick_best_box(pred_xyxy_all, gt_xyxy)
        if best_idx is not None and best_box_iou > 0.0:
            det_hits += 1

        miou = 0.0
        mdice = 0.0
        if best_idx is not None:
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                sam2_predictor.set_image(item["image_rgb"])
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
                mdice = mask_dice(pred_mask, gt_mask)

        box_ious.append(best_box_iou)
        mask_ious.append(miou)
        mask_dices.append(mdice)

    n = max(len(box_ious), 1)
    return {
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "num_instances": len(box_ious),
        "detection_rate": det_hits / n,
        "mean_box_iou": float(np.mean(box_ious)) if box_ious else 0.0,
        "mean_mask_iou": float(np.mean(mask_ious)) if mask_ious else 0.0,
        "mean_mask_dice": float(np.mean(mask_dices)) if mask_dices else 0.0,
        "mask_iou@0.5": float(np.mean([x >= 0.5 for x in mask_ious])) if mask_ious else 0.0,
    }


def sweep_model(
    model_name: str,
    gdino_checkpoint: str,
    samples,
    coco: COCO,
    cat_id_to_name: dict,
    attr_id_to_name: dict,
    cfg: SweepConfig,
    sam2_predictor: SAM2ImagePredictor,
) -> Tuple[List[dict], dict]:
    print(f"\n=== Sweep: {model_name} ===")
    print(f"Checkpoint: {gdino_checkpoint}")

    gdino_model = load_model(DEFAULT_GDINO_CFG, gdino_checkpoint, device=cfg.device)
    cached_samples: List[dict] = []

    for img_info, ann in tqdm(samples, desc=f"Cache {model_name}"):
        img_path = str(Path(cfg.img_dir) / img_info["file_name"])
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        category_name = cat_id_to_name[ann["category_id"]]
        prompt = build_prompt(
            category_name,
            ann.get("attribute_ids", []),
            attr_id_to_name,
            "category",
            cfg.prompt_strategy,
        )
        image_rgb, image_tensor = load_image(img_path)
        h, w = image_rgb.shape[:2]
        logits, boxes = gdino_forward(gdino_model, image_tensor, prompt, cfg.device)

        cached_samples.append(
            {
                "image_id": img_info["id"],
                "ann_id": ann["id"],
                "file_name": img_info["file_name"],
                "category": category_name,
                "prompt": prompt,
                "gdino_model": gdino_model,
                "logits": logits,
                "boxes": boxes,
                "image_rgb": image_rgb,
                "h": h,
                "w": w,
                "gt_xyxy": xywh_to_xyxy(ann["bbox"]),
                "gt_mask": coco.annToMask(ann).astype(bool),
            }
        )

    grid_results: List[dict] = []
    combos = [(b, t) for b in cfg.box_thresholds for t in cfg.text_thresholds]
    n_samples = len(cached_samples)
    print(f"Sweeping {len(combos)} threshold combos x {n_samples} samples (SAM2)...")

    for box_th, text_th in tqdm(combos, desc=f"Sweep {model_name}"):
        metrics = evaluate_thresholds_on_cached(
            cached_samples,
            sam2_predictor,
            box_th,
            text_th,
            cfg.device,
        )
        grid_results.append(metrics)

    best = max(grid_results, key=lambda x: x["mean_mask_iou"])
    best_info = {
        "model": model_name,
        "checkpoint": gdino_checkpoint,
        "best_box_threshold": best["box_threshold"],
        "best_text_threshold": best["text_threshold"],
        "best_mean_mask_iou": best["mean_mask_iou"],
        "best_detection_rate": best["detection_rate"],
        "best_mean_box_iou": best["mean_box_iou"],
    }
    return grid_results, best_info


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    coco = COCO(cfg.ann_file)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    attr_id_to_name = {a["id"]: a["name"] for a in coco.dataset.get("attributes", [])}
    samples = build_samples(coco, cfg.subset_size, cfg.seed)
    n_combos = len(cfg.box_thresholds) * len(cfg.text_thresholds)
    models = (
        {cfg.model_name: cfg.gdino_checkpoint}
        if cfg.gdino_checkpoint
        else {
            "zero_shot": DEFAULT_GDINO_CKPT,
            "lora_v2": DEFAULT_LORA_V2_CKPT,
        }
    )
    est_sam2 = len(samples) * n_combos * len(models)
    print(
        f"Subset: {len(samples)} instances | grid: {n_combos} combos/model | "
        f"models: {list(models.keys())} | ~{est_sam2} SAM2 forwards"
    )

    subset_meta = [
        {
            "image_id": img_info["id"],
            "ann_id": ann["id"],
            "file_name": img_info["file_name"],
            "category_id": ann["category_id"],
            "category": cat_id_to_name[ann["category_id"]],
        }
        for img_info, ann in samples
    ]

    print("Loading SAM2...")
    sam2_model = build_sam2(DEFAULT_SAM2_CFG, DEFAULT_SAM2_CKPT, device=cfg.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    all_results: Dict[str, dict] = {}
    for model_name, ckpt in models.items():
        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        grid_results, best_info = sweep_model(
            model_name,
            ckpt,
            samples,
            coco,
            cat_id_to_name,
            attr_id_to_name,
            cfg,
            sam2_predictor,
        )

        with open(model_dir / "grid_results.json", "w") as f:
            json.dump(grid_results, f, indent=2)
        with open(model_dir / "best_threshold.json", "w") as f:
            json.dump(best_info, f, indent=2)

        all_results[model_name] = {
            "grid_results": grid_results,
            "best": best_info,
        }

        print(
            f"{model_name} best: box={best_info['best_box_threshold']}, "
            f"text={best_info['best_text_threshold']}, "
            f"mIoU={best_info['best_mean_mask_iou']:.4f}"
        )

        del grid_results
        torch.cuda.empty_cache()

    comparison = {
        "sweep_config": {
            **asdict(cfg),
            "box_thresholds": list(cfg.box_thresholds),
            "text_thresholds": list(cfg.text_thresholds),
        },
        "subset_size": len(samples),
        "subset_seed": cfg.seed,
        "models": {name: res["best"] for name, res in all_results.items()},
    }
    if "zero_shot" in all_results and "lora_v2" in all_results:
        comparison["zero_shot_best"] = all_results["zero_shot"]["best"]
        comparison["lora_v2_best"] = all_results["lora_v2"]["best"]
        comparison["delta_mask_iou_lora_minus_zero_shot"] = (
            all_results["lora_v2"]["best"]["best_mean_mask_iou"]
            - all_results["zero_shot"]["best"]["best_mean_mask_iou"]
        )

    with open(out_dir / "sweep_config.json", "w") as f:
        json.dump(comparison["sweep_config"], f, indent=2)
    with open(out_dir / "subset_samples.json", "w") as f:
        json.dump(subset_meta, f, indent=2)
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Human-readable CSV
    csv_lines = ["model,box_threshold,text_threshold,detection_rate,mean_box_iou,mean_mask_iou"]
    for model_name in models:
        for row in all_results[model_name]["grid_results"]:
            csv_lines.append(
                f"{model_name},{row['box_threshold']},{row['text_threshold']},"
                f"{row['detection_rate']:.4f},{row['mean_box_iou']:.4f},{row['mean_mask_iou']:.4f}"
            )
    with open(out_dir / "grid_results.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    print("\n" + "=" * 52)
    print(f"Subset size: {len(samples)} (seed={cfg.seed})")
    for model_name, best in comparison["models"].items():
        print(
            f"{model_name:12s} best mIoU: {best['best_mean_mask_iou']:.4f} "
            f"@ {best['best_box_threshold']}/{best['best_text_threshold']}"
        )
    if "delta_mask_iou_lora_minus_zero_shot" in comparison:
        print(f"Delta (LoRA v2 - ZS): {comparison['delta_mask_iou_lora_minus_zero_shot']:+.4f}")
    print("=" * 52)
    print(f"Results -> {out_dir}")


if __name__ == "__main__":
    main()
