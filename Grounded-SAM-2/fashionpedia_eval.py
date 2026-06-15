#!/usr/bin/env python3
"""
Fashionpedia val baseline: Grounding DINO (text) -> box -> SAM2 mask.

Usage (from Grounded-SAM-2 repo root):
  export TRANSFORMERS_OFFLINE=1
  python fashionpedia_eval.py --max-images 50 --prompt-mode category
  python fashionpedia_eval.py --max-images 50 --prompt-mode category --prompt-strategy synonyms
  python fashionpedia_prompt_eval.py --max-images 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from pycocotools.coco import COCO
from torchvision.ops import box_convert
from tqdm import tqdm

from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
from prompt_templates import PROMPT_STRATEGIES, build_structured_prompt, primary_category_name
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANN = "/root/autodl-tmp/fashionpedia/annotations/instances_attributes_val2020.json"
DEFAULT_IMG_DIR = "/root/autodl-tmp/fashionpedia/images/test"
DEFAULT_SAM2_CKPT = str(PROJECT_ROOT / "checkpoints/sam2.1_hiera_large.pt")
DEFAULT_GDINO_CKPT = str(PROJECT_ROOT / "gdino_checkpoints/groundingdino_swint_ogc.pth")
DEFAULT_GDINO_CFG = str(PROJECT_ROOT / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


@dataclass
class EvalConfig:
    ann_file: str = DEFAULT_ANN
    img_dir: str = DEFAULT_IMG_DIR
    output_dir: str = str(PROJECT_ROOT / "outputs/fashionpedia_baseline")
    prompt_mode: str = "category"  # category | attr
    prompt_strategy: str = "baseline"  # see prompt_templates.PROMPT_STRATEGIES
    max_images: int = 50
    seed: int = 42
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    save_vis: int = 20
    device: str = "cuda"
    gdino_checkpoint: Optional[str] = None


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Fashionpedia GDINO+SAM2 baseline eval")
    p.add_argument("--ann-file", default=DEFAULT_ANN)
    p.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--prompt-mode", choices=["category", "attr"], default="category")
    p.add_argument(
        "--prompt-strategy",
        choices=list(PROMPT_STRATEGIES.keys()),
        default="baseline",
        help="structured template when prompt-mode=category",
    )
    p.add_argument("--max-images", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--save-vis", type=int, default=20, help="number of visualizations to save")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--gdino-checkpoint",
        default=None,
        help="Override GDINO weights (e.g. merged LoRA export)",
    )
    args = p.parse_args()
    out = args.output_dir or str(
        PROJECT_ROOT
        / f"outputs/fashionpedia_baseline_{args.prompt_mode}_{args.prompt_strategy}"
        if args.prompt_mode == "category"
        else PROJECT_ROOT / f"outputs/fashionpedia_baseline_{args.prompt_mode}"
    )
    return EvalConfig(
        ann_file=args.ann_file,
        img_dir=args.img_dir,
        output_dir=out,
        prompt_mode=args.prompt_mode,
        prompt_strategy=args.prompt_strategy,
        max_images=args.max_images,
        seed=args.seed,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        save_vis=args.save_vis,
        device=args.device,
        gdino_checkpoint=args.gdino_checkpoint,
    )


def build_prompt(
    category_name: str,
    attribute_ids: List[int],
    attr_id_to_name: Dict[int, str],
    mode: str,
    strategy: str = "baseline",
) -> str:
    if mode == "attr":
        cat = primary_category_name(category_name)
        parts = [cat] + [attr_id_to_name[i].lower() for i in attribute_ids if i in attr_id_to_name]
        return " . ".join(parts) + "."
    return build_structured_prompt(category_name, strategy)


def xywh_to_xyxy(bbox: List[float]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float(inter / union) if union > 0 else 0.0


def mask_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    return float(2 * inter / denom) if denom > 0 else 0.0


def build_samples(coco: COCO, max_images: int, seed: int) -> List[Tuple[dict, dict]]:
    rng = random.Random(seed)
    img_ids = list(coco.getImgIds())
    rng.shuffle(img_ids)
    samples: List[Tuple[dict, dict]] = []
    for img_id in img_ids:
        if len(samples) >= max_images:
            break
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        if not ann_ids:
            continue
        img_info = coco.loadImgs(img_id)[0]
        ann = coco.loadAnns(ann_ids[0])[0]
        samples.append((img_info, ann))
    return samples


def pick_best_box(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray) -> Tuple[Optional[int], float]:
    if pred_xyxy.shape[0] == 0:
        return None, 0.0
    ious = [box_iou_xyxy(pred_xyxy[i], gt_xyxy) for i in range(pred_xyxy.shape[0])]
    best_idx = int(np.argmax(ious))
    return best_idx, float(ious[best_idx])


def draw_comparison(
    image_bgr: np.ndarray,
    gt_xyxy: np.ndarray,
    pred_xyxy: Optional[np.ndarray],
    gt_mask: np.ndarray,
    pred_mask: Optional[np.ndarray],
    title: str,
) -> np.ndarray:
    vis = image_bgr.copy()
    gx1, gy1, gx2, gy2 = gt_xyxy.astype(int)
    cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 200, 0), 2)
    cv2.putText(vis, "GT", (gx1, max(0, gy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    if pred_xyxy is not None:
        px1, py1, px2, py2 = pred_xyxy.astype(int)
        cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.putText(vis, "PRED", (px1, min(vis.shape[0] - 4, py2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    overlay = vis.copy()
    overlay[gt_mask.astype(bool)] = (overlay[gt_mask.astype(bool)] * 0.5 + np.array([0, 180, 0]) * 0.5).astype(np.uint8)
    if pred_mask is not None:
        overlay[pred_mask.astype(bool)] = (overlay[pred_mask.astype(bool)] * 0.5 + np.array([0, 0, 220]) * 0.5).astype(np.uint8)
    vis = cv2.addWeighted(overlay, 0.7, vis, 0.3, 0)

    cv2.putText(vis, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return vis


def save_supervision_vis(
    image_bgr: np.ndarray,
    pred_xyxy: np.ndarray,
    pred_mask: np.ndarray,
    labels: List[str],
    out_path: Path,
) -> None:
    detections = sv.Detections(
        xyxy=pred_xyxy,
        mask=pred_mask.astype(bool),
        class_id=np.arange(len(pred_xyxy)),
    )
    scene = image_bgr.copy()
    scene = sv.BoxAnnotator().annotate(scene=scene, detections=detections, labels=labels)
    scene = sv.MaskAnnotator().annotate(scene=scene, detections=detections)
    cv2.imwrite(str(out_path), scene)


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations: {cfg.ann_file}")
    coco = COCO(cfg.ann_file)
    cat_id_to_name = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    attr_id_to_name = {a["id"]: a["name"] for a in coco.dataset.get("attributes", [])}

    samples = build_samples(coco, cfg.max_images, cfg.seed)
    print(f"Evaluating {len(samples)} instances | prompt_mode={cfg.prompt_mode} strategy={cfg.prompt_strategy}")

    print("Loading SAM2...")
    sam2_model = build_sam2(DEFAULT_SAM2_CFG, DEFAULT_SAM2_CKPT, device=cfg.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print("Loading Grounding DINO...")
    gdino_ckpt = cfg.gdino_checkpoint or DEFAULT_GDINO_CKPT
    print(f"  checkpoint: {gdino_ckpt}")
    gdino_model = load_model(DEFAULT_GDINO_CFG, gdino_ckpt, device=cfg.device)

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    records: List[dict] = []
    box_ious: List[float] = []
    mask_ious: List[float] = []
    mask_dices: List[float] = []
    det_hits = 0

    for idx, (img_info, ann) in enumerate(tqdm(samples, desc="Evaluating")):
        img_path = os.path.join(cfg.img_dir, img_info["file_name"])
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        category_name = cat_id_to_name[ann["category_id"]]
        prompt = build_prompt(
            category_name,
            ann.get("attribute_ids", []),
            attr_id_to_name,
            cfg.prompt_mode,
            cfg.prompt_strategy,
        )
        gt_xyxy = xywh_to_xyxy(ann["bbox"])
        gt_mask = coco.annToMask(ann).astype(bool)

        image_rgb, image_tensor = load_image(img_path)
        h, w = image_rgb.shape[:2]

        boxes, confidences, phrases = predict(
            model=gdino_model,
            image=image_tensor,
            caption=prompt,
            box_threshold=cfg.box_threshold,
            text_threshold=cfg.text_threshold,
            device=cfg.device,
        )

        if boxes.shape[0] > 0:
            boxes = boxes * torch.tensor([w, h, w, h])
            pred_xyxy_all = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            conf_np = confidences.numpy()
        else:
            pred_xyxy_all = np.zeros((0, 4), dtype=np.float32)
            conf_np = np.array([])

        best_idx, best_box_iou = pick_best_box(pred_xyxy_all, gt_xyxy)
        detected = best_idx is not None and best_box_iou > 0.0
        if detected:
            det_hits += 1

        pred_mask = None
        pred_xyxy = None
        pred_conf = 0.0
        pred_phrase = ""
        miou = 0.0
        mdice = 0.0

        with torch.inference_mode(), torch.autocast(cfg.device, dtype=torch.bfloat16):
            sam2_predictor.set_image(image_rgb)
            if best_idx is not None:
                pred_xyxy = pred_xyxy_all[best_idx]
                pred_conf = float(conf_np[best_idx])
                pred_phrase = phrases[best_idx] if best_idx < len(phrases) else ""
                masks, scores, _ = sam2_predictor.predict(
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

        rec = {
            "index": idx,
            "image_id": img_info["id"],
            "ann_id": ann["id"],
            "file_name": img_info["file_name"],
            "category": category_name,
            "prompt": prompt,
            "num_detections": int(pred_xyxy_all.shape[0]),
            "detected": detected,
            "box_iou": best_box_iou,
            "mask_iou": miou,
            "mask_dice": mdice,
            "pred_confidence": pred_conf,
            "pred_phrase": pred_phrase,
            "gt_bbox_xywh": ann["bbox"],
            "pred_bbox_xyxy": pred_xyxy.tolist() if pred_xyxy is not None else None,
        }
        records.append(rec)

        if idx < cfg.save_vis:
            title = f"{primary_category_name(category_name)} | box={best_box_iou:.2f} mask={miou:.2f}"
            cmp_path = vis_dir / f"{idx:04d}_{primary_category_name(category_name)}_miou{miou:.2f}.jpg"
            cv2.imwrite(
                str(cmp_path),
                draw_comparison(image_bgr, gt_xyxy, pred_xyxy, gt_mask, pred_mask, title),
            )
            if pred_xyxy is not None and pred_mask is not None:
                save_supervision_vis(
                    image_bgr,
                    pred_xyxy[np.newaxis],
                    pred_mask[np.newaxis],
                    [f"{pred_phrase} {pred_conf:.2f}"],
                    vis_dir / f"{idx:04d}_pred_overlay.jpg",
                )

    summary = {
        "config": asdict(cfg),
        "num_instances": len(records),
        "detection_rate": det_hits / max(len(records), 1),
        "mean_box_iou": float(np.mean(box_ious)) if box_ious else 0.0,
        "mean_mask_iou": float(np.mean(mask_ious)) if mask_ious else 0.0,
        "mean_mask_dice": float(np.mean(mask_dices)) if mask_dices else 0.0,
        "box_iou@0.5": float(np.mean([x >= 0.5 for x in box_ious])) if box_ious else 0.0,
        "mask_iou@0.5": float(np.mean([x >= 0.5 for x in mask_ious])) if mask_ious else 0.0,
    }

    with open(out_dir / "per_instance.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 48)
    print(f"Instances evaluated : {summary['num_instances']}")
    print(f"Detection rate      : {summary['detection_rate']:.4f}")
    print(f"Mean box IoU        : {summary['mean_box_iou']:.4f}")
    print(f"Mean mask mIoU      : {summary['mean_mask_iou']:.4f}")
    print(f"Mean mask Dice      : {summary['mean_mask_dice']:.4f}")
    print(f"Box IoU@0.5         : {summary['box_iou@0.5']:.4f}")
    print(f"Mask IoU@0.5        : {summary['mask_iou@0.5']:.4f}")
    print("=" * 48)
    print(f"Results -> {out_dir}")
    print(f"Visualizations -> {vis_dir}")


if __name__ == "__main__":
    main()
