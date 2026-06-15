#!/usr/bin/env python3
"""Generate summary figures for Fashionpedia experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
OUT = PROJECT_ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_summary(path: str) -> float:
    return json.load(open(path))["mean_mask_iou"]


def fig_overall_miou() -> Path:
    models = ["Zero-shot", "LoRA r16", "LoRA r32"]
    paths = [
        "outputs/fashionpedia_zero_shot_synonyms_th015_best/summary.json",
        "outputs/fashionpedia_lora_r16_synonyms_th015/summary.json",
        "outputs/fashionpedia_lora_v2_synonyms_th015_best/summary.json",
    ]
    values = [load_summary(p) * 100 for p in paths]
    colors = ["#6baed6", "#9ecae1", "#2171b5"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(models, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Mask mIoU (%)")
    ax.set_title("Fashionpedia val (1158) — synonyms, th=0.15/0.15")
    ax.set_ylim(0, 80)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)
    fig.tight_layout()
    path = OUT / "overall_mask_miou.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_category_delta() -> Path:
    data = json.load(open("outputs/category_breakdown_th015_lora_vs_zs_min10/category_breakdown.json"))
    # highlight categories + garment parts aggregate
    cats = ["neckline", "pocket", "jacket", "top, t-shirt, sweatshirt", "dress", "shoe"]
    rows = {r["category"]: r for r in data["categories"]}
    names = [c for c in cats if c in rows]
    deltas = [rows[c]["delta_mask_iou"] * 100 for c in names]
    labels = [c.replace(", ", "\n") for c in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#238b45" if d >= 0 else "#cb181d" for d in deltas]
    ax.barh(labels, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ mask mIoU (LoRA − Zero-shot, pp)")
    ax.set_title("Per-category Δ (n≥10)")
    fig.tight_layout()
    path = OUT / "category_delta_min10.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_threshold_sensitivity() -> Path:
    # from sweep grid at fixed text=0.15 for ZS and LoRA r32
    zs = json.load(open("outputs/threshold_sweep_20260615_102629/zero_shot/grid_results.json"))
    lora = json.load(open("outputs/threshold_sweep_20260615_102629/lora_v2/grid_results.json"))
    box_ths = sorted({r["box_threshold"] for r in zs})

    def series(grid, text_th=0.15):
        return [
            next(r["mean_mask_iou"] for r in grid if r["box_threshold"] == b and r["text_threshold"] == text_th) * 100
            for b in box_ths
        ]

    zs_y = series(zs)
    lora_y = series(lora)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(box_ths, zs_y, "o-", label="Zero-shot", color="#6baed6", linewidth=2)
    ax.plot(box_ths, lora_y, "s-", label="LoRA r32", color="#2171b5", linewidth=2)
    ax.axvline(0.15, color="gray", linestyle="--", alpha=0.7, label="optimal 0.15")
    ax.set_xlabel("Box threshold (text=0.15)")
    ax.set_ylabel("Mask mIoU (%) — subset 200")
    ax.set_title("Threshold sensitivity")
    ax.legend()
    fig.tight_layout()
    path = OUT / "threshold_sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    paths = [fig_overall_miou(), fig_category_delta(), fig_threshold_sensitivity()]
    print("Saved figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
