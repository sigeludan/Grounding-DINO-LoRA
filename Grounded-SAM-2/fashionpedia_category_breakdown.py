#!/usr/bin/env python3
"""
Per-category breakdown: compare two fashionpedia_eval.py runs (e.g. LoRA vs zero-shot).

Reads paired per_instance.json files (same val order / ann_id), aggregates mask IoU,
box IoU, and detection rate by category, and reports LoRA − zero-shot deltas.

Usage:
  python fashionpedia_category_breakdown.py \\
    --lora-dir outputs/fashionpedia_lora_v2_synonyms_th015_best \\
    --zs-dir outputs/fashionpedia_zero_shot_synonyms_th015_best
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ANN = PROJECT_ROOT.parent / "fashionpedia/annotations/instances_attributes_val2020.json"


@dataclass
class CatStats:
    n: int = 0
    mask_iou_sum: float = 0.0
    box_iou_sum: float = 0.0
    det_count: int = 0
    delta_mask_sum: float = 0.0
    delta_box_sum: float = 0.0
    lora_wins: int = 0
    zs_wins: int = 0
    ties: int = 0

    @property
    def mask_iou(self) -> float:
        return self.mask_iou_sum / max(self.n, 1)

    @property
    def box_iou(self) -> float:
        return self.box_iou_sum / max(self.n, 1)

    @property
    def det_rate(self) -> float:
        return self.det_count / max(self.n, 1)

    @property
    def delta_mask(self) -> float:
        return self.delta_mask_sum / max(self.n, 1)

    @property
    def delta_box(self) -> float:
        return self.delta_box_sum / max(self.n, 1)


def load_records(path: Path) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def load_supercategories(ann_file: Optional[Path]) -> Dict[str, str]:
    if ann_file is None or not ann_file.exists():
        return {}
    with open(ann_file) as f:
        data = json.load(f)
    return {c["name"]: c.get("supercategory", "") for c in data["categories"]}


def index_by_ann_id(records: List[dict]) -> Dict[int, dict]:
    return {r["ann_id"]: r for r in records}


def breakdown(
    lora_records: List[dict],
    zs_records: List[dict],
    supercats: Dict[str, str],
) -> Dict[str, CatStats]:
    lora_by_id = index_by_ann_id(lora_records)
    zs_by_id = index_by_ann_id(zs_records)
    common_ids = sorted(set(lora_by_id) & set(zs_by_id))
    if len(common_ids) != len(lora_records) or len(common_ids) != len(zs_records):
        raise ValueError(
            f"ann_id mismatch: lora={len(lora_records)} zs={len(zs_records)} common={len(common_ids)}"
        )

    by_cat: Dict[str, CatStats] = defaultdict(CatStats)
    for ann_id in common_ids:
        lr = lora_by_id[ann_id]
        zr = zs_by_id[ann_id]
        cat = lr["category"]
        if zr["category"] != cat:
            raise ValueError(f"category mismatch ann_id={ann_id}: {cat} vs {zr['category']}")

        st = by_cat[cat]
        st.n += 1
        st.mask_iou_sum += lr["mask_iou"]
        st.box_iou_sum += lr["box_iou"]
        st.det_count += int(lr["detected"])

        d_mask = lr["mask_iou"] - zr["mask_iou"]
        d_box = lr["box_iou"] - zr["box_iou"]
        st.delta_mask_sum += d_mask
        st.delta_box_sum += d_box
        if d_mask > 1e-6:
            st.lora_wins += 1
        elif d_mask < -1e-6:
            st.zs_wins += 1
        else:
            st.ties += 1

    return dict(by_cat)


def zs_stats_by_cat(zs_records: List[dict]) -> Dict[str, CatStats]:
    by_cat: Dict[str, CatStats] = defaultdict(CatStats)
    for r in zs_records:
        st = by_cat[r["category"]]
        st.n += 1
        st.mask_iou_sum += r["mask_iou"]
        st.box_iou_sum += r["box_iou"]
        st.det_count += int(r["detected"])
    return dict(by_cat)


def rows_for_table(
    by_cat: Dict[str, CatStats],
    zs_by_cat: Dict[str, CatStats],
    supercats: Dict[str, str],
    min_count: int,
) -> List[dict]:
    rows = []
    for cat, st in by_cat.items():
        if st.n < min_count:
            continue
        zs = zs_by_cat[cat]
        rows.append(
            {
                "category": cat,
                "supercategory": supercats.get(cat, ""),
                "n": st.n,
                "zs_mask_iou": zs.mask_iou,
                "lora_mask_iou": st.mask_iou,
                "delta_mask_iou": st.delta_mask,
                "zs_box_iou": zs.box_iou,
                "lora_box_iou": st.box_iou,
                "delta_box_iou": st.delta_box,
                "lora_det_rate": st.det_rate,
                "lora_wins": st.lora_wins,
                "zs_wins": st.zs_wins,
                "ties": st.ties,
            }
        )
    rows.sort(key=lambda r: r["delta_mask_iou"], reverse=True)
    return rows


def print_table(rows: List[dict]) -> None:
    print(
        f"{'category':<42} {'n':>4}  {'ZS mIoU':>7} {'LoRA':>7} {'Δ':>7}  "
        f"{'win/loss':>8}  supercat"
    )
    print("-" * 95)
    for r in rows:
        wl = f"{r['lora_wins']}/{r['zs_wins']}"
        print(
            f"{r['category']:<42} {r['n']:4d}  "
            f"{r['zs_mask_iou']:7.3f} {r['lora_mask_iou']:7.3f} {r['delta_mask_iou']:+7.3f}  "
            f"{wl:>8}  {r['supercategory']}"
        )


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_markdown(path: Path, rows: List[dict], lora_label: str, zs_label: str) -> None:
    lines = [
        f"# Per-category breakdown: {lora_label} vs {zs_label}",
        "",
        "| Category | n | ZS mIoU | LoRA mIoU | Δ mIoU | ZS box | LoRA box | Δ box | LoRA wins |",
        "|----------|--:|--------:|----------:|-------:|-------:|---------:|------:|----------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['category']} | {r['n']} | {r['zs_mask_iou']:.3f} | {r['lora_mask_iou']:.3f} | "
            f"{r['delta_mask_iou']:+.3f} | {r['zs_box_iou']:.3f} | {r['lora_box_iou']:.3f} | "
            f"{r['delta_box_iou']:+.3f} | {r['lora_wins']}/{r['zs_wins']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-category LoRA vs zero-shot breakdown")
    p.add_argument("--lora-dir", type=Path, help="LoRA eval output dir (contains per_instance.json)")
    p.add_argument("--zs-dir", type=Path, help="Zero-shot eval output dir")
    p.add_argument("--lora-json", type=Path, help="LoRA per_instance.json path")
    p.add_argument("--zs-json", type=Path, help="Zero-shot per_instance.json path")
    p.add_argument("--ann-file", type=Path, default=DEFAULT_ANN, help="For supercategory labels")
    p.add_argument("--output-dir", type=Path, default=None, help="Write CSV/MD/JSON here")
    p.add_argument("--min-count", type=int, default=1, help="Min instances per category to show")
    p.add_argument("--lora-label", default="LoRA v2 th015")
    p.add_argument("--zs-label", default="Zero-shot th015")
    return p.parse_args()


def resolve_json(arg_dir: Optional[Path], arg_json: Optional[Path], name: str) -> Path:
    if arg_json is not None:
        return arg_json
    if arg_dir is not None:
        return arg_dir / "per_instance.json"
    raise ValueError(f"Provide --{name}-dir or --{name}-json")


def main() -> None:
    args = parse_args()
    lora_path = resolve_json(args.lora_dir, args.lora_json, "lora")
    zs_path = resolve_json(args.zs_dir, args.zs_json, "zs")
    out_dir = args.output_dir or (args.lora_dir or lora_path.parent) / "category_breakdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    lora_records = load_records(lora_path)
    zs_records = load_records(zs_path)
    supercats = load_supercategories(args.ann_file)

    by_cat = breakdown(lora_records, zs_records, supercats)
    zs_by_cat = zs_stats_by_cat(zs_records)
    rows = rows_for_table(by_cat, zs_by_cat, supercats, args.min_count)

    print(f"LoRA: {lora_path}")
    print(f"ZS:   {zs_path}")
    print(f"Instances: {len(lora_records)} | Categories shown (n>={args.min_count}): {len(rows)}")
    print()
    print_table(rows)

    # Macro average over categories (unweighted by n)
    if rows:
        macro_delta = sum(r["delta_mask_iou"] for r in rows) / len(rows)
        micro_delta = sum(by_cat[r["category"]].delta_mask_sum for r in rows) / sum(r["n"] for r in rows)
        print()
        print(f"Micro-averaged Δ mIoU (instance-weighted): {micro_delta:+.4f}")
        print(f"Macro-averaged Δ mIoU (per-category mean): {macro_delta:+.4f}")

    payload = {
        "lora_json": str(lora_path),
        "zs_json": str(zs_path),
        "lora_label": args.lora_label,
        "zs_label": args.zs_label,
        "num_instances": len(lora_records),
        "categories": rows,
    }
    write_csv(out_dir / "category_breakdown.csv", rows)
    write_markdown(out_dir / "category_breakdown.md", rows, args.lora_label, args.zs_label)
    with open(out_dir / "category_breakdown.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote -> {out_dir}/")


if __name__ == "__main__":
    main()
