#!/usr/bin/env python3
"""Merge LoRA checkpoint into base SwinT weights for Grounded-SAM-2 inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from groundingdino.util.inference import load_model
from groundingdino.util.misc import clean_state_dict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "weights/groundingdino_swint_ogc.pth"
DEFAULT_CFG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export merged Grounding DINO LoRA weights")
    p.add_argument(
        "--lora",
        required=True,
        help="LoRA checkpoint path, e.g. weights/fashionpedia_lora/.../checkpoint_epoch_10.pth",
    )
    p.add_argument("--base", default=str(DEFAULT_BASE), help="Base SwinT checkpoint")
    p.add_argument("--config", default=DEFAULT_CFG, help="Model config py file")
    p.add_argument(
        "--output",
        default=None,
        help="Output .pth path (default: same dir as lora, merged_swint_ogc.pth)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lora_path = Path(args.lora).resolve()
    if not lora_path.is_file():
        raise FileNotFoundError(lora_path)

    out_path = Path(args.output) if args.output else lora_path.parent / "merged_swint_ogc.pth"

    model_config = ModelConfig(
        config_path=args.config,
        weights_path=str(Path(args.base).resolve()),
        lora_weigths=str(lora_path),
    )
    print(f"Loading base: {model_config.weights_path}")
    print(f"Loading LoRA: {model_config.lora_weigths}")
    model = load_model(model_config, use_lora=True)
    model.eval()

    torch.save(
        {"model": clean_state_dict(model.state_dict())},
        out_path,
    )
    print(f"Saved merged weights -> {out_path}")


if __name__ == "__main__":
    main()
