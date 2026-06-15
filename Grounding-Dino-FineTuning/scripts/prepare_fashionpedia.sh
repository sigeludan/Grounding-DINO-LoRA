#!/usr/bin/env bash
# Prepare Fashionpedia data + environment for Grounding DINO LoRA fine-tuning.
set -euo pipefail

ROOT="/root/autodl-tmp/Grounding-Dino-FineTuning"
FP="/root/autodl-tmp/fashionpedia"
GS2="/root/autodl-tmp/Grounded-SAM-2"

cd "$ROOT"

echo "==> 1) Link pretrained SwinT weights"
mkdir -p weights
ln -sf "$GS2/gdino_checkpoints/groundingdino_swint_ogc.pth" \
  "$ROOT/weights/groundingdino_swint_ogc.pth"

echo "==> 2) Convert Fashionpedia annotations to CSV"
mkdir -p "$FP/gdino_csv"
python tools/convert_fashionpedia_to_csv.py \
  --ann "$FP/annotations/instances_attributes_train2020.json" \
  --img-dir "$FP/images/train" \
  --out "$FP/gdino_csv/train_annotations.csv" \
  --max-images 5000 \
  --seed 42

python tools/convert_fashionpedia_to_csv.py \
  --ann "$FP/annotations/instances_attributes_val2020.json" \
  --img-dir "$FP/images/test" \
  --out "$FP/gdino_csv/val_annotations.csv"

echo "==> 3) Install Python dependencies"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
pip install -U "setuptools>=69" wheel
pip install -q "transformers==4.41.2" "peft==0.11.1" ema_pytorch pyyaml
pip install -q -r requirements.txt
pip install -q --no-build-isolation -e . || true
python setup.py build_ext --inplace
echo "/root/autodl-tmp/Grounding-Dino-FineTuning" > /root/miniconda3/lib/python3.12/site-packages/_gdino_finetune_path.pth

echo "==> 4) Quick sanity check"
python - <<'PY'
import csv
from pathlib import Path
train_csv = Path("/root/autodl-tmp/fashionpedia/gdino_csv/train_annotations.csv")
with train_csv.open() as f:
    rows = list(csv.DictReader(f))
print("train rows:", len(rows))
print("sample:", rows[0])
PY

echo ""
echo "Done. Train with:"
echo "  bash $ROOT/scripts/run_train.sh"
