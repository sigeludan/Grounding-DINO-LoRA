#!/usr/bin/env bash
# Start Fashionpedia LoRA training (online BERT from HuggingFace).
set -euo pipefail
cd /root/autodl-tmp/Grounding-Dino-FineTuning
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PYTHONPATH="/root/autodl-tmp/Grounding-Dino-FineTuning:${PYTHONPATH:-}"
# AutoDL 国内镜像（可选，下载 BERT 更快）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
python train.py --config configs/fashionpedia_train_config.yaml "$@"
