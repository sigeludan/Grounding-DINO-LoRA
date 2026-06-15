# Fashionpedia LoRA 微调

## 一键准备环境

```bash
cd /root/autodl-tmp/Grounding-Dino-FineTuning
bash scripts/prepare_fashionpedia.sh
```

包含：权重软链、CSV 数据转换、依赖安装、CUDA 算子编译。

## 开始训练

```bash
cd /root/autodl-tmp/Grounding-Dino-FineTuning
bash scripts/run_train.sh
```

或手动：

```bash
cd /root/autodl-tmp/Grounding-Dino-FineTuning
export CUDA_HOME=/usr/local/cuda
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选
export PYTHONPATH="/root/autodl-tmp/Grounding-Dino-FineTuning:$PYTHONPATH"
python train.py --config configs/fashionpedia_train_config.yaml
```

## 配置说明

| 项 | 值 |
|---|---|
| 数据 | 5000 train / 1158 val |
| 模型 | SwinT + LoRA rank=32 |
| Epochs | 10，每 2 epoch 存一次 |
| LR | 1e-4，batch=4 |
| 输出 | `weights/fashionpedia_lora/YYYYMMDD_HHMM/checkpoint_epoch_*.pth` |

## 依赖版本

- `transformers==4.41.2`（与 Grounded-SAM-2 一致）
- `peft==0.11.1`
- 不要用 `setuptools==65.5.1`（Python 3.12 不兼容）

## 训练后

将 LoRA checkpoint 接入 `Grounded-SAM-2/fashionpedia_eval.py`，用 `--prompt-strategy synonyms` 对比 zero-shot vs LoRA。
