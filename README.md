# Fashionpedia 文本驱动分割实验

Grounding DINO + SAM2 在 [Fashionpedia](https://fashionpedia.github.io/) val 上的 zero-shot、Prompt 工程与 LoRA 微调实验。

详细实验记录见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 环境

```bash
cd /root/autodl-tmp/Grounded-SAM-2
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选
```

依赖：`transformers==4.41.2`、`peft==0.11.1`、SAM2、Grounding DINO（见 `EXPERIMENT_LOG.md` §1）。

## 数据获取与目录结构

> **本仓库不包含 Fashionpedia 数据集**。需自行下载并解压。

### 官方来源

- 项目主页：[Fashionpedia](https://fashionpedia.github.io/home/)
- ECCV 2020 论文：[Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460307.pdf)
- 数据托管（Amazon S3，`ifashionist-dataset`）：

| 文件          | 下载链接                                                     |
| ------------- | ------------------------------------------------------------ |
| 训练集图片    | https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip |
| 验证/测试图片 | https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip |
| 训练集标注    | https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json |
| 验证集标注    | https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json |

也可从 [Kaggle iMaterialist Fashion 2020](https://www.kaggle.com/competitions/imaterialist-fashion-2020-fgvc7/data) 等镜像获取同套文件。

### 推荐目录布局

```text
${FASHIONPEDIA_ROOT}/          # 例如 /data/fashionpedia
├── annotations/
│   ├── instances_attributes_train2020.json
│   └── instances_attributes_val2020.json
├── images/
│   ├── train/                 # 从 train2020.zip 解压
│   └── test/                  # 从 val_test2020.zip 解压（验证集图片在此目录）
└── gdino_csv/                 # 运行 prepare 脚本后生成，勿手改
    ├── train_annotations.csv
    └── val_annotations.csv
```

本仓库评测与 LoRA 训练使用：

| 用途                      | 图片目录        | 标注                                |
| ------------------------- | --------------- | ----------------------------------- |
| 评测（val，1158 图）      | `images/test/`  | `instances_attributes_val2020.json` |
| LoRA 训练（子集 5000 图） | `images/train/` | `gdino_csv/train_annotations.csv`   |
| LoRA 验证                 | `images/test/`  | `gdino_csv/val_annotations.csv`     |

### 一键下载与解压（示例）

```bash
# 自定义数据根目录（与仓库同级即可）
export FASHIONPEDIA_ROOT="${FASHIONPEDIA_ROOT:-/data/fashionpedia}"
mkdir -p "$FASHIONPEDIA_ROOT/annotations" "$FASHIONPEDIA_ROOT/images"

# 标注（JSON，体积小）
wget -O "$FASHIONPEDIA_ROOT/annotations/instances_attributes_train2020.json" \
  https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json
wget -O "$FASHIONPEDIA_ROOT/annotations/instances_attributes_val2020.json" \
  https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json

# 图片（zip，体积大，耗时较长）
wget -O /tmp/train2020.zip \
  https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip
wget -O /tmp/val_test2020.zip \
  https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip

unzip -q /tmp/train2020.zip -d "$FASHIONPEDIA_ROOT/images/"
unzip -q /tmp/val_test2020.zip -d "$FASHIONPEDIA_ROOT/images/"
# 解压后应存在 images/train/ 与 images/test/（若结构不同，请对齐到上述布局）
```

### 生成 LoRA 训练用 CSV

在 [Grounding-Dino-FineTuning](https://github.com/) 仓库中（与本仓库并列克隆）：

```bash
cd /path/to/Grounding-Dino-FineTuning

# 修改 scripts/prepare_fashionpedia.sh 内 FP= 为你的 FASHIONPEDIA_ROOT，或手动：
mkdir -p "$FASHIONPEDIA_ROOT/gdino_csv"
python tools/convert_fashionpedia_to_csv.py \
  --ann "$FASHIONPEDIA_ROOT/annotations/instances_attributes_train2020.json" \
  --img-dir "$FASHIONPEDIA_ROOT/images/train" \
  --out "$FASHIONPEDIA_ROOT/gdino_csv/train_annotations.csv" \
  --max-images 5000 --seed 42

python tools/convert_fashionpedia_to_csv.py \
  --ann "$FASHIONPEDIA_ROOT/annotations/instances_attributes_val2020.json" \
  --img-dir "$FASHIONPEDIA_ROOT/images/test" \
  --out "$FASHIONPEDIA_ROOT/gdino_csv/val_annotations.csv"
```

`train_annotations.csv` 预期约 **36k 行**（5000 图上的实例框）；`val_annotations.csv` 约 **8774 行**。

### 评测时指定数据路径

`fashionpedia_eval.py` 支持命令行覆盖默认路径：

```bash
python fashionpedia_eval.py \
  --ann-file "$FASHIONPEDIA_ROOT/annotations/instances_attributes_val2020.json" \
  --img-dir "$FASHIONPEDIA_ROOT/images/test" \
  --max-images 1158 --prompt-strategy synonyms \
  --box-threshold 0.15 --text-threshold 0.15
```

LoRA 训练请在 `Grounding-Dino-FineTuning/configs/fashionpedia_train_config.yaml` 中将 `train_dir` / `val_dir` / `*_ann` 改为你的 `FASHIONPEDIA_ROOT` 下路径。

## 核心结论（th=0.15/0.15，synonyms prompt，1158 实例）

| 方法              | Mask mIoU | vs Zero-shot |
| ----------------- | --------: | -----------: |
| Zero-shot         |     0.735 |            — |
| LoRA r16          |     0.743 |       +0.8pp |
| **LoRA r32 (v2)** | **0.746** |   **+1.1pp** |

- **Prompt**：synonyms 相对 baseline +4.0pp（0.663，默认阈值）
- **阈值**：三模型最优均为 **0.15/0.15**；LoRA 在默认 0.35/0.25 下几乎不可用
- **按类**：garment parts +4.8pp；**neckline +12.6pp**（n=141）；pocket/jacket 等类 LoRA 可能下降

## 结果图表

![Overall mIoU](Grounded-SAM-2\outputs\figures\overall_mask_miou.png)

![Per-category Δ](Grounded-SAM-2\outputs\figures\category_delta_min10.png)

![Threshold sensitivity](Grounded-SAM-2\outputs\figures\threshold_sensitivity.png)

## Failure / Success 可视化

三列对比：**GT | Zero-shot | LoRA v2**（绿=GT，红=预测）

生成命令：

```bash
python fashionpedia_failure_viz.py \
  --categories neckline pocket jacket \
  --top-k 5 \
  --output-dir outputs/failure_viz_th015
```

输出目录：`outputs/failure_viz_th015/`  
案例索引：[outputs/failure_viz_th015/GALLERY.md](outputs/failure_viz_th015/GALLERY.md)

| 类别         | 选取策略           | 观察                                                 |
| ------------ | ------------------ | ---------------------------------------------------- |
| **neckline** | LoRA 提升最大 5 例 | ZS mIoU≈0，LoRA 拉到 0.65–0.71（**主成功叙事**）     |
| **pocket**   | LoRA 下降最大 5 例 | 多数 LoRA **完全漏检**（mIoU=0），ZS 已有较好框      |
| **jacket**   | LoRA 下降最大 5 例 | 前 2 例 LoRA 失败明显；其余 ZS/LoRA 均≈0.9（已饱和） |

## 常用命令

```bash
# 全量评测（zero-shot）
python fashionpedia_eval.py --max-images 1158 --prompt-strategy synonyms \
  --box-threshold 0.15 --text-threshold 0.15 \
  --output-dir outputs/fashionpedia_zero_shot_synonyms_th015_best

# 全量评测（LoRA v2）
python fashionpedia_eval.py --max-images 1158 --prompt-strategy synonyms \
  --box-threshold 0.15 --text-threshold 0.15 \
  --gdino-checkpoint /root/autodl-tmp/Grounding-Dino-FineTuning/weights/fashionpedia_lora/20260614_2321/merged_swint_ogc.pth \
  --output-dir outputs/fashionpedia_lora_v2_synonyms_th015_best

# 按类 breakdown（n≥10）
python fashionpedia_category_breakdown.py \
  --lora-dir outputs/fashionpedia_lora_v2_synonyms_th015_best \
  --zs-dir outputs/fashionpedia_zero_shot_synonyms_th015_best \
  --output-dir outputs/category_breakdown_th015_lora_vs_zs_min10 \
  --min-count 10

# 阈值扫描（子集 200）
python fashionpedia_threshold_sweep.py --subset-size 200

# 生成图表
python fashionpedia_make_figures.py

# Failure case 可视化
python fashionpedia_failure_viz.py --top-k 5
```

## 脚本一览

| 脚本                                 | 功能                    |
| ------------------------------------ | ----------------------- |
| `fashionpedia_eval.py`               | 单配置全量/子集评测     |
| `fashionpedia_prompt_eval.py`        | 多 Prompt 策略对比      |
| `fashionpedia_threshold_sweep.py`    | 阈值网格扫描            |
| `fashionpedia_category_breakdown.py` | 按服装类别统计 Δ mIoU   |
| `fashionpedia_failure_viz.py`        | GT / ZS / LoRA 对比图   |
| `fashionpedia_make_figures.py`       | 汇总柱状图/折线图       |
| `prompt_templates.py`                | synonyms 等 Prompt 模板 |

## LoRA 训练

仓库：`/root/autodl-tmp/Grounding-Dino-FineTuning`

```bash
cd /root/autodl-tmp/Grounding-Dino-FineTuning
bash scripts/run_train.sh
python tools/export_merged_lora.py \
  --lora weights/fashionpedia_lora/<run>/checkpoint_epoch_5.pth \
  --config groundingdino/config/GroundingDINO_SwinT_OGC_localbert.py
```

## 输出目录结构

```
outputs/
├── fashionpedia_zero_shot_synonyms_th015_best/   # ZS 全量结果
├── fashionpedia_lora_v2_synonyms_th015_best/     # LoRA r32 全量
├── fashionpedia_lora_r16_synonyms_th015/         # LoRA r16 全量
├── category_breakdown_th015_lora_vs_zs_min10/    # 按类统计
├── threshold_sweep_20260615_102629/              # ZS vs LoRA 扫阈
├── threshold_sweep_r16/                          # r16 扫阈
├── failure_viz_th015/                            # 可视化对比图
└── figures/                                      # 论文用汇总图
```

## 权重

| 模型            | 路径                                                         |
| --------------- | ------------------------------------------------------------ |
| GDINO SwinT     | `gdino_checkpoints/groundingdino_swint_ogc.pth`              |
| SAM2            | `checkpoints/sam2.1_hiera_large.pt`                          |
| LoRA r32 merged | `Grounding-Dino-FineTuning/weights/fashionpedia_lora/20260614_2321/merged_swint_ogc.pth` |
| LoRA r16 merged | `Grounding-Dino-FineTuning/weights/fashionpedia_lora/20260615_1330/merged_swint_ogc.pth` |
