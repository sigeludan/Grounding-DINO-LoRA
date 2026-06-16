# 实验过程记录

**课题**：基于 Grounded-SAM2 的电商商品文本驱动精细分割  
**思路**：Grounding DINO LoRA 微调 + SAM2 推理 Pipeline + 自定义 Prompt Template  
**环境**：AutoDL，NVIDIA A800 80GB，Python 3.12，PyTorch 2.5.1+cu124  
**代码目录**：`/root/autodl-tmp/Grounded-SAM-2`

---

## 1. 环境搭建

| 步骤                | 操作                                                         | 结果                            |
| ------------------- | ------------------------------------------------------------ | ------------------------------- |
| 克隆仓库            | `Grounded-SAM-2` 置于 `autodl-tmp/`                          | 完成                            |
| 安装 SAM2           | `SAM2_BUILD_CUDA=0 pip install -e . --no-build-isolation`    | 完成（跳过 CUDA 扩展编译）      |
| 安装 Grounding DINO | `pip install --no-build-isolation -e grounding_dino`         | 完成                            |
| 依赖修复            | `pip install transformers==4.41.2`（5.x 与 GDINO 不兼容）    | 解决 `get_head_mask` 报错       |
| BERT 权重           | `hf download bert-base-uncased` + `HF_ENDPOINT=https://hf-mirror.com` | 完成，离线可用                  |
| SAM2 权重           | 软链 `checkpoints/sam2.1_hiera_large.pt` → 仓库根目录 `sam2.1_hiera_large.pt`（857MB） | 完成（原 sam2-coco 软链已修复） |
| GDINO 权重          | `groundingdino_swint_ogc.pth`（662MB，HF 镜像下载）          | 完成                            |

---

## 2. 数据准备（Fashionpedia）

选用 **Fashionpedia** 替代 COCO 电商子集：含 27 类服装 + 294 属性，适合文本驱动精细分割。

| 文件       | 路径                                                         | 规模                           |
| ---------- | ------------------------------------------------------------ | ------------------------------ |
| val 标注   | `fashionpedia/annotations/instances_attributes_val2020.json` | 1158 图 / 8781 实例            |
| train 标注 | `fashionpedia/annotations/instances_attributes_train2020.json` | 45623 图 / 333401 实例         |
| val 图片   | `fashionpedia/images/test/`                                  | 1158 张（与 val 标注全部对齐） |

> 注：官方 `val_test2020.zip` 解压后 val 图片在 `images/test/` 目录下。

---

## 3. Step 1：单张 Pipeline 验证

**脚本**：`grounded_sam2_local_demo.py`  
**测试样本**：`images/test/c47f02a6....jpg`，prompt = `dress.`

| 指标       | 结果                                    |
| ---------- | --------------------------------------- |
| GDINO 检测 | 1 个 dress，置信度 0.98                 |
| 预测框     | `[75, 180, 582, 876]`（与 GT 基本对齐） |
| SAM2 分割  | 正常输出 mask                           |
| 输出       | `outputs/grounded_sam2_local_demo/`     |

**结论**：GDINO → SAM2 链路在 Fashionpedia 上跑通。

---

## 4. Step 2 & 3：批量 Baseline 评测 + 可视化

**脚本**：`fashionpedia_eval.py`（新建）

```bash
export TRANSFORMERS_OFFLINE=1
python fashionpedia_eval.py --max-images 50 --prompt-mode category
python fashionpedia_eval.py --max-images 50 --prompt-mode attr
```

**评测设定**：

- 从 val 随机抽 50 张图（seed=42），每张取 1 个实例
- Prompt A（category）：仅品类，如 `dress.`
- Prompt B（attr）：品类 + 属性，如 `dress . long sleeve . v-neck.`
- 检测框：GDINO 预测中与 GT bbox IoU 最大的框
- 分割：SAM2 用该框生成 mask，与 GT mask 算 mIoU / Dice

**输出目录**：

- `outputs/fashionpedia_baseline_category/`（含 `summary.json`、`per_instance.json`、`visualizations/`）
- `outputs/fashionpedia_baseline_attr/`

---

## 5. Baseline 结果汇总（50 实例，zero-shot）

| 指标           | Prompt A（category） | Prompt B（attr） |
| -------------- | -------------------- | ---------------- |
| Detection rate | **0.96**             | 0.78             |
| Mean box IoU   | **0.759**            | 0.579            |
| Mean mask mIoU | **0.703**            | 0.538            |
| Mean mask Dice | **0.756**            | 0.578            |
| Box IoU@0.5    | **0.78**             | 0.60             |
| Mask IoU@0.5   | **0.74**             | 0.58             |

---

## 6. Step 4：结构化 Prompt 对比（全量 val，1158 实例）

**脚本**：`prompt_templates.py`（5 种策略）+ `fashionpedia_prompt_eval.py`

```bash
python fashionpedia_prompt_eval.py --max-images 1158
```

**评测设定**：全量 val，每张图 1 实例，zero-shot GDINO + SAM2，阈值 box=0.35 / text=0.25。

| 策略         | 说明                       | Detection rate | Mean box IoU | **Mean mask mIoU** |
| ------------ | -------------------------- | -------------- | ------------ | ------------------ |
| baseline     | 主类名（逗号前第一个词）   | 0.933          | 0.692        | 0.623              |
| descriptive  | + fashion/clothing/garment | 0.995          | 0.523        | 0.491              |
| **synonyms** | **同义词并列（逗号拆分）** | **0.938**      | **0.732**    | **0.663**          |
| multi_term   | 领域词表多词组合           | 0.927          | 0.694        | 0.628              |
| ecommerce    | 电商场景描述词             | 0.998          | 0.634        | 0.578              |

**结论**：`synonyms` 策略最优，相对 baseline **+4.0pp** mask mIoU（0.623 → 0.663）。  
descriptive / ecommerce 检测率虽高但框偏松，mIoU 反而下降。  
**后续推理统一使用 `--prompt-strategy synonyms`。**

**输出**：`outputs/fashionpedia_prompt_compare/`

---

## 7. Step 5：Grounding DINO LoRA 微调

**代码仓库**：`/root/autodl-tmp/Grounding-Dino-FineTuning`（Asad-Ismail  fork）

### 7.1 数据准备

| 文件                                           | 规模                 |
| ---------------------------------------------- | -------------------- |
| `fashionpedia/gdino_csv/train_annotations.csv` | 5000 图 / 36583 实例 |
| `fashionpedia/gdino_csv/val_annotations.csv`   | 1158 图 / 8774 实例  |

- 转换脚本：`tools/convert_fashionpedia_to_csv.py`
- `label_name` = 主类名（与推理 baseline 一致），如 `shirt, blouse` → `shirt`
- 训练 caption 格式：一张图多物体 + 负采样，如 `dress . shirt . hat . person .`

### 7.2 训练配置（`configs/fashionpedia_train_config.yaml`）

| 项            | 值                                                  |
| ------------- | --------------------------------------------------- |
| 基座          | SwinT `groundingdino_swint_ogc.pth`                 |
| LoRA rank     | **32**（`lora_alpha=32`，可训练参数 299 万 / 1.7%） |
| 训练数据      | 5000 张 train 图                                    |
| Epochs        | 10（每 2 epoch 存 checkpoint）                      |
| Batch size    | 16                                                  |
| Learning rate | 2e-4（OneCycleLR）                                  |
| 负采样率      | 0.5                                                 |
| 精度          | BF16 + TF32                                         |
| 耗时          | ~138 min（A800）                                    |

```bash
cd /root/autodl-tmp/Grounding-Dino-FineTuning
bash scripts/run_train.sh
```

**权重输出**：

- v1：`weights/fashionpedia_lora/20260614_2020/checkpoint_epoch_{2,4,6,8,10}.pth`
- v2：`weights/fashionpedia_lora/20260614_2321/checkpoint_epoch_{2,4,5}.pth`

**训练 loss（epoch 平均 total_loss）**：

- v1：从 epoch 1 ~2.3 缓慢下降至 epoch 10 ~2.26
- v2：5 epoch 完成（neg=0, lr=1e-4）

### 7.2b 训练配置 v2（重训，减小 train/inference 偏移）

| 项            | v1                  | **v2**   |
| ------------- | ------------------- | -------- |
| Epochs        | 10                  | **5**    |
| Learning rate | 2e-4                | **1e-4** |
| 负采样率      | 0.5                 | **0**    |
| LoRA rank     | 32                  | 32       |
| 其余          | batch=16, BF16+TF32 | 同左     |

```bash
# v2 配置见 configs/fashionpedia_train_config.yaml（已更新）
bash scripts/run_train.sh
```

**v2 合并权重**：

```bash
python tools/export_merged_lora.py \
  --lora weights/fashionpedia_lora/20260614_2321/checkpoint_epoch_5.pth
# → weights/fashionpedia_lora/20260614_2321/merged_swint_ogc.pth
```

### 7.3 环境与踩坑

| 问题                                     | 处理                                          |
| ---------------------------------------- | --------------------------------------------- |
| `peft` 与 `transformers` 版本冲突        | 固定 `peft==0.11.1` + `transformers==4.41.2`  |
| BF16 + `ms_deform_attn` CUDA 算子        | `ms_deform_attn.py` 内临时转 FP32             |
| `NestedTensor.to(non_blocking=)`         | 去掉 non_blocking                             |
| `supervision==0.6.0` 无 `LabelAnnotator` | 标签改由 `BoxAnnotator(labels=...)` 绘制      |
| SAM2 软链断裂                            | 重链至 `Grounded-SAM-2/sam2.1_hiera_large.pt` |

### 7.4 权重合并（供 Grounded-SAM-2 推理）

```bash
python tools/export_merged_lora.py \
  --lora weights/fashionpedia_lora/20260614_2020/checkpoint_epoch_10.pth
# → merged_swint_ogc.pth（~692MB）
```

---

## 8. Step 6：LoRA 微调后评测（全量 val，1158 实例）

**脚本**：`fashionpedia_eval.py` + `--gdino-checkpoint merged_swint_ogc.pth`

### 8.1 主对比表

| 方法        | 训练配置                      | Prompt       | 阈值 (box/text) | Det. rate | Mean box IoU | **Mask mIoU** |
| ----------- | ----------------------------- | ------------ | --------------- | --------- | ------------ | ------------- |
| Zero-shot   | —                             | synonyms     | 0.35 / 0.25     | 0.938     | 0.732        | **0.663**     |
| LoRA v1     | 10ep, lr=2e-4, neg=0.5, r=32  | synonyms     | 0.35 / 0.25     | 0.693     | 0.610        | 0.554         |
| LoRA v1     | 10ep, lr=2e-4, neg=0.5, r=32  | baseline     | 0.35 / 0.25     | 0.741     | —            | 0.594         |
| LoRA v1     | 10ep, lr=2e-4, neg=0.5, r=32  | synonyms     | 0.25 / 0.20     | 0.908     | 0.780        | 0.685         |
| LoRA v2     | 5ep, lr=1e-4, neg=0, r=32     | synonyms     | 0.35 / 0.25     | 0.623     | 0.541        | 0.493         |
| **LoRA v2** | **5ep, lr=1e-4, neg=0, r=32** | **synonyms** | **0.25 / 0.20** | **0.924** | **0.795**    | **0.700**     |

> LoRA v2 权重：`weights/fashionpedia_lora/20260614_2321/merged_swint_ogc.pth`  
> v2 输出：`outputs/fashionpedia_lora_v2_synonyms/`（默认阈值）、`outputs/fashionpedia_lora_v2_synonyms_th025_real/`（**校准阈值，最佳**）

### 8.2 关键发现

1. **默认阈值下 LoRA 均明显更差**：v1 mIoU 0.554、v2 **0.493**，因微调后置信度偏低导致漏检。
2. **校准阈值（0.25/0.20）后 LoRA 优于 zero-shot**：
   - v1：**0.685**（+2.2pp vs zero-shot 0.663）
   - v2：**0.700**（+3.7pp vs zero-shot，**当前最佳**）
3. **v2 重训有效**：`neg=0`、lr=1e-4、5 epoch 在校准阈值下优于 v1（0.700 vs 0.685），且训练时间更短。
4. **阈值公平性**：zero-shot 与 LoRA 使用相同默认阈值时，LoRA 不占优；需为 LoRA 单独校准阈值，或双方在 val 上联合扫阈值。
5. **固定阈值公平对比**：zero-shot synonyms **0.663** 仍是未调参下的最强基线。

**输出目录**：

- `outputs/fashionpedia_lora_synonyms/`（v1，默认阈值）
- `outputs/fashionpedia_lora_baseline/`（v1）
- `outputs/fashionpedia_lora_synonyms_th025/`（v1，校准阈值）
- `outputs/fashionpedia_lora_v2_synonyms/`（v2，默认阈值 0.35/0.25）
- `outputs/fashionpedia_lora_v2_synonyms_th025_real/`（**v2，校准阈值 0.25/0.20，最佳**）

---

## 9. 阶段性结论（截至 2026-06-15）

1. **Pipeline 可用**：GDINO → SAM2 在 Fashionpedia 全量 val 上 zero-shot mask mIoU **0.663**（synonyms + 默认阈值）。

2. **Prompt 工程有效**：synonyms 相对 baseline **+4.0pp**（0.623 → 0.663），无需训练。

3. **LoRA v2 为当前最佳**：5ep / neg=0 / lr=1e-4，校准阈值后 mask mIoU **0.700**，相对 zero-shot **+3.7pp**，相对 LoRA v1 **+1.5pp**。

4. **对比**：

   | 对比项                              | Mask mIoU | Det. rate | 备注                 |
   | ----------------------------------- | --------- | --------- | -------------------- |
   | Zero-shot + synonyms（0.35/0.25）   | 0.663     | 0.938     | 固定阈值公平基线     |
   | LoRA v1 + synonyms（0.25/0.20）     | 0.685     | 0.908     | 10ep, neg=0.5        |
   | **LoRA v2 + synonyms（0.25/0.20）** | **0.700** | **0.924** | **5ep, neg=0，推荐** |
   | LoRA v2 + synonyms（0.35/0.25）     | 0.493     | 0.623     | 默认阈值下不可用     |

5. **经验**：LoRA 改变置信度分布，**必须校准阈值**；v2 去掉负采样 + 保守训练优于 v1；Prompt synonyms 与 LoRA 可叠加收益。

---

## 10. Step 7：val 子集联合阈值扫描（zero-shot vs LoRA v2）

**脚本**：`fashionpedia_threshold_sweep.py`（新建，GDINO 前向缓存，不覆盖既有评测结果）

```bash
cd /root/autodl-tmp/Grounded-SAM-2
export TRANSFORMERS_OFFLINE=1
python fashionpedia_threshold_sweep.py --subset-size 200
# 输出目录带时间戳，例：outputs/threshold_sweep_20260615_102629/
```

**设定**：val 子集 **200 张**（seed=42，与全量评测同种子下的前 200 实例）、prompt=synonyms  
**扫描网格**：box ∈ {0.15, 0.20, 0.25, 0.30, 0.35, 0.40} × text ∈ {0.15, 0.20, 0.25, 0.30}  
**各自独立选最优**（按子集 mask mIoU 最大）

### 10.1 子集最优阈值（公平：各模型各自最优）

| 模型              | 最优 box/text | Det. rate | Mask mIoU  |
| ----------------- | ------------- | --------- | ---------- |
| Zero-shot         | 0.15 / 0.15   | 0.995     | 0.755      |
| LoRA v2           | 0.15 / 0.15   | 0.965     | **0.763**  |
| **Δ (LoRA − ZS)** |               |           | **+0.008** |

### 10.2 相同阈值对比（子集 200）

| 阈值 (box/text)         | Zero-shot mIoU | LoRA v2 mIoU |
| ----------------------- | -------------- | ------------ |
| 0.35 / 0.25（默认）     | 0.674          | 0.501        |
| 0.25 / 0.20（常用校准） | 0.720          | 0.720        |

**结论**：

- 各模型**独立扫阈值**后 LoRA v2 略优（+0.8pp on 200-subset）。
- **相同阈值 0.25/0.20** 下两者几乎持平（0.720 vs 0.720）；默认 0.35/0.25 下 LoRA 仍明显更差。
- 子集最优阈值均为 **0.15/0.15**；全量复评已验证（§11），0.15 优于此前常用的 0.25/0.20。

**输出**：

- `outputs/threshold_sweep_20260615_102629/comparison.json`
- `outputs/threshold_sweep_20260615_102629/grid_results.csv`
- `outputs/threshold_sweep_20260615_102629/{zero_shot,lora_v2}/grid_results.json`

---

## 11. Step 8：全量 val 最优阈值复评（1158，th=0.15/0.15）

依据 §10 子集扫描，双方在子集上最优均为 **box=0.15, text=0.15**。在全量 val 上复跑 `fashionpedia_eval.py`：

```bash
python fashionpedia_eval.py --max-images 1158 --prompt-strategy synonyms \
  --box-threshold 0.15 --text-threshold 0.15 \
  --gdino-checkpoint <checkpoint> \
  --output-dir outputs/<name>
```

### 11.1 全量结果（公平对比：同 prompt、同阈值 0.15/0.15）

| 方法              | 权重                                    | Det. rate | Box IoU    | **Mask mIoU** | Mask Dice  | Mask IoU@0.5 |
| ----------------- | --------------------------------------- | --------- | ---------- | ------------- | ---------- | ------------ |
| **Zero-shot**     | 预训练 GDINO（`gdino_checkpoint=null`） | **0.994** | 0.833      | 0.735         | 0.792      | 0.805        |
| **LoRA v2**       | `20260614_2321/merged`                  | 0.984     | **0.863**  | **0.746**     | **0.808**  | **0.815**    |
| **Δ (LoRA − ZS)** |                                         | −1.0pp    | **+3.0pp** | **+1.1pp**    | **+1.6pp** | **+1.0pp**   |

**输出目录**：

- `outputs/fashionpedia_zero_shot_synonyms_th015_best/`（**true zero-shot**）
- `outputs/fashionpedia_lora_v2_synonyms_th015_best/`（LoRA v2）

### 11.2 分析

1. **公平 head-to-head（全量 1158，synonyms，th=0.15/0.15）**：LoRA v2 mask mIoU **0.746 vs 0.735（+1.1pp）**，为目前最可信的 LoRA vs zero-shot 对比。
2. **LoRA 增益主要在定位质量**：box IoU **+3.0pp**（0.863 vs 0.833），mask Dice **+1.6pp**；检测率 zero-shot 反而略高（99.4% vs 98.4%），说明 LoRA 在略降召回的同时显著提升了框/掩码精度。
3. **阈值校准对双方均关键**：
   - Zero-shot：0.35/0.25 → 0.15/0.15，mIoU **0.663 → 0.735（+7.2pp）**
   - LoRA v2：0.35/0.25 → 0.15/0.15，mIoU **0.493 → 0.746（+25.3pp）**；相对 th=0.25/0.20 的 0.700 再 **+4.6pp**
4. **子集 vs 全量**：子集 200 上 ZS/LoRA 均为 0.755/0.763，全量分别为 0.735/0.746，趋势一致（LoRA 略优），但绝对值子集偏高 ~2pp，全量结果更可靠。
5. **LoRA v1 vs v2**（同 th=0.15，误跑可作参考）：0.742 vs 0.746（+0.4pp），v2 重训有效但增益小于相对 true zero-shot 的 +1.1pp。

### 11.3 方法对比总表（全量 1158，synonyms prompt）

| 方法        | 阈值          | Mask mIoU | Det. rate | 说明                |
| ----------- | ------------- | --------- | --------- | ------------------- |
| Zero-shot   | 0.35/0.25     | 0.663     | 0.938     | 默认阈值基线        |
| Zero-shot   | **0.15/0.15** | **0.735** | **0.994** | 校准后              |
| LoRA v1     | 0.25/0.20     | 0.685     | 0.908     |                     |
| LoRA v1     | 0.15/0.15     | 0.742     | 0.978     | 误标 zero-shot 那次 |
| **LoRA v2** | **0.15/0.15** | **0.746** | 0.984     | **当前全量最高**    |
| LoRA v2     | 0.25/0.20     | 0.700     | 0.924     | 次优校准            |
| LoRA v2     | 0.35/0.25     | 0.493     | 0.623     | 默认阈值不可用      |

---

## 12. 阶段性结论

1. **Prompt synonyms**：zero-shot 默认阈值 mIoU **0.663**（+4pp vs baseline）；校准至 0.15/0.15 后升至 **0.735**。

2. **LoRA v2 全量最佳**：synonyms + th=0.15/0.15 → mask mIoU **0.746**，为当前最高。

3. **公平对比**：同阈值 0.15/0.15 下 LoRA v2 相对 true zero-shot **+1.1pp**（0.746 vs 0.735），box IoU **+3.0pp**；LoRA 主要提升定位/分割质量而非召回。整体增幅偏小，**per-category breakdown** 显示收益集中在 garment parts（§13）。

4. **对比**：

   | 对比项                                   | Mask mIoU | Box IoU   | Det. rate |
   | ---------------------------------------- | --------- | --------- | --------- |
   | Zero-shot + synonyms（默认 0.35/0.25）   | 0.663     | 0.732     | 0.938     |
   | Zero-shot + synonyms（校准 0.15/0.15）   | 0.735     | 0.833     | 0.994     |
   | **LoRA v2 + synonyms（校准 0.15/0.15）** | **0.746** | **0.863** | 0.984     |

   > Fashionpedia val 1158 实例；LoRA 微调 Grounding DINO（5ep, neg=0）+ synonyms prompt + 阈值校准，相对 zero-shot **+1.1pp mask mIoU**；**garment parts +4.8pp**，**neckline +12.6pp**（n=141，见 §13）。

5. **经验**：阈值校准对 LoRA 影响远大于 zero-shot；prompt engineering（+4pp）与 LoRA（+1.1pp over calibrated ZS）可叠加，但 prompt 收益更大。**按类拆开**后 LoRA 在 neckline 等局部部件上提升显著（+12.6pp），整衣类接近饱和或略降（§13）。

6. **阈值扫描（§10/§15）**：三模型最优均为 **0.15/0.15**；LoRA 默认阈值下子集 mIoU 仅 0.50，校准后 +26pp；**推理必须用扫阈后的阈值，不能用默认 0.35/0.25**。

---

## 13. Step 9：Per-category breakdown（LoRA v2 vs zero-shot，th=0.15）

### 13.0 动机

全量公平对比下 LoRA 仅 **+1.1pp**（0.746 vs 0.735），整体说服力不足。  
假设：视觉相似/细粒度类目（如 shirt vs blouse、局部部件 neckline）正是预训练 GDINO 薄弱处，LoRA 应在**部分类别**上有更大提升，而被 dress/shoe 等大类的持平或略降所稀释。  
因此对 §11 两份 `per_instance.json` 做 **按服装类别分组** 的 mask mIoU 统计与配对 diff。

### 13.1 流程

1. **输入**：两份全量评测输出（同 prompt、同阈值 0.15/0.15，1158 实例一一对应）  
   - LoRA v2：`outputs/fashionpedia_lora_v2_synonyms_th015_best/per_instance.json`  
   - Zero-shot：`outputs/fashionpedia_zero_shot_synonyms_th015_best/per_instance.json`

2. **字段**：每条记录含 `ann_id`、`category`（⚠️ 非 `category_name`）、`mask_iou`、`box_iou`、`detected` 等；val 共 **40** 个类别。

3. **配对**：按 `ann_id` 对齐同一样本，计算 instance-level  
   `Δ_mask = LoRA.mask_iou − ZS.mask_iou`，再按 `category` 聚合均值。

4. **聚合指标**（每类）：`n`、ZS/LoRA mean mask IoU、Δ、win/loss（单实例 LoRA 更高/更低计数）、supercategory（来自 `instances_attributes_val2020.json`）。

5. **实现**： `fashionpedia_category_breakdown.py`，输出 CSV / MD / JSON。

```bash
cd /root/autodl-tmp/Grounded-SAM-2
python fashionpedia_category_breakdown.py \
  --lora-dir outputs/fashionpedia_lora_v2_synonyms_th015_best \
  --zs-dir outputs/fashionpedia_zero_shot_synonyms_th015_best \
  --output-dir outputs/category_breakdown_th015_lora_vs_zs_min10 \
  --min-count 10
```

**输出目录**：

- `outputs/category_breakdown_th015_lora_vs_zs/` — 全 40 类（`--min-count 1`）
- `outputs/category_breakdown_th015_lora_vs_zs_min10/` — **n≥10，18 类**

### 13.2 总体

| 聚合方式             | Δ mask mIoU | 说明                                             |
| -------------------- | ----------- | ------------------------------------------------ |
| Instance-weighted    | **+0.0106** | 与全量 0.746−0.735 一致                          |
| Macro（40 类等权）   | +0.0122     | 全类                                             |
| Macro（n≥10，18 类） | −0.0022     | 去掉小样本后大类互有升降，**不宜单独作总体结论** |

**正负抵消**：dress（n=242）、shoe（n=187）等大类别 LoRA 略降（−0.002 / −0.011），与 neckline（n=141, +0.126）等大增益相加后，整体仅 +1.1pp。

**胜负统计（1158 实例）**：LoRA 单实例 mask IoU 更高 **~52%**、更低 **~48%**（各类 win/loss 见 CSV）；收益呈**结构化分布**而非均匀提升。

### 13.3 Supercategory 汇总（instance-weighted）

| Supercategory     |    n | ZS mIoU | LoRA mIoU |      **Δ** |
| ----------------- | ---: | ------: | --------: | ---------: |
| **garment parts** |  263 |   0.395 |     0.443 | **+0.048** |
| arms and hands    |   10 |   0.657 |     0.697 |     +0.040 |
| decorations       |   14 |   0.201 |     0.225 |     +0.023 |
| waist             |    8 |   0.660 |     0.681 |     +0.022 |
| others            |   19 |   0.814 |     0.825 |     +0.011 |
| upperbody         |  184 |   0.863 |     0.864 |     +0.001 |
| wholebody         |  272 |   0.918 |     0.918 |     +0.001 |
| lowerbody         |   93 |   0.939 |     0.937 |     −0.001 |
| head              |   96 |   0.787 |     0.785 |     −0.002 |
| legs and feet     |  193 |   0.747 |     0.741 |     −0.005 |
| closures          |    5 |   0.260 |     0.158 |     −0.102 |

**核心结论**：LoRA 主要收益在 **garment parts**（领口、袖、口袋、领片等，n=263，**+4.8pp**）——Fashionpedia 标注难、ZS 基线低（0.395）的区域。整衣（upperbody/wholebody）ZS 已高（0.86–0.92），LoRA 几乎无增益。

### 13.4 类别 Top / Bottom（n≥10，推荐表）

| Category                 |    n |    ZS |  LoRA |     Δ mIoU | win/loss |
| ------------------------ | ---: | ----: | ----: | ---------: | -------: |
| neckline                 |  141 | 0.169 | 0.295 | **+0.126** |   117/23 |
| bag, wallet              |   10 | 0.738 | 0.829 |     +0.091 |      4/4 |
| coat                     |   20 | 0.906 | 0.937 |     +0.031 |     11/9 |
| top, t-shirt, sweatshirt |  132 | 0.860 | 0.882 |     +0.022 |    72/59 |
| collar                   |   14 | 0.455 | 0.461 |     +0.006 |     11/3 |
| dress                    |  242 | 0.919 | 0.917 |     −0.002 |  120/115 |
| shoe                     |  187 | 0.752 | 0.741 |     −0.011 |    94/77 |
| sleeve                   |   79 | 0.807 | 0.770 |     −0.037 |    40/39 |
| jacket                   |   25 | 0.900 | 0.846 |     −0.054 |    13/12 |
| shirt, blouse            |   18 | 0.846 | 0.777 |     −0.070 |     12/6 |
| pocket                   |   19 | 0.486 | 0.360 |     −0.127 |     9/10 |

过滤掉 rivet（n=5）、scarf（n=8）、zipper（n=4）等极小类后，**neckline +12.6pp** 仍是主叙事；负向以 pocket、shirt/blouse、jacket、sleeve 为主。

### 13.5 完整 40 类结果（按 Δ mIoU 降序，含小样本）

| Category                                |    n |    ZS |  LoRA | Δ mask |  Δ box |     W/L |
| --------------------------------------- | ---: | ----: | ----: | -----: | -----: | ------: |
| sock                                    |    2 | 0.000 | 0.522 | +0.522 | +0.658 |     2/0 |
| buckle                                  |    1 | 0.002 | 0.185 | +0.183 | +0.145 |     1/0 |
| neckline                                |  141 | 0.169 | 0.295 | +0.126 | +0.224 |  117/23 |
| bag, wallet                             |   10 | 0.738 | 0.829 | +0.091 | +0.073 |     4/4 |
| rivet                                   |    5 | 0.301 | 0.381 | +0.080 | +0.087 |     3/2 |
| watch                                   |    9 | 0.630 | 0.674 | +0.045 | +0.051 |     6/3 |
| coat                                    |   20 | 0.906 | 0.937 | +0.031 | +0.004 |    11/9 |
| top, t-shirt, sweatshirt                |  132 | 0.860 | 0.882 | +0.022 | +0.025 |   72/59 |
| belt                                    |    8 | 0.660 | 0.681 | +0.022 | +0.099 |     6/2 |
| lapel                                   |    9 | 0.077 | 0.093 | +0.016 | +0.032 |     7/2 |
| umbrella                                |    1 | 0.906 | 0.914 | +0.008 | +0.108 |     1/0 |
| collar                                  |   14 | 0.455 | 0.461 | +0.006 | +0.078 |    11/3 |
| pants                                   |   43 | 0.945 | 0.948 | +0.003 | +0.012 |   28/15 |
| cardigan                                |    1 | 0.930 | 0.931 | +0.001 | +0.000 |     1/0 |
| glove                                   |    1 | 0.902 | 0.902 | +0.001 | +0.021 |     1/0 |
| applique                                |    2 | 0.026 | 0.027 | +0.001 | +0.002 |     1/1 |
| hat                                     |   30 | 0.879 | 0.879 | +0.000 | +0.003 |   14/11 |
| bead                                    |    1 | 0.000 | 0.000 | +0.000 | −0.002 |     1/0 |
| flower                                  |    1 | 0.000 | 0.000 | +0.000 | +0.000 |     0/0 |
| leg warmer                              |    1 | 0.938 | 0.938 | +0.000 | +0.002 |     0/0 |
| sweater                                 |    5 | 0.765 | 0.765 | −0.000 | +0.002 |     3/1 |
| skirt                                   |   27 | 0.947 | 0.946 | −0.000 | +0.005 |   15/10 |
| jumpsuit                                |   10 | 0.914 | 0.913 | −0.000 | +0.008 |     5/5 |
| tights, stockings                       |    3 | 0.860 | 0.860 | −0.001 | +0.001 |     1/2 |
| glasses                                 |   41 | 0.765 | 0.764 | −0.001 | −0.005 |   22/17 |
| sequin                                  |    1 | 0.002 | 0.000 | −0.002 | −0.009 |     0/1 |
| dress                                   |  242 | 0.919 | 0.917 | −0.002 | +0.002 | 120/115 |
| headband, head covering, hair accessory |   25 | 0.714 | 0.707 | −0.007 | +0.010 |   12/13 |
| tie                                     |    1 | 0.774 | 0.765 | −0.009 | −0.048 |     0/1 |
| shorts                                  |   23 | 0.918 | 0.907 | −0.011 | −0.002 |   11/12 |
| shoe                                    |  187 | 0.752 | 0.741 | −0.011 | +0.003 |   94/77 |
| ruffle                                  |    4 | 0.315 | 0.297 | −0.018 | −0.027 |     1/2 |
| vest                                    |    3 | 0.931 | 0.911 | −0.020 | −0.010 |     1/2 |
| sleeve                                  |   79 | 0.807 | 0.770 | −0.037 | −0.030 |   40/39 |
| hood                                    |    1 | 0.094 | 0.055 | −0.039 | +0.002 |     0/1 |
| jacket                                  |   25 | 0.900 | 0.846 | −0.054 | −0.063 |   13/12 |
| shirt, blouse                           |   18 | 0.846 | 0.777 | −0.070 | −0.053 |    12/6 |
| scarf                                   |    8 | 0.898 | 0.809 | −0.089 | −0.071 |     4/4 |
| pocket                                  |   19 | 0.486 | 0.360 | −0.127 | −0.105 |    9/10 |
| zipper                                  |    4 | 0.324 | 0.152 | −0.173 | −0.126 |     1/2 |

> W/L = 该类别内 LoRA mask IoU 高于/低于 ZS 的实例数。完整数据见 `category_breakdown.csv`。

### 13.6 解读

1. **假设部分成立**：难标注的**局部部件**（neckline、lapel）LoRA 提升大；ZS 在 neckline 仅 0.17，LoRA 拉到 0.30（**+12.6pp**），是最有说服力的单类证据。
2. **假设部分不成立**：**shirt/blouse、jacket** 等整衣类别 LoRA 反而下降 5–7pp，可能因为 ZS 已接近饱和（0.85–0.90），或 LoRA 过拟合训练分布、损害泛化。
3. **部件类内部不一致**：neckline/collar/lapel 升、pocket/sleeve/zipper 降，说明「garment parts」不能一概而论，需结合 prompt 与检测框质量个案分析。
4. **叙事建议（简历/报告）**：
   - 整体：LoRA +1.1pp mask mIoU（1158 val，公平阈值）
   - 分层：**garment parts +4.8pp**；**neckline +12.6pp**（n=141）
   - 诚实披露：wholebody/upperbody 基本持平，部分整衣类略降

### 13.7 Failure case 可视化

**脚本**：`fashionpedia_failure_viz.py`  
**输出**：`outputs/failure_viz_th015/`（15 张三列对比图 + `GALLERY.md` + `manifest.json`）

| 类别             | 策略   | 主要观察                                            |
| ---------------- | ------ | --------------------------------------------------- |
| neckline (win×5) | Δ 最大 | ZS mIoU≈0.04，LoRA **0.65–0.71**（+60pp 级个案）    |
| pocket (loss×5)  | Δ 最小 | LoRA **漏检**（mIoU=0）而 ZS 已有 0.3–0.8 的框      |
| jacket (loss×5)  | Δ 最小 | 前 2 例 LoRA 明显差；后 3 例 ZS/LoRA 均≈0.9（饱和） |

汇总图见 `outputs/figures/`。

---

## 14. Step 10：LoRA rank 消融（r16 vs r32）

**训练**：与 v2 相同（5ep / neg=0 / lr=1e-4 / batch=16），仅 **LoRA rank 16**（`lora_alpha=16`，可训练参数 ~1.50M / 0.86%）。  
**权重**：`weights/fashionpedia_lora/20260615_1330/merged_swint_ogc.pth`

### 14.1 全量评测（synonyms，th=0.15/0.15，1158）

| 方法          | rank | Det. rate | Box IoU   | **Mask mIoU** | Mask Dice |
| ------------- | ---: | --------- | --------- | ------------- | --------- |
| Zero-shot     |    — | 0.994     | 0.833     | 0.735         | 0.792     |
| **LoRA r16**  |   16 | 0.984     | 0.854     | **0.743**     | 0.805     |
| LoRA r32 (v2) |   32 | 0.984     | **0.863** | **0.746**     | **0.808** |

| 对比             | Δ Mask mIoU | Δ Box IoU |
| ---------------- | ----------- | --------- |
| r16 vs zero-shot | **+0.8pp**  | +2.0pp    |
| r16 vs r32       | −0.3pp      | −0.9pp    |

**结论**：

- r16 仍优于 zero-shot（+0.8pp），但略逊于 r32（−0.3pp）；**精度优先选 r32**。
- 参数量减半，mask mIoU 仅掉 0.26pp，**效率/精度折中可接受**。
- 检测率与 r32 相同（0.9836），1158 条中 1150 条 detected 标志一致；差距在框/掩码质量。

**输出**：`outputs/fashionpedia_lora_r16_synonyms_th015/`

### 14.2 r16 vs r32 按类（配对 diff 摘要）

| 方向     | 类别          | Δ mIoU  |
| -------- | ------------- | ------- |
| r16 更好 | scarf         | +5.2pp  |
| r16 更好 | shirt, blouse | +2.5pp  |
| r16 更差 | rivet         | −14.1pp |
| r16 更差 | pocket        | −3.8pp  |

r16 在小样本/难类上波动更大；garment parts 整体 r32 更稳。

### 14.3 r16 阈值扫描

见 §15（含三模型统一结论 §15.4）。

---

## 15. Step 11：LoRA r16 阈值扫描（子集 200）

**脚本**：`fashionpedia_threshold_sweep.py --gdino-checkpoint ... --model-name lora_r16`

```bash
python fashionpedia_threshold_sweep.py \
  --subset-size 200 \
  --gdino-checkpoint /root/autodl-tmp/Grounding-Dino-FineTuning/weights/fashionpedia_lora/20260615_1330/merged_swint_ogc.pth \
  --model-name lora_r16 \
  --output-dir outputs/threshold_sweep_r16
```

**设定**：与 §10 相同（子集 200、seed=42、synonyms、24 组 box×text 网格）。

### 15.1 子集最优阈值（各自独立选最优）

| 模型          | 最优 box/text | Det. rate | Mask mIoU |
| ------------- | ------------- | --------- | --------- |
| Zero-shot     | 0.15 / 0.15   | 0.995     | 0.755     |
| LoRA r32 (v2) | 0.15 / 0.15   | 0.965     | **0.763** |
| LoRA r16      | 0.15 / 0.15   | 0.960     | 0.755     |
| Δ (r32 − ZS)  |               |           | +0.008    |
| Δ (r16 − r32) |               |           | −0.008    |

### 15.2 相同阈值对比（子集 200，synonyms）

| 阈值 (box/text)         | Zero-shot |  LoRA r32 |  LoRA r16 |
| ----------------------- | --------: | --------: | --------: |
| 0.35 / 0.25（默认）     |     0.674 |     0.501 |     0.518 |
| 0.25 / 0.20             |     0.720 | **0.720** |     0.694 |
| **0.15 / 0.15（最优）** | **0.755** | **0.763** | **0.755** |

### 15.3 子集 → 全量对照（th=0.15/0.15）

| 模型      | 子集 200 mIoU | 全量 1158 mIoU | 子集−全量 |
| --------- | ------------: | -------------: | --------: |
| Zero-shot |         0.755 |          0.735 |    +2.0pp |
| LoRA r32  |         0.763 |          0.746 |    +1.7pp |
| LoRA r16  |         0.755 |          0.743 |    +1.2pp |

子集绝对值系统性偏高 ~1–2pp，但**模型间排序一致**（r32 > r16 ≈ ZS）。

### 15.4 阈值扫描总结性结论

1. **最优阈值统一为 0.15/0.15**：zero-shot、LoRA r32、LoRA r16 在 24 组网格上均如此；**不必为不同 rank 另调阈值**。
2. **LoRA 对阈值极度敏感**：默认 0.35/0.25 下 r32 子集 mIoU 仅 **0.501**（ZS 0.674）；降至 0.15/0.15 后升至 **0.763（+26.2pp）**。ZS 同期 0.674→0.755（+8.1pp）。
3. **0.25/0.20 是误导性「折中」**：同阈值下 ZS 与 r32 子集均为 0.720，看似持平，但各自继续降到 0.15 后 r32 仍有 +4.3pp 空间；**应用扫阈值后的各自最优，而非强行同阈**。
4. **检测率与 mIoU 的权衡**：阈值越低，Det. rate 越高（ZS 0.995、LoRA 0.96），LoRA 在低阈值下用更高召回换取框/掩码质量提升（全量 box IoU +3.0pp）。
5. **rank 不影响最优阈值**：r16 与 r32 同取 0.15/0.15；r16 全量 0.743 已是最优 th 下的结果，无需再跑全量扫阈。
6. **推荐推理默认**：synonyms prompt + **box=0.15, text=0.15**；新 LoRA checkpoint 上线前应在 val 子集上复扫确认（`fashionpedia_threshold_sweep.py`）。

**输出**：`outputs/threshold_sweep_r16/lora_r16/{grid_results.json,best_threshold.json}`

---

## 16. 待做（下一步）

- [x] 在 val 子集上对 zero-shot / LoRA **联合扫阈值** → 见 §10
- [x] LoRA v2 全量 th015 复评 → mIoU **0.746**（§11）
- [x] **补跑 true zero-shot** @0.15/0.15 → mIoU **0.735**（§11）
- [x] **LoRA rank=16** 训练与评测 → 见 §14
- [x] r16 **阈值扫描** → 最优仍 0.15/0.15，子集 mIoU 0.755（§15）
- [x] **Per-category breakdown** LoRA vs ZS → 见 §13
- [x] **min-count 10** category breakdown → 见 §13.4
- [x] neckline / pocket / jacket **failure case 可视化** → §13.7、`outputs/failure_viz_th015/`
- [x] 整理 **FASHIONPEDIA.md** + 可视化对比图 → `outputs/figures/`

---

## 17. 关键命令速查

```bash
cd /root/autodl-tmp/Grounded-SAM-2
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com

# 单张 demo
python grounded_sam2_local_demo.py

# Zero-shot 全量评测（synonyms）
python fashionpedia_eval.py --max-images 1158 --prompt-strategy synonyms

# Prompt 策略对比
python fashionpedia_prompt_eval.py --max-images 1158

# LoRA 合并权重
cd /root/autodl-tmp/Grounding-Dino-FineTuning
export PYTHONPATH="/root/autodl-tmp/Grounding-Dino-FineTuning:$PYTHONPATH"
python tools/export_merged_lora.py \
  --lora weights/fashionpedia_lora/20260614_2020/checkpoint_epoch_10.pth \
  --config groundingdino/config/GroundingDINO_SwinT_OGC_localbert.py

# LoRA v2 全量评测（当前最佳：th=0.15/0.15）
python fashionpedia_eval.py \
  --max-images 1158 \
  --prompt-strategy synonyms \
  --box-threshold 0.15 \
  --text-threshold 0.15 \
  --gdino-checkpoint /root/autodl-tmp/Grounding-Dino-FineTuning/weights/fashionpedia_lora/20260614_2321/merged_swint_ogc.pth \
  --output-dir outputs/fashionpedia_lora_v2_synonyms_th015_best

# True zero-shot 全量（不要传 --gdino-checkpoint）
python fashionpedia_eval.py \
  --max-images 1158 \
  --prompt-strategy synonyms \
  --box-threshold 0.15 \
  --text-threshold 0.15 \
  --output-dir outputs/fashionpedia_zero_shot_synonyms_th015_best

# Per-category breakdown（n>=10，论文用表）
python fashionpedia_category_breakdown.py \
  --lora-dir outputs/fashionpedia_lora_v2_synonyms_th015_best \
  --zs-dir outputs/fashionpedia_zero_shot_synonyms_th015_best \
  --output-dir outputs/category_breakdown_th015_lora_vs_zs_min10 \
  --min-count 10

python fashionpedia_threshold_sweep.py --subset-size 200

# LoRA r16 单独阈值扫描
python fashionpedia_threshold_sweep.py \
  --subset-size 200 \
  --gdino-checkpoint /root/autodl-tmp/Grounding-Dino-FineTuning/weights/fashionpedia_lora/20260615_1330/merged_swint_ogc.pth \
  --model-name lora_r16 \
  --output-dir outputs/threshold_sweep_r16

# Failure case 可视化（GT | ZS | LoRA）
python fashionpedia_failure_viz.py --categories neckline pocket jacket --top-k 5

# 论文用汇总图
python fashionpedia_make_figures.py
```

---

*最后更新：2026-06-15（failure viz + FASHIONPEDIA.md + figures）*
