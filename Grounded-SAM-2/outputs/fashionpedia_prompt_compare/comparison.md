# Prompt Strategy Comparison

Instances: 1158 | seed: 42


| Strategy | Description | Det.Rate | Box mIoU | **Mask mIoU** | vs baseline |
|----------|-------------|----------|----------|---------------|-------------|
| synonyms | 类目全部同义词并列（逗号拆分） | 0.94 | 0.732 | **0.663** | +0.040 |
| multi_term | 品类相关多词组合（领域词表） | 0.93 | 0.694 | **0.628** | +0.005 |
| baseline | 主类名（逗号前第一个词） | 0.93 | 0.692 | **0.623** | — |
| ecommerce | 电商场景描述（product / apparel / fashion） | 1.00 | 0.634 | **0.578** | -0.045 |
| descriptive | 加结构描述词（fashion / clothing / garment） | 0.99 | 0.523 | **0.491** | -0.132 |

**Best:** `synonyms`
