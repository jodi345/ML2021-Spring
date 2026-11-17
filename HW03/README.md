# HW03 — Food-11 图像分类（CNN）

Kaggle: [ml2021spring-hw3](https://www.kaggle.com/competitions/ml2021spring-hw3) · metric: accuracy · 11 类，RGB 128×128

## 现状：没有排行榜可用

比赛当前是**受邀制**：`kaggle competitions download` / `submit` 都返回 403，kernel 也挂不上 `competition_sources`。所以这份作业改成**自测**：

- 数据用官方 notebook 里的 GDrive 链接（`food-11.zip`，1.1 GB）下载，不走 Kaggle 竞赛数据；
- 分数只在自建的 train / val / holdout 上报；
- **所有分数都不能和官方 simple baseline 44.862、strong baseline 82.138 直接比较**。官方那条线是 3347 张私测图上的 public 分，我们这里没有。

## 数据（官方改版 split）

| 目录 | 数量 | 用途 |
|---|---|---|
| `training/labeled` | 3080（280×11） | 有标签训练 |
| `validation` | 660（60×11） | 选模型 / 早停 |
| `training/unlabeled` | 6786 | Hard 档的伪标签池 |
| `testing` | 3347 | 无标签，官方私测集；本地无法评分 |

数据不进仓库（`.gitignore` 已排除 `*.zip`、`data/`），只在 Kaggle kernel 里下载到 `/tmp`。

## 自测协议

| 集合 | 来源 | 数量 | 用途 |
|---|---|---|---|
| train | labeled 的 90% | 2772 | 训练 |
| val | 官方 `validation` | 660 | 每个 epoch 评估，用于选模型 |
| holdout | labeled 的 10% | 308 | **只在训练结束后评估一次**，不参与任何选择 |

- 分层留出，每类 28 张，种子固定 `42`；后续所有版本沿用同一套划分。
- 主指标 accuracy，同时报 **macro F1 + per-class precision/recall + 混淆矩阵**。图像任务里 accuracy 高但在少数类上全崩很常见，只看一个数会漏掉。
- 代价：训练图比官方少 10%（2772 vs 3080），绝对分预计低 2–4 个点。**版本之间只看相对差异。**

## 结果

| 版本 | 说明 | val acc | holdout acc | macro F1 | 备注 |
|---|---|---|---|---|---|
| baseline | 官方 notebook 原样：3×(Conv+BN+ReLU+Pool) + 3×FC，Adam 3e-4，wd 1e-5，80 epoch | **0.5240**（第 35 epoch 峰值） | **0.5162** ±0.0558 | **0.5146** | kernel v2，CPU 跑 4.6 h |
| — | Medium（增强 / 架构） | — | — | — | 待做 |
| — | Hard（伪标签） | — | — | — | 待做 |

官方给出的线（仅供参考，非同一协议）：simple 44.862 / medium 52.807 / strong 82.138。

## 诊断：baseline 是典型过拟合，不是容量不够

| epoch | train acc | val acc | val loss |
|---|---|---|---|
| 1 | 0.188 | 0.096 | 2.803 |
| 10 | 0.701 | 0.461 | 1.722 |
| 20 | 0.991 | 0.496 | 1.931 |
| 30 | **1.000** | 0.510 | 2.216 |
| 35 | 1.000 | **0.524**（峰值） | 2.306 |
| 80 | 1.000 | 0.507 | 2.822 |

- **第 30 epoch 训练集就到 100%**，之后 val loss 从 1.72 一路涨到 2.82，val acc 反而回落。train − val ≈ 48 个点，按 AGENT.md 的阈值（>10 点）属于「先加正则 / 减容量，别再加宽」。
- 因此 **80 epoch 里后 45 个是白跑的**：早停能省一半时间（≈2.3 h CPU）。后续版本一律用 val 早停，patience = 10。
- per-class F1 从 0.18（类 2）到 0.76（类 10），差 4 倍。少数类不是「差一点」，是基本没学会——只看 accuracy 会把这个问题平均掉。
- 4.26 M 参数里 92% 在 `flatten(16384)→256` 那层全连接，这正是过拟合的主要来源，也解释了 E3（GAP）为什么值得单独跑一轮。

## 优化路线（按代价 / 收益排序，一轮只动一类变量）

| # | 改动 | 变量类别 | 粗估收益 | 备注 |
|---|---|---|---|---|
| E1 | 数据增强：`RandomResizedCrop(0.6–1.0)`、水平翻转、ColorJitter、RandAugment | 数据 | +3~8 | 强增强要配更长训练，80 epoch 下单独上可能反而掉 |
| E2 | 80 → 200 epoch + cosine / step 衰减 | 训练 | +2~4 | 和 E1 联动，最好一起看 |
| E3 | 全局平均池化替换 `flatten + 3×FC` | 架构 | 0~+3 | 参数 4.26 M → ~0.37 M（−91%），抗过拟合，但容量也降 |
| E4 | 水平翻转 TTA | 推理 | +0.5~1 | 几乎零成本，收尾再加 |
| E5 | 伪标签（阈值 ~0.9，跑 2–3 轮） | 半监督 | +8~15 | 6786 张 unlabeled 第一次真正被用上 |
| E6 | FixMatch（弱/强增强一致性 + 0.95 置信度） | 半监督 | +10~20 | 实现复杂，超参敏感 |
| E7 | 多轮 teacher → student 迭代 | 半监督 | 递减 | 放最后，收益不明确 |

三点注意：

1. **E1 和 E2 要一起判**。强增强 + 80 epoch 是欠训练组合，单独跑 E1 容易得出"增强没用"的错误结论。
2. **val 只有 660，σ ≈ 1.95**。小于 2 个点的差异当噪声；跨版本比较看 holdout 和 per-class F1，不要盯单个 val 数。
3. **真正的分水岭是 E5/E6**。官方 medium 52.8 → strong 82.1 那 30 个点，全部来自无标签数据；E1–E4 加起来大概 5–10 点。所以前面几轮是"把监督学习做到位"，不是终点。

**从 baseline 学到的一条**：每版都挂 val 早停（patience 10）。baseline 第 30 epoch 就 train acc = 1.0，后 45 个 epoch 全是负收益，白烧 2.3 小时 CPU。E1+E2 如果还在 CPU 上跑，预计 200 epoch 要 10 小时以上——这一版建议等 GPU，或者先把 epoch 砍到 60 + 早停。

## 文件

- [`HW03.ipynb`](HW03.ipynb) — 课程原始 notebook，保持原样不动
- [`HW03.pdf`](HW03.pdf) — 作业说明（三档要求、评分、截止时间）
- `README.md` — 本文：现状 + 协议 + 结果
- （后续）`HW03_solution.ipynb` — 单模型最终方案
- （后续）`SOLUTION.md` — 详解：诊断、负结果、为什么停

## 运行

Kaggle kernel：<https://www.kaggle.com/code/aviator22/ml2021spring-hw3-baseline>

配置：**GPU 开 / internet 开 / competition source 留空**。notebook 首次运行会：

1. 从 GDrive 下载 `food-11.zip`（失败自动改 Dropbox）→ 解压到 `/tmp/food-11`；
2. 断言四个目录的图片数量（3080 / 660 / 6786 / 3347），对不上直接报错；
3. 分层切出 holdout 后训练，存 `probs.npy` + `submission.csv`；
4. 最后在 holdout 上报 accuracy / macro F1 / 混淆矩阵。

## 与官方设置的差异（重要）

1. **没有排行榜**：所有数字是自测，不能当 LB 用，也不能和 44.862 / 82.138 对齐。
2. **训练集少 10%**：为了留出 holdout。
3. **数据来源是 GDrive 的官方 zip**，不是竞赛挂载；如果 GDrive 限流，需要用镜像数据集或手动上传。
4. **unlabeled 目录结构待确认**：官方代码用 `DatasetFolder` 读它，说明至少有一层类子目录；若实际是扁平目录，notebook 里已用 `FlatDataset` 兜底，但 Hard 阶段的伪标签写法要跟着调整。
5. 官方代码的 `loader=lambda x: Image.open(x)` 遇到灰度/CMYK 图会让 ToTensor 变成 1 通道；notebook 里加了 `.convert("RGB")`，这是唯一的语义改动。

## 待办

- [x] baseline 跑完，回填结果表（val 0.5240 / holdout 0.5162 / macro F1 0.5146）
- [x] S3 诊断（第一轮）：过拟合为主，train acc 第 30 epoch 到 1.0，val loss 随后回升
- [ ] 剩余诊断：类别分布、图像尺寸分布、重复图 / train-test 泄漏检查
- [ ] E1+E2：强增强 + 200 epoch + cosine + 早停（patience 10）
- [ ] E3：GAP 替换 flatten+FC，单独一轮
- [ ] Medium：数据增强 + 架构（ResNet 自建权重，不许预训练）
- [ ] Hard：伪标签 / 一致性正则，把 6786 张 unlabeled 用起来
- [ ] 固化成 `HW03_solution.ipynb` + `SOLUTION.md`
