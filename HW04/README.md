# HW04 — Speaker Classification（Self-Attention / Transformer）

Kaggle: [ml2021spring-hw4](https://www.kaggle.com/competitions/ml2021spring-hw4) · 指标 **@1 accuracy** · 600 类

## 任务卡片（S0）

| 项 | 值 | 来源 |
|---|---|---|
| 输入 | mel-spectrogram，`(T, 40)` float，T 每句不同；训练时随机裁 `segment_len=128` 帧 | `HW04.ipynb` + 实测 `metadata.json` |
| 输出 | 600 维 logits → 600 个 speaker id（`id2speaker` 映射回字符串） | `Classifier` |
| 训练量 | 69,438 条 utterance，600 个 speaker，`n_mels=40` | `metadata.json` |
| 测试量 | 6,000 条无标签，**不做裁剪**（喂整句） | `testdata.json` |
| 指标 | top-1 accuracy（`@1 Accuracy`） | HW04.pdf p15 |
| 官方三档 | simple 0.82523 / medium 0.90547 / hard 0.95404（conformer） | HW04.pdf p12–14 |
| 榜单天花板 | 0.98333（LB 第 1） | `kaggle competitions leaderboard` |
| 随机瞎猜 | 0.00167（1/600） | 计算 |

一句话：**输入一段 128 帧 × 40 维的 log-mel，输出 600 个说话人里最像的那一个，按 top-1 命中率算分；测试时喂的是整句（中位数 583 帧）。**

## 环境验证（开工前必须先确认的两件事）

| 检查 | 结果 | 说明 |
|---|---|---|
| 比赛能否下载/提交 | ✅ 能 | `competitions files` 正常列出；`leaderboard` 可见；`userHasEntered=True`；HW02 的先例证明 2021 社区赛至今仍接受 late submission（2026-11-16 仍在提交） |
| 竞赛数据是否完整 | ❌ **不完整** | 竞赛 zip 只有 **1.76 MB**：`metadata.json` / `mapping.json` / `testdata.json`，**没有一个 `.pt`**。特征文件必须走 notebook markdown 里的 GDrive id `1gaFy8RaQVUEXo2n0peCBR5gYKCB-mNHc` |
| 本地能否分析标签 | ✅ 能 | 三个 JSON 够做标签/长度/泄漏的全部诊断，不用下 3 GB 特征 |
| GPU | ✅ 无需排队 | probe kernel 秒开工（见下表）；HW03 的 180 s/epoch 是 CPU 上 4.26 M CNN 的数，HW04 结构小得多 |

## 数据（已实测，来自竞赛 `metadata.json` / `testdata.json`）

### ① 600 类严重不均衡（28 倍）

| 统计 | 值 |
|---|---|
| 每类 utterance 数 | min **35** / p10 47 / median **96** / p90 213 / max **992** |
| max ÷ min | **28.3×** |
| 每类 < 50 条的 speaker | 81 个（占数据 5.0%） |
| top-5 最大类占比 | 4.2% |

⚠️ **accuracy 会骗人**：模型在 992 条的类上刷对，在 35 条的类上全错，总分照样好看。所以每个版本都同时报 **macro F1** 和「有多少 speaker 从来没被预测过」。

### ② 训练/测试长度错配（这是本作业最实的一个 mismatch）

| | 长度（帧） |
|---|---|
| 训练 | 强制裁到 **128**（随机起点） |
| 验证 | 同样裁 128 |
| 测试 | **整句**：min 158 / p5 359 / median **583** / mean 727 / p95 1584 / max 7676 |

测试序列中位数是训练段的 **4.6 倍**。官方 sample code 里模型在 128 帧上训、在 583 帧上推。
按 `docs/model-training-notes.md` 的树，这属于**第 0 分支（测量/构造是否可信）**，不是模型容量问题：
先怀疑「训练与推理的输入分布不是同一个」，再谈加容量。

### ③ 无泄漏

| 检查 | 结果 |
|---|---|
| train 内部重复 feature_path | 0 |
| test 内部重复 feature_path | 0 |
| train ∩ test 文件名 | **0** |
| 600 个 speaker 是否 train/test 都出现 | 是（测试集就是同一批 600 人的另一些句子，所以「同人不同句」的随机切分**正是**测试协议，不存在泄漏） |
| `mapping.json`(竞赛) == `mapping.json`(GDrive) | prep kernel 校验（不一致会让提交 id 全错） |

> 结论：「人声泄漏」这条**不成立**——测试集本身就含训练过的 600 人，这是任务设定（closed-set speaker ID），不是数据错误。真正需要防的是「同一句同时进 train 和 test」，实测为 0。

## 官方 sample code 的实现坑（S1 会遇到）

1. **`gdown --id` 是占位符**（`'paste your own data download link'`），要换成真实 id。
2. **所谓「2 层 TransformerEncoder」在 sample 里是 1 层**：`self.encoder = nn.TransformerEncoder(..., num_layers=2)` 那一行被注释掉了，`forward()` 直接调 `self.encoder_layer`。所以「medium = 调 transformer 参数」的第一个动作就是把层数真正接上。
3. **推理按 `batch_size=1` 且不裁剪**：6000 次单条前向，且序列长度分布和训练完全不同（见 ②）。
4. **验证集用 `random_split` 非分层**：6944 条、600 类，σ(accuracy) ≈ 0.6 点，小类在 val 里可能只有 3–5 条 → macro F1 噪声大。
5. 官方 `valid()` 返回的是 `running_accuracy / len(dataloader)`，`drop_last=True` 会丢掉不满一个 batch 的尾巴（694 个 val batch，误差可忽略，但要知道）。

## 硬件实测（probe kernel，Tesla T4）

| 配置 | 参数 | ms/step（bsz32 / 128 帧） | 70k steps 预计 |
|---|---|---|---|
| 官方（d80 h2 ff256 **1 层**） | 193 k | **18.6** | **21.7 min** |
| 2 层 encoder | 261 k | 26.9 | 31.3 min |
| ff 1024 | 441 k | 16.0 | 18.6 min |
| d160 h4 ff512 | 665 k | 25.2 | 29.4 min |

- 机器：T4 15.6 GB / 4 vCPU / 33.7 GB RAM / torch 2.11.0+cu130，**GPU 无需排队**（两次 push 都是秒级 RUNNING）。
- 推理：整句（median 583 帧）6000 条在 bsz32 下约 **3 s**。官方 sample 的 bsz=1 写法会慢几十倍但也就几分钟。
- 结论：**算力不是瓶颈**（模型只有 193 k 参数）。真正的时间在**数据加载**——69 438 条 × 32 epoch = 2.2 M 次 `torch.load`，这才是要优化的地方（prep kernel 在测）。

## 进度快照（2026-11-16 晚，暂存点）

已完成：
1. S0 任务卡片（上表）+ 官方 notebook 坑清单。
2. 环境验证：比赛**能提交**（late submission 先例是 HW02）；竞赛数据缺 `.pt`（1.76 MB），必须走 GDrive。
3. S3 标签级诊断（全部来自 1.76 MB 的 JSON，没下大数据）：28× 类不均衡、测试整句 vs 训练 128 帧的 4.6× 长度错配、无泄漏。
4. probe kernel 完成：T4，21.7 min / 70k steps。

在跑 / 待办：
- `hw4-prep` kernel：GDrive 下载 → 校验 → 建 f16 memmap（4.4 GB）→ 测官方 loader 吞吐（**仍在 RUNNING**，输出会留在 Kaggle，下次直接复用）。
- `hw4-diag` kernel 已写好未 push（依赖 prep 输出）。
- `hw4-baseline` notebook 已写好未 push（官方原样 + seed + macro F1 + probs.npy）。**故意等 prep 的下载结束再 push，避免两个 kernel 同时拉 GDrive 触发配额。**
- 之后：S4 单变量实验（① 测试端 128 帧滑窗 + softmax 平均修长度错配 ② SpecAugment ③ attention pooling ④ 真正接上 2 层 ⑤ margin softmax 治不均衡）。

## 结果

| 版本 | 说明 | val acc | macro F1 | public | 备注 |
|---|---|---|---|---|---|
| baseline | 官方 sample 原样（d80/h2/ff256/**1 层**，AdamW 1e-3，70k steps） | 待跑 | — | — | notebook 已就绪 |

## 文件

- [`HW04.ipynb`](HW04.ipynb) — 课程原始 notebook（保持原样不动）
- [`HW04.pdf`](HW04.pdf) — 作业说明
- `README.md` — 本文
- （后续）`HW04_solution.ipynb` / `SOLUTION.md`

## 运行

Kaggle kernel：<https://www.kaggle.com/code/aviator22/ml2021spring-hw4-baseline>

配置：**GPU 开 / internet 开 / competition source = ml2021spring-hw4**。首次运行会 gdown 下载
Dataset.zip（约 3 GB）到 `/tmp`，不写进 kernel output。

## Kernel 索引

| kernel | 作用 | 状态 |
|---|---|---|
| [`ml2021spring-hw4-probe`](https://www.kaggle.com/code/aviator22/ml2021spring-hw4-probe) | 硬件/耗时探针 | ✅ COMPLETE |
| [`ml2021spring-hw4-prep`](https://www.kaggle.com/code/aviator22/ml2021spring-hw4-prep) | 下载 + 校验 + f16 memmap + loader 吞吐 | 🔄 RUNNING |
| `ml2021spring-hw4-diag` | 特征漂移 / 帧冗余 / 相似度泄漏 / 最近质心参考 | 待 push（依赖 prep） |
| `ml2021spring-hw4-baseline` | S1 官方 sample 锚点 | 待 push |

本地 kernel 源码在仓库外：`default-workspace/hw4_kaggle/{probe,prep,diag,baseline}/`（含 `build_baseline.py` 生成器）。
