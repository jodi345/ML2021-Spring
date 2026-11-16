# HW02 — Phoneme Classification（音素分类）

Kaggle: [ml2021spring-hw2](https://www.kaggle.com/competitions/ml2021spring-hw2) · metric: accuracy

## 结果

| 方案 | public | private |
|---|---|---|
| Simple baseline（官方线） | 0.68334 | — |
| Strong baseline（官方线） | 0.76023 | — |
| 官方 notebook 原样跑（Sigmoid / Adam 1e-4 / 20 epoch） | 0.68917 | 0.68869 |
| **单模型 v3（ReLU + BN + 窗口 + Dropout）← 全程最大提升 +4.29 点** | **0.73203** | **0.73073** |
| **单模型（最终，全量数据 + TTA）** | **0.74719** | **0.74657** |
| **七模型集成（最终提交）** | **0.75869** | **0.75778** |

距 strong baseline 差 **0.15 分**（约 1.5σ），停下理由见 [SOLUTION.md](SOLUTION.md) 最后一节。

## 文件

- [`HW02_solution.ipynb`](HW02_solution.ipynb) — 单模型最终方案，Kaggle 上挂比赛数据直接跑，输出 `submission.csv` + `probs.npy`
- [`HW02_ensemble.ipynb`](HW02_ensemble.ipynb) — 七模型概率加权集成
- [`SOLUTION.md`](SOLUTION.md) — 完整 write-up（数据陷阱、泄漏检查、容量拐点、HPO 陷阱、失败尝试）
- `HW02-1.ipynb` — 课程原始 notebook（分类）
- `HW02-2.ipynb` — Hessian 理论题（本解答不涉及）

## 四个关键点

1. **数据给的 429 维已经是 11 帧拼接**

   `429 = 11 × 39`，相邻 39 维块相关系数 0.843、首尾块 −0.11，说明这 11 块是时间上连续的 11 帧——每行自带约 125 ms 上下文。2022 版 HW2 那道「自己 concat 帧」的核心技巧，在 2021 版里已经被预处理做掉了。所以自己再加窗口只涨 1 个点，不存在「拼上就过线」的捷径。

2. **容量拐点在 2.8 M 参数**

   | 参数量 | train | val |
   |---|---|---|
   | 1.04 M | 0.791 | 0.704 |
   | **2.80 M** | 0.791 | **0.750** |
   | 12.81 M | 0.997 | 0.745 |

   12.8 M 那版把 98 万帧背下来了。「加宽 + 残差 + GELU」在这个任务上是负收益。

3. **数据没有泄漏，但也不能随机切**

   前后两半的标签分布 TV 距离 0.0195、KL 0.0013；特征均值漂移 0.01σ（最大 0.035σ）。所以不存在分布漂移或泄漏。切分仍用顺序切 80/20：线性探针下随机切与顺序切几乎一样（0.5231 vs 0.5216），但相邻帧共享 10/11 的特征，高容量模型在随机切分下会被虚高的验证分骗。

4. **短预算超参数搜索会给出错误答案**

   10 组配置 × 4 epoch 的随机搜索，冠军（hid 1536、drop 0.3、wd 0.1）跑满 40 epoch 就是过拟合那版（val 0.7448），输给 4-epoch 时排第 4 的配置（v3，val 0.7495）。**lr / batch / k 可以短预算搜；容量 / dropout / weight decay 必须接近满预算搜。**

## 最终集成

```
0.20 * v7a   MLP 1024/512/128, ReLU, 全量数据, seed 11
0.20 * v7b   MLP 896/448/112, ReLU, 全量数据, seed 22
0.18 * v8a   MLP, k=1 窗口(3帧), ELU
0.18 * v8b   MLP 1280/640/160, GELU
0.10 * v3    MLP 1024/512/128, 80% 数据, val 0.7495
0.08 * v5    残差 MLP 1536, GELU（过拟合，但误差互补）
0.06 * v4    Conv1d over 55×39（最弱单体，补多样性）
```

成员间两两预测分歧率 11%–21%，这是集成有效的原因；单体分数只决定权重。

## 运行

在 Kaggle 新建 notebook，Add Data → Competition `ml2021spring-hw2`，然后跑 `HW02_solution.ipynb`（GPU T4，约 25 分钟）。

集成版需要把 7 个成员 kernel 的输出挂上来（Add Data → Your Work → 选对应 kernel）：

```
ml2021spring-hw2-v7a, -v7b, -v8a, -v8b, -v5-mlp, -v4-cnn, -v3probs
```

Kernel:
- 单模型：<https://www.kaggle.com/code/aviator22/ml2021spring-hw2-v7a>
