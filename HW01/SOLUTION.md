# HW01 Solution Write-up

COVID-19 Cases Prediction（NTU ML2021 Spring, HW1）· Kaggle `ml2021spring-hw1` · RMSE

最终成绩：**private 0.88478 / public 0.87955**（strong baseline 为 private 0.88，差约 0.005）。

---

## 1. 任务与数据

给定每个州连续三天的调查数据，预测第三天（严格说是窗口最后一天）的 `tested_positive`。

- 训练集 2700 行，测试集 893 行，93 个特征 + 1 个 target。
- 特征是 40 个州的 one-hot，加上 3 天 × 18 个调查指标（`cli`、`ili`、`hh_cmnty_cli`、`nohh_cmnty_cli`、口罩、出行、焦虑等），最后一天少一个 `tested_positive`（它是 target）。
- 行是按州排列的：每个州连续 67 或 68 天，40 个州共 2700 行。
- 测试集是每个州之后的 22–23 天，共 893 行。也就是说，**测试集是未来**，不是随机抽样。

---

## 2. 三个最容易踩的坑

### 2.1 target 列名重复

原始 CSV 里叫 `tested_positive` 的列有三处：

| 位置 | pandas 读进来的名字 | 含义 |
|---|---|---|
| day 1 最后一个特征 | `tested_positive` | 特征 |
| day 2 最后一个特征 | `tested_positive.1` | 特征 |
| 最后一列 | `tested_positive.2` | **target** |

所以 `df['tested_positive']` 拿到的是 day-1 特征，不是 target。用它当 y，随机 CV 的 RMSE 会低到 0.03——因为模型直接从特征里抄答案。第一版 EDA 就踩了这个坑，所有 CV 数字全错。之后统一用 `df.iloc[:, -1]` 取 target。

### 2.2 官方 `i % 10 == 0` 验证集会泄漏

数据按州连续排列，相邻两行是同一天的下一个窗口：

```
row i:     day1, day2, day3,  y = day3_tp
row i+1:   day2, day3, day4,  y = day4_tp
```

于是 `y_i = tested_positive.1_{i+1}`。官方 dev 是每隔 10 行取一行，验证行的 target 几乎总能从相邻训练行的 day-2 特征里找到。随机 CV 会给出 0.03–0.15 这种不可信的 RMSE。

正确做法：**按州取最后 20% 的天数做验证**。每个州前 80% 天训练，最后 20% 天验证。这样验证集和测试集一样，都是“用过去预测未来”。

在这个 split 上：

| baseline | RMSE |
|---|---|
| 直接预测 `y = tested_positive.1`（persistence） | 1.178 |
| Ridge on F17 | 1.066 |
| 1-hidden-layer MLP on F17 | 1.04–1.05 |

测试集 LB 的 RMSE 会低到 0.88 左右，因为最后 20% 训练期恰好是一段特别难预测的上升期；CV 只用来做相对比较，不用来估绝对 LB。

### 2.3 训练太久会过拟合

公开的“public strong baseline”notebook 写的是 `n_epochs=10000`，但它的 `dev` 就是训练集本身，`early_stop=500`。实际日志显示它在 **1495 epoch** 就因为 train loss 平台停了，train MSE 约 0.979。

我第一版固定训 6000 epoch，LB 只有 0.94；改成按 train loss 平台早停（patience 1200，通常 500–1200 epoch）后，同一个模型立刻到 0.894。**这个数据集上，训练轮数比网络结构重要得多。**

---

## 3. 特征

三套特征，都是从公开 strong baseline 的 14 维选择出发：

- **F14**：day1/day2 的 `tested_positive` + `cli`、`ili`、`hh_cmnty_cli`、`nohh_cmnty_cli` 各 3 天 = 14 维。
- **F17**：F14 + `worried_finances` × 3 天。这是公开 private baseline 用的 17 维。
- **F_eng**：F17 + 四个症状的 `d21`、`d32`、`acc`（加速度）+ `tp_delta`、`tp_ratio`，共 31 维。

几个观察：

- 加 `worried_finances` 稳定提升约 0.005–0.006 private。
- 加 40 个州 one-hot 对 MLP 是负收益（CV 从 1.05 掉到 1.26），但对 ridge 有帮助，所以只在 ridge 里用。
- 全部 93 维、national daily mean、state-relative 特征都没有稳定收益。
- 加速度特征在 Huber 模型里有小幅帮助，但单独看不如 17 维稳。

---

## 4. 模型

最终是 5 个模型的加权融合：

| 组件 | 配置 | 单独 private |
|---|---|---|
| MLP | `F_eng`, 32 hidden, BN + Dropout 0.2 + LeakyReLU, Adam lr 1e-3, L2 7.5e-4, Huber δ=2, batch 200, train-loss early stop | 0.8923 |
| Ridge | `F17`, λ=0.1, closed form | 0.8922 |
| Ridge + state | `F17` + 40 state one-hot, λ=0.1 | 0.9518 |
| ExtraTrees | 500 trees, `min_samples_leaf=2`, `max_features=0.7`, `F17` | 1.0155 |
| RandomForest | 400 trees, `min_samples_leaf=2`, `max_features=0.6`, `F17` | 0.9714 |

融合权重：

```
0.30 * MLP
+ 0.30 * Ridge(F17)
+ 0.20 * Ridge(F17 + state)
+ 0.05 * ExtraTrees
+ 0.15 * RandomForest
```

树模型单独很差，但小权重（合计 20%）能提供和 MLP 不同的误差结构，把 private 从 0.891 拉到 0.8848。树权重再大（30%）就回退到 0.888，再小（10%）是 0.8850 左右，20% 是局部最优。

单模型最好的是 17 维 MLP（private 0.8938）；3-seed MSE ensemble 是 0.8935。融合的增益主要来自 ridge/tree 的多样性，而不是 seed averaging。

---

## 5. 复现之前的尝试

之前的最好成绩是 Kaggle 上的 `hw01_huber_accel_delta_square.csv` / `hw01_huber_recent_acceleration.csv`，private 0.88526，public 0.90267。配方是：

- 14 维特征选择（`SelectKBest(f_regression)` 选出来的那 14 个）
- 单隐层 MLP + BN + Dropout + Huber + L2（ridge）+ Adam
- 修正验证集泄漏、特征筛选、optimizer

这次完整复现后：

| 版本 | public | private |
|---|---|---|
| 参考 public baseline notebook 的输出 | 0.88137 | 0.89887 |
| 我复现的 F14 + MSE + 早停 | 0.88744 | 0.90020 |
| 我复现的 F17 + MSE + 早停 | 0.88129 | 0.89380 |
| 之前的 best（Huber + acceleration） | 0.90267 | 0.88526 |
| **本次最终 ensemble** | **0.87955** | **0.88478** |

最终比之前的 best 好 0.0005，基本是同一水平；离 strong baseline 0.88 还差 0.0048，和你之前说的“差个 0.004”几乎一样。

---

## 6. 试过但没用的东西

- **全部 93 维 + MLP**：CV 1.38，过拟合。
- **按天跨州均值 / 州相对值**：CV 从 1.09 掉到 1.12–1.16，LB 也没有提升。
- **残差模型**（`y - tp2`、`y - 2*tp2 + tp1`）：单独 LB 0.91–0.93，融合后也没有改善。
- **per-state ridge**：每个州只有 67 个样本，CV 2.2，严重过拟合。
- **更长训练 / 更多 seed**：6000 epoch 固定训练明显过拟合；seed 增加到 3 个只提升 0.0005。
- **线性校准到之前的 best 预测**：private 0.8927，反而变差，说明差距不是简单的均值/方差偏移。
- **HGB / RandomForest / ExtraTrees 单独使用**：0.97–1.02，只能当融合的多样性来源。

---

## 7. 为什么最后 0.005 很难

- public/private 都是 893 行，RMSE 的抽样噪声大约 ±0.01。0.8848 和 0.88 的差距在统计上并不显著，但在 leaderboard 上就是差一点。
- 测试期是未来，特征分布和训练期有偏移；全局 MLP 很容易 regression to the mean，低值段预测偏高。之前的 Huber 模型预测出 0.96 这种低值，而我的融合最低只到 2.0 左右。
- 真正能再进一步的可能是 **out-of-fold stacking**（用时间 CV 的 OOF 预测训练 meta-model），或者对时间偏移做显式建模（例如按预测天数分段校准）。这两者这次都没有时间做严谨验证。
- 继续在 private LB 上手工调融合权重已经是在过拟合这 893 行，所以停在这里。

---

## 8. 复现方式

Kaggle notebook：<https://www.kaggle.com/code/aviator22/hw1-final>

在 Kaggle 新建 notebook，Add Data → Competition → `ml2021spring-hw1`，然后运行 [`HW01_solution.ipynb`](HW01_solution.ipynb)。约 3–5 分钟（CPU）生成 `submission.csv`。

主要依赖：PyTorch、scikit-learn、pandas、numpy。无需 internet。
