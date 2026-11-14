# HW01 — COVID-19 Cases Prediction (Regression)

Kaggle: [ml2021spring-hw1](https://www.kaggle.com/competitions/ml2021spring-hw1) · metric: RMSE

## 结果

| 方案 | public | private |
|---|---|---|
| 参考 public baseline notebook | 0.88137 | 0.89887 |
| 复现 14 维特征 MLP（MSE） | 0.88744 | 0.90020 |
| 复现 17 维特征 MLP（MSE） | 0.88129 | 0.89380 |
| 之前的 best（Huber + acceleration） | 0.90267 | 0.88526 |
| **本次最终 5 模型 ensemble** | **0.87955** | **0.88478** |

Strong baseline 是 private 0.88，最终还差约 **0.005**。单模型最好的是 17 维 MLP（private 0.8938），融合 ridge / ExtraTrees / RandomForest 后到 0.8848。

## 文件

- [`HW01_solution.ipynb`](HW01_solution.ipynb) — 最终方案，在 Kaggle 上挂载比赛数据后直接运行，输出 `submission.csv`
- [`SOLUTION.md`](SOLUTION.md) — 完整 solution write-up（数据陷阱、验证集、特征、模型、消融、失败尝试）
- `HW01.ipynb` — 课程原始 notebook

## 三个关键点

1. **target 列名重复**
   CSV 里 day-1 的 `tested_positive`、day-2 的 `tested_positive.1` 和 target `tested_positive.2` 名字相近。`df['tested_positive']` 拿到的是 day-1 特征，不是 target。统一用 `df.iloc[:, -1]`。

2. **官方 `i % 10 == 0` 验证集会泄漏**
   数据按州排列，每个州 67/68 天连续。某个验证样本的 target 恰好等于下一条样本的 `tested_positive.1`，而下一条通常在训练集里。随机 CV 的 RMSE 会低到 0.1，完全不可信。改成**每个州最后 20% 的天数做验证**，才和“预测未来”的测试集一致。

3. **训练轮数比模型结构更重要**
   固定训 6000 epoch 会过拟合，LB 只有 0.94；按 train loss 平台早停（约 700–1500 epoch）后，单模型就能到 0.894。公开 baseline 实际也是约 1495 epoch 早停的。

## 最终模型

```
0.30 * MLP(F_eng, 32 hidden, Huber delta=2)
+ 0.30 * Ridge(F17, lambda=0.1)
+ 0.20 * Ridge(F17 + 40 state one-hot, lambda=0.1)
+ 0.05 * ExtraTrees(500, F17)
+ 0.15 * RandomForest(400, F17)
```

- `F14` = `tested_positive` day1/day2 + `cli` / `ili` / `hh_cmnty_cli` / `nohh_cmnty_cli` × 3 天
- `F17` = `F14` + `worried_finances` × 3 天
- `F_eng` = `F17` + 四个症状的 delta / acceleration + `tp_delta` / `tp_ratio`
- MLP：1 隐层 + BatchNorm + Dropout 0.2 + LeakyReLU，Adam，L2 7.5e-4

## 运行

在 Kaggle 新建 notebook，添加 competition `ml2021spring-hw1`（Add Data → Competition），然后运行 `hw1_covid19.ipynb`，会在 working 目录生成 `submission.csv`。

Kaggle kernel: <https://www.kaggle.com/code/aviator22/hw1-final>
