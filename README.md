# BF-SAC：昂贵交通仿真的代理辅助标定

行为指纹代理辅助标定（**B**ehavioral **F**ingerprint **S**urrogate-**A**ssisted **C**alibration）的开源实现，在 **固定 100 次 SUMO 仿真预算** 下标定微观交通数字孪生的跟驰 / 换道参数。

## 核心思想

单次参数评估需一次完整 SUMO–TraCI 仿真，成本高昂。BF-SAC 用三步把有限预算花在刀刃上：

1. **行为指纹**：从真实轨迹提取 8 维统计指纹（速度水平 / 分位、加减速强度、停车占比…），以加权相对误差 $J_b$ 度量仿真与真实的差距——把崎岖的「参数 → 误差」关系压平到代理可学。
2. **LHS 初始设计**：拉丁超立方在 10 维参数空间采 40 个空间填充点，各触发一次仿真。
3. **序贯 LCB 加点**：训练回归代理，按下置信界 $\text{LCB}=\mu-\kappa\sigma$（κ 从 2.0 线性退火至 0.5）把剩余 60 次仿真定向投放到最有希望的区域。

代理可插拔、二选一：**随机森林（默认，轻量稳健）** 或 **MLP 深度集成（高容量，难场景更细）**，序贯机制完全共用。

## 仓库结构

| 路径 | 说明 |
|------|------|
| [`code/calibration/`](code/calibration/) | BF-SAC 标定主程序 `unified_calibration.py` |
| [`code/preprocessing/`](code/preprocessing/) | 由 SIND / UTE 原始轨迹构建 SUMO 场景 |
| [`code/experiments/`](code/experiments/) | 公平对比、N_init 消融、基线与作图 |
| [`data/`](data/) | 原始 / 处理后数据 |
| [`outputs/results/`](outputs/results/) | 对比 CSV 与 `comparison_cache/*_b100.json` |
| [`docs/`](docs/) | 文档（见下） |

## 文档

| 文档 | 作用 |
|------|------|
| 本 README | 项目概览、安装、一键出图、方法简介 |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | 完整复现：实验矩阵与规模、一键脚本、代理超参、输出位置 |
| [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) | 实验设计、结果与解读（含图表） |
| [`docs/DATA.md`](docs/DATA.md) | 数据获取、目录结构与预处理 |
| [`docs/ACADEMIC_NARRATIVE.md`](docs/ACADEMIC_NARRATIVE.md) | 论文写作大纲 |

## 研究场景

六个异构场景（三个 SIND 信号交叉口 + 三个 UTE 城市快速路）：
`Tianjin`、`Changchun`、`Xian`、`YTDJ`、`RML`、`XAM-N6`。

## 环境

- Python 3.10+；`pip install -r requirements.txt`
- [Eclipse SUMO](https://eclipse.dev/sumo/)，`sumo` 在系统 `PATH` 中（重跑标定 / 对比时需要）；TraCI 通常随 SUMO 安装
- 基线 CMA-ES / TPE 分别依赖 `cma` / `optuna`（已列入 `requirements.txt`）

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始：一键出图（无需 SUMO）

仓库已提交处理后的对比结果，克隆后可直接复现全部 release 图表：

```bash
python code/experiments/plot_all_figures.py
```

输出至 `outputs/figures/`：四张主图（PNG / PDF / SVG）及 `comparison_significance.csv`。

## 完整复现（需要 SUMO）

```bash
# 1. 数据准备（若从上游原始轨迹起步）
python code/preprocessing/organize_sind.py
python code/preprocessing/build_all_sumo.py
python code/preprocessing/build_ytdj_sumo.py
python code/preprocessing/build_rml_sumo.py
python code/preprocessing/build_xam_sumo.py

# 2. BF-SAC 标定（每场景 100 次 SUMO）
python code/calibration/unified_calibration.py               # 全场景，RF
python code/calibration/unified_calibration.py --mlp XAM-N6  # 单场景 + MLP 代理

# 3. 公平对比与消融（100 次预算 × 5 seed，断点续跑）
python code/experiments/run_batch_parallel.py --workers 12 --phases main
python code/experiments/run_batch_parallel.py --workers 12 --phases sweep

# 4. 作图
python code/experiments/plot_all_figures.py
```

算力机一键全流程：`bash scripts/run_all.sh`（Linux）或 `.\scripts\run_all.ps1`（Windows），见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

环境自检：`python code/experiments/check_sumo_scenes.py`（六场景配置 + 单次 SUMO 试跑，通过则退出码 0）。

## 方法对比

主对比把 BF-SAC（RF / MLP）放在四个机制各异的基线面前：**SPSA**（局部梯度估计）、
**GA**（种群进化）、**CMA-ES**（演化策略）、**TPE**（贝叶斯密度估计，Optuna）。
六场景 × 5 seed、统一 100 次预算下，BF-SAC 跨四种范式一致领先。详见 [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md)。

## 学习型代理：RF 与 MLP（二选一）

BF-SAC 的学习型组件是回归 **代理**，学习「10 维参数 → 标定误差 $J_b$」，在固定预算下用廉价预测指导序贯采点。两种实现接口一致、可直接互换（GA / SPSA 是优化器、8 维指纹是人工特征，二者均非学习模型）。

| | 随机森林（默认） | MLP 深度集成（`--mlp`） |
|---|---|---|
| 实现 | `RandomForestRegressor`（500 树） | `MLPEnsemble`：10 个 `(64,64)` tanh / lbfgs 网络 |
| 不确定度 σ | 树间预测离散度 | 成员间分歧（× σ 校准系数 1.2） |
| 采集函数 | $\text{LCB}=\mu-\kappa\sigma$，κ 退火 2.0→0.5 | 同左，选点逻辑独立实现 |
| 适用 | 轻量、对小样本稳健 | 容量更高，难拟合响应面更细 |
| 落盘 | 不落盘 | `mlp_models/*.joblib`，结果带 `__mlp_h64-64_m10` 后缀、与 RF 并存 |

MLP 默认开启 bootstrap + 子采样以维持集成成员分歧——这是不确定度 σ 不塌缩、LCB 探索有效的前提；超参可经 CLI / 环境变量调整，见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

输入 / 输出归一化：RF 路径不归一化（树模型对单调缩放不变）；MLP 对输入按物理边界做 Min-Max、对输出按训练集做 Z-score，训练与推理共用同一套参数（无尺度错位、无泄漏），σ 与 LCB 均落在真实 $J_b$ 尺度上。

## 数据集

- [SIND](https://github.com/SOTIF-AVLab/SinD) — 信号交叉口轨迹
- [UTE](https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye) — 城市快速路轨迹

使用原始数据请遵守各数据提供方的许可协议。

## 许可

MIT — 见 [LICENSE](LICENSE)。数据集许可另行适用。
