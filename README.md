# BF-SAC：昂贵交通仿真的代理辅助标定

面向 **固定 SUMO 仿真预算** 的行为指纹代理辅助标定（Behavioral Fingerprint Surrogate-Assisted Calibration, **BF-SAC**）开源实现，用于标定 SUMO 数字孪生微观参数。

## 仓库结构

| 路径 | 说明 |
|------|------|
| [`code/calibration/`](code/calibration/) | BF-SAC 标定（`unified_calibration.py`） |
| [`code/preprocessing/`](code/preprocessing/) | 由 SIND / UTE 原始轨迹构建 SUMO 场景 |
| [`code/experiments/`](code/experiments/) | 公平对比、消融、基线（GA/SPSA）与作图 |
| [`data/`](data/) | 原始/处理后数据（见 [`data/README.md`](data/README.md)） |
| [`outputs/results/`](outputs/results/) | 对比 CSV 与 `comparison_cache/*_b101.json` |
| [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) | 实验协议（标定 + 对比 + 作图） |

## 研究场景

六个异构场景：

`Tianjin`、`Changchun`、`Xian`、`YTDJ`、`RML`、`XAM-N6`

（三个 SIND 信号交叉口 + 三个 UTE 快速路场景）

## 环境要求

- Python 3.10+
- [Eclipse SUMO](https://eclipse.dev/sumo/)，且 `sumo` 在系统 `PATH` 中（重跑标定/对比实验时需要）
- Python 包：`pip install -r requirements.txt`
- TraCI：通常随 SUMO 安装；若 `import traci` 失败，请将 SUMO 的 `tools` 目录加入 `PYTHONPATH`

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
pip install -r requirements.txt
```

## 一键出图（无需 SUMO）

仓库已提交处理后的对比结果，克隆后可直接复现 release 图表：

```bash
pip install -r requirements.txt
python code/experiments/plot_all_figures.py
```

生成文件：

- `outputs/figures/bfsac_calibration_convergence.png` — 六场景 BF-SAC 标定收敛
- `outputs/figures/method_comparison_panel.png` — 方法对比与消融组合图
- `outputs/figures/multiscene_method_convergence.png` — 六场景多方法收敛
- `outputs/results/ablation_summary.csv` — 消融汇总表

## 完整复现（需要 SUMO）

详见 [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md)。典型顺序：

```bash
# 1. 数据准备（若从 SinD 上游起步）
python code/preprocessing/organize_sind.py
python code/preprocessing/build_all_sumo.py
python code/preprocessing/build_ytdj_sumo.py
python code/preprocessing/build_rml_sumo.py
python code/preprocessing/build_xam_sumo.py

# 2. BF-SAC 标定（每场景 101 次 SUMO，耗时较长）
python code/calibration/unified_calibration.py
python code/calibration/unified_calibration.py XAM-N6   # 单场景

# 3. 对比与消融（公平预算 101 次，支持断点续跑）
python code/experiments/run_comparison.py --scenes XAM-N6 --methods BF-SAC,No-RF
python code/experiments/run_comparison.py --resume

# 4. 作图
python code/experiments/plot_all_figures.py
```

### SUMO 环境自检

运行下列脚本，检查六场景配置、真实指纹与**单次** SUMO 评估是否可跑通：

```bash
python code/experiments/check_sumo_scenes.py
```

全部通过时退出码为 0。本机最近一次检查：六场景 `sumocfg`、`rou.xml`、真实指纹与试跑 SUMO 均正常（需已安装 SUMO 且数据目录完整）。

## 方法概要

真实轨迹提取 **8 维行为指纹**；每个候选 SUMO 参数向量触发一次仿真，用 $J_b$ 度量与真值指纹的偏差。BF-SAC 先用 **拉丁超立方（LHS）** 填满设计空间，再拟合 **随机森林** 代理，并以 **序贯 LCB** 加点直至预算用尽。

## 数据集

- [SIND](https://github.com/SOTIF-AVLab/SinD) — 信号交叉口轨迹
- [UTE](https://github.com/Ruyi-Feng/Ubiquitous-Traffic-Eye) — 城市快速路轨迹

使用原始数据请遵守各数据提供方许可协议。

## 许可

MIT — 见 [LICENSE](LICENSE)。数据集许可另行适用。

## 实验工作流（从原始数据到结果）

下表按执行顺序列出从原始轨迹到最终图表所经过的**文件**与**核心函数**。SIND（信号交叉口）与 UTE（快速路）两类场景在预处理阶段分两条支线，自标定阶段起合并为统一流程。

### 阶段 0 — 预处理：原始轨迹 → SUMO 场景

**SIND 支线**（Tianjin / Changchun / Xian）

| 步骤 | 文件 | 核心函数 | 作用 |
|------|------|----------|------|
| 0.1 整理原始记录 | [`code/preprocessing/organize_sind.py`](code/preprocessing/organize_sind.py) | `organize()` → `copy_if_exists()` | 将上游 SinD 记录按统一命名复制为 `raw_data/SIND/{城市}/veh_tracks.csv`、`map.osm`、`traffic_lights.csv` |
| 0.2 清洗静止车辆 | [`code/preprocessing/clean_static_vehicle_tracks.py`](code/preprocessing/clean_static_vehicle_tracks.py) | `clean_city()` → `classify_tracks()` | 剔除路边停放/全程静止轨迹（位移与最大速度双阈值），写回 `veh_tracks.csv` 并生成 `static_filter_report.json` |
| 0.3 构建交叉口场景 | [`code/preprocessing/build_all_sumo.py`](code/preprocessing/build_all_sumo.py) | `parse_intersection_center()`、`extract_od()`、`extract_signal_timing()`、`build_connections()`、`build_tls_state()` | 解析几何中心、从轨迹反推 OD 与转向比例、从信号 CSV 提取配时，调用 `netconvert` 生成 `sumo/*.net.xml / *.rou.xml / *.sumocfg`，并汇总 `all_cities_od_stats.csv` |

**UTE 支线**（YTDJ / RML / XAM-N6，原始输入为各场景 `frenet.csv`）

| 步骤 | 文件 | 核心函数 | 作用 |
|------|------|----------|------|
| 0.4 构建快速路场景 | [`code/preprocessing/build_ytdj_sumo.py`](code/preprocessing/build_ytdj_sumo.py)、[`build_rml_sumo.py`](code/preprocessing/build_rml_sumo.py)、[`build_xam_sumo.py`](code/preprocessing/build_xam_sumo.py) | `main()` → `build_*_network()` → `run_netconvert()` → `estimate_flows()`（YTDJ 为 `estimate_ytdj_flows()`）→ `build_routes()` → `build_sumocfg()` → `run_quick_sumo()` | 按路段几何写出 node/edge/connection，`netconvert` 生成路网；由 `frenet.csv` 估计各方向/匝道流量，生成路由与配置，并写 `*_build_summary.json` |

### 阶段 1 — BF-SAC 标定：SUMO 场景 → 标定参数

文件：[`code/calibration/unified_calibration.py`](code/calibration/unified_calibration.py)，入口 `main()` 对每个场景调用 `calibrate_one()`。

| 子步骤 | 核心函数 | 作用 |
|--------|----------|------|
| 1.1 提取真实指纹 | `extract_real_features()` → `compute_features()` | 从 `veh_tracks.csv`（SIND）或 `frenet.csv`（UTE）计算 8 维行为指纹作为标定目标 |
| 1.2 阶段 A：LHS 初始设计 | `lhs_samples_for_scene()`（`scipy` 拉丁超立方）+ `_eval_params()` | 在 10 维参数空间采 `N_INIT=60` 点，每点经一次真仿真评估 |
| 1.3 单次仿真评估 | `_run_sumo_traci()` → `compute_features()` → `feature_error()` | 经 TraCI 启动 SUMO、写入候选参数、采集速度/加速度，计算与真值指纹的加权误差 $J_b$ |
| 1.4 阶段 B：序贯 LCB 加点 | `_fit_rf()`、`_pick_lcb_point()`（含 `_propose_candidates()`、`_rf_lcb()`、`_min_rel_dist()`） | 重训随机森林代理 → 信赖域+全局候选 → 下置信界选点 → 真仿真并入库，循环至预算 `BUDGET_SUMO=101` |
| 1.5 落盘与回写 | `calibrate_one()` 返回结果；`apply_calibration()` | 写 `{场景}_calibration.json`、`_with_history.json`、`calibration_summary.csv`、`calibration_all.json`，并把最优参数写回 `*.rou.xml` |

### 阶段 2 — 方法对比与消融：SUMO 场景 → 对比结果

文件：[`code/experiments/run_comparison.py`](code/experiments/run_comparison.py)，入口 `main()` → `run_one()` 按方法分派。

| 方法 | 核心函数 | 作用 |
|------|----------|------|
| 评估器 | `SumoEvaluator.evaluate_vector()` → `_run_sumo_traci()` + `feature_error()` | 统一封装每次真仿真并记录收敛轨迹（`error_at_budget` 取第 101 次快照） |
| BF-SAC | `method_bfsac()` → 复用 `calibrate_one()` | 序贯代理标定 |
| 消融/基线 | `method_no_rf()`、`method_no_lhs()`、`method_spsa()`、`method_ga()` | 无 RF（纯 LHS）、无 LHS（随机）、SPSA、GA，统一公平预算 101 |
| 输出 | `save_cached()` / `main()` | 写 `comparison_summary.csv`、`comparison_convergence.csv`、`comparison_cache/{场景}_{方法}_b101.json` |

环境自检：[`code/experiments/check_sumo_scenes.py`](code/experiments/check_sumo_scenes.py) 的 `main()` 校验六场景配置、真实指纹与单次 SUMO 试跑。

### 阶段 3 — 作图：结果 CSV/JSON → 图表

文件：[`code/experiments/plot_all_figures.py`](code/experiments/plot_all_figures.py)，`main()` 顺序调用三个作图脚本的 `main()`：[`plot_calibration_convergence.py`](code/experiments/plot_calibration_convergence.py)、[`plot_method_comparison.py`](code/experiments/plot_method_comparison.py)、[`plot_multiscene_convergence.py`](code/experiments/plot_multiscene_convergence.py)。输出三张 PNG 与 `outputs/results/ablation_summary.csv`。

> 数据流一图概览：
> `原始轨迹 (veh_tracks.csv / frenet.csv)` → **阶段 0** → `SUMO 场景 (*.sumocfg)` → **阶段 1（BF-SAC）** → `calibration/*.json` →（**阶段 2** 对比 → `comparison_*.csv`）→ **阶段 3** → `figures/*.png`

## 机器学习模型：随机森林代理

BF-SAC 中唯一的**学习型模型**是随机森林回归代理（surrogate）。它学习「参数向量 → 标定误差 $J_b$」的映射，从而在固定 SUMO 预算下用廉价预测替代昂贵真仿真，指导序贯采点。GA/SPSA 是优化器、8 维行为指纹是人工设计的特征，二者均**非**学习型模型；下表只描述随机森林代理。

| 项 | 内容 |
|----|------|
| 实现 | `sklearn.ensemble.RandomForestRegressor`，于 [`unified_calibration.py`](code/calibration/unified_calibration.py) 的 `_fit_rf()` 构建 |
| 结构 | `n_estimators=500`、`max_depth=20`、`random_state=42`（`n_jobs` 随场景取 1 或 -1）；500 棵回归树的集成 |
| 训练数据 | 截至当前已评估的设计点：输入 `X[:n_seen]`、标签 `Y[:n_seen]`，过滤掉失败样本（`NaN` 或 `Y≥10`），有效样本需 `≥ MIN_VALID_RF=10` 才训练 |
| 输入 | **10 维参数向量**（`PARAM_NAMES`）：6 维 Krauss 跟驰 `accel, decel, sigma, tau, minGap, speedFactor` + 4 维 LC2013 换道 `lcStrategic, lcCooperative, lcAssertive, lcSpeedGain`，取值受 `PARAM_BOUNDS` / `get_param_bounds()` 约束 |
| 输出 | **标量 $J_b$**（加权行为指纹误差，越小越优），即对真仿真目标函数 `feature_error()` 的回归预测 |
| 采集函数 | `_rf_lcb()`：对候选点取 500 棵树预测的均值 $\mu$ 与标准差 $\sigma$，返回下置信界 $\text{LCB}=\mu-\kappa\sigma$（`LC_KAPPA=1.96`）；`_pick_lcb_point()` 在信赖域+全局候选中选 LCB 最小、且与已评估点距离足够远的点 |
| 调用位置 | `calibrate_one()` 阶段 B 每轮循环：`_fit_rf()` 重训 → `_pick_lcb_point()` 选点 → `_eval_params()` 真仿真验证 → 入库后再训，直至用尽预算 |
| 作用 | 在「百次仿真」预算内以代理外推压低 $J_b$；其训练拟合质量 `rf_r2` / `rf_rmse` 一并记入标定结果 JSON |

> 训练—预测—验证闭环：`真实指纹（目标）` ← 比较 → `_run_sumo_traci()` 仿真指纹 ⇒ `feature_error()` 得到 $J_b$ ⇒ 作为标签训练 `_fit_rf()` 随机森林 ⇒ `_rf_lcb()` 预测并选下一参数点 ⇒ 再次真仿真验证（闭环）。
