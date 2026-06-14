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

真实轨迹提取 **8 维行为指纹**；每个候选 SUMO 参数向量触发一次仿真，用 $J_b$ 度量与真值指纹的偏差。BF-SAC 先用 **拉丁超立方（LHS）** 填满设计空间，再拟合 **代理模型**，并以 **序贯 LCB** 加点直至预算用尽。代理可二选一：**随机森林（默认）** 或 **MLP 深度集成**（`--mlp` 切换），详见下文「机器学习模型」与「如何使用 MLP 代理」两节。

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
| 1.4 阶段 B：序贯 LCB 加点 | `_fit_surrogate()`（分派 `_fit_rf()` / `_fit_mlp()`）、`_pick_lcb_point()`（含 `_propose_candidates()`、`_rf_lcb()` / `_mlp_lcb()`、`_min_rel_dist()`） | 重训代理（RF 默认或 MLP，二选一）→ 信赖域+全局候选 → 下置信界选点 → 真仿真并入库，循环至预算 `BUDGET_SUMO=101` |
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

## 机器学习模型：两种可切换的代理（随机森林 / MLP 深度集成）

BF-SAC 中的**学习型模型**是回归**代理（surrogate）**：学习「10 维参数向量 → 标定误差 $J_b$」的映射，从而在固定 SUMO 预算下用廉价预测替代昂贵真仿真，指导序贯采点。代理有两种实现，**二选一**、输入/输出接口完全一致：**随机森林（默认）** 与 **MLP 深度集成**（用 `--mlp` 切换，见下一节）。两者在代码中分成互不共用的两块，仅共享 `_fit_surrogate()` / `_pick_lcb_point()` 两个分派接口。GA/SPSA 是优化器、8 维行为指纹是人工设计的特征，二者均**非**学习型模型。

### 代理一：随机森林（默认）

| 项 | 内容 |
|----|------|
| 实现 | `sklearn.ensemble.RandomForestRegressor`，于 [`unified_calibration.py`](code/calibration/unified_calibration.py) 的 `_fit_rf()` 构建 |
| 结构 | `n_estimators=500`、`max_depth=20`、`random_state=42`（`n_jobs` 随场景取 1 或 -1）；500 棵回归树的集成 |
| 训练数据 | 截至当前已评估的设计点：输入 `X[:n_seen]`、标签 `Y[:n_seen]`，过滤掉失败样本（`NaN` 或 `Y≥10`），有效样本需 `≥ MIN_VALID_RF=10` 才训练 |
| 输入 | **10 维参数向量**（`PARAM_NAMES`）：6 维 Krauss 跟驰 `accel, decel, sigma, tau, minGap, speedFactor` + 4 维 LC2013 换道 `lcStrategic, lcCooperative, lcAssertive, lcSpeedGain`，取值受 `PARAM_BOUNDS` / `get_param_bounds()` 约束 |
| 输出 | **标量 $J_b$**（加权行为指纹误差，越小越优），即对真仿真目标函数 `feature_error()` 的回归预测 |
| 不确定度 $\sigma$ | 来自 **500 棵回归树预测的离散度**（`rf.estimators_` 各树预测的标准差），供 `_rf_lcb()` 计算 LCB |
| 采集函数 | `_rf_lcb()`：对候选点取 500 棵树预测的均值 $\mu$ 与标准差 $\sigma$，返回下置信界 $\text{LCB}=\mu-\kappa\sigma$（`LC_KAPPA=1.96`）；`_pick_lcb_point()` 在信赖域+全局候选中选 LCB 最小、且与已评估点距离足够远的点 |
| 调用位置 | `calibrate_one()` 阶段 B 每轮循环：`_fit_rf()` 重训 → `_pick_lcb_point()` 选点 → `_eval_params()` 真仿真验证 → 入库后再训，直至用尽预算 |
| 作用 | 在「百次仿真」预算内以代理外推压低 $J_b$；其训练拟合质量 `rf_r2` / `rf_rmse` 一并记入标定结果 JSON |

### 代理二：MLP 深度集成（`--mlp` 切换）

| 项 | 内容 |
|----|------|
| 实现 | `sklearn.neural_network.MLPRegressor` 的**深度集成**，封装为 `MLPEnsemble` 类，于 [`unified_calibration.py`](code/calibration/unified_calibration.py) 的 `_fit_mlp()` 构建 |
| 结构 | `MLP_ENSEMBLE_SIZE=10` 个独立 MLP 的集成；每个网络 `MLP_HIDDEN_LAYERS=(64,64)`、`MLP_ACTIVATION="tanh"`、`MLP_SOLVER="lbfgs"`、`MLP_ALPHA=1e-3`、`MLP_MAX_ITER=2000`；成员间 `MLP_BOOTSTRAP=True` 有放回重采样以增加分歧 |
| 训练数据 | 与 RF 相同：已评估设计点 `X[:n_seen]`、`Y[:n_seen]`，过滤失败样本（`NaN` 或 `Y≥10`），有效样本需 `≥ MIN_VALID_RF=10`；集成内部按 `PARAM_BOUNDS` 归一化输入、标准化输出。**子采样抗过拟合**：每个成员只在固定大小 `MLP_TRAIN_SUBSET=60`（= 阶段 A 数据量 `N_INIT`）的随机子集上训练——阶段 B 数据从 60 增至 100 时，单网络始终只见 ≤60 个点，既降低过拟合、又让各成员看到不同子集而增大分歧 |
| 续训方式 | **warm-start 增量续训**（`MLP_WARM_START=True`）：阶段 B 首轮从零训练（`_fit_mlp`），之后每轮在「上一轮各成员的权重」上继续训练（`_update_mlp`→`MLPEnsemble.update`，lbfgs 从已有权重再优化），而非每轮从零重训；为使续训处于同一目标尺度，输出归一化在首训时**冻结** |
| 输入 | 与 RF 相同的 **10 维参数向量**（`PARAM_NAMES`），取值受 `PARAM_BOUNDS` / `get_param_bounds()` 约束 |
| 输出 | **标量 $J_b$**，取 `MLP_ENSEMBLE_SIZE` 个成员预测的均值 $\mu$，即对 `feature_error()` 的回归预测 |
| 不确定度 $\sigma$ | 来自**集成成员间分歧**（成员预测的标准差，而非单棵树方差），再乘 σ 校准系数 `MLP_SIGMA_CALIB=1.2` 修正过自信；每个子网络用**运行时随机种子**初始化（记于 `MLPEnsemble.seeds`）以保证多样性 |
| 采集函数 | `_mlp_lcb()`：成员均值 $\mu$ 与（校准后）成员间标准差 $\sigma$ 给出 $\text{LCB}=\mu-\kappa\sigma$（`LC_KAPPA=1.96`）；`_mlp_pick_lcb_point()` 与 RF 同样的选点逻辑但**独立实现**（不复用 RF 任何函数） |
| 调用位置 | 与 RF 相同，`calibrate_one()` 阶段 B 每轮：首轮 `_fit_mlp()` 从零训练、之后 `_update_mlp()` 续训 → `_pick_lcb_point()` 选点 → `_eval_params()` 真仿真验证 → 入库再训；循环结束 `_save_mlp_model()` 把集成持久化为 `.joblib` |
| 作用 | 作为 RF 的**平行平替**（二选一）；结果文件名追加 `__mlp_h64-64_m10` 与 RF 结果并存。集成规模与 σ 校准由**日志离线交叉验证**标定：`M:5→10` 使留出 RMSE 降约 6%、$\sigma$ 的 95% 覆盖率从 0.85 升到 0.91 |

#### MLP 的输入/输出归一化（训练与推理一致）

MLP 集成在内部对数据做归一化（**RF 路径不归一化**——树模型对单调缩放不变，无需处理）；归一化参数随各成员一并保存在 `_ScaledMLP` 中，训练与推理用**同一套**参数，无尺度错位、无数据泄漏。

| 数据 | 方式 | 公式（训练 / 推理共用） | 基准 |
|------|------|------------------------|------|
| **输入**（10 维参数向量） | Min-Max 归一化 | `xs = (x − x_lo) / x_span` | `x_lo`、`x_span` 取 `PARAM_BOUNDS` **物理边界**（非数据 min/max），缩放固定、确定 |
| **输出**（标量 $J_b$） | Z-score 标准化 | 训练 `ys = (y − y_mean) / y_std`；推理 `y = ŷ·y_std + y_mean`（反标准化） | `y_mean`、`y_std` 由**整个有效训练集**计算（`bootstrap`/子采样只改采样、不改标准化基准）；并在**首训时冻结**，供后续 warm-start 续训复用，保证增量更新处于同一目标尺度 |

- **训练**（`MLPEnsemble.fit`）：先按上表归一化输入、标准化标签，再用归一化后的数据训练每个 `MLPRegressor`，并把 `(x_lo, x_span, y_mean, y_std)` 存入对应 `_ScaledMLP`；同时把归一化基准冻结到集成上。
- **续训**（`MLPEnsemble.update`）：复用冻结的归一化基准，对各成员重抽一份子集后在已有权重上 warm-start 续训（见上文「续训方式」）。
- **推理**（`_ScaledMLP.predict`）：同样归一化输入 → 网络预测 → 把输出**反标准化**回原始 $J_b$ 尺度。
- 因此聚合得到的 $\mu$、$\sigma$ 以及 $\text{LCB}=\mu-\kappa\sigma$（含 σ 校准系数 `MLP_SIGMA_CALIB`）全部落在**真实 $J_b$ 物理尺度**上，量纲一致。

> 训练—预测—验证闭环（两代理通用）：`真实指纹（目标）` ← 比较 → `_run_sumo_traci()` 仿真指纹 ⇒ `feature_error()` 得到 $J_b$ ⇒ 作为标签训练 `_fit_surrogate()` 代理（RF 或 MLP）⇒ `_rf_lcb()` / `_mlp_lcb()` 预测并选下一参数点 ⇒ 再次真仿真验证（闭环）。

## 如何使用 MLP 代理（随机森林的平行平替）

除随机森林外，`unified_calibration.py` 还内置一个 **MLP 深度集成** 代理，与 RF **二选一**：两者输入/输出接口一致（同样学习「10 维参数 → $J_b$」并以序贯 LCB 加点），可直接互换。**默认仍为随机森林**；要改用 MLP，只需在运行标定时切换代理，无需改任何代码。

### 启用方式（三选一，等价）

```bash
# 1) 命令行开关（推荐）
python code/calibration/unified_calibration.py --mlp            # 全部场景用 MLP
python code/calibration/unified_calibration.py --mlp XAM-N6     # 单场景 + MLP

# 2) --surrogate= 写法
python code/calibration/unified_calibration.py --surrogate=mlp

# 3) 环境变量
BFSAC_SURROGATE=mlp python code/calibration/unified_calibration.py

# 显式用回随机森林（默认行为）：--rf 或 --surrogate=rf
python code/calibration/unified_calibration.py --rf
```

> `--mlp` / `--rf` / `--surrogate=<kind>` 之外的位置参数仍按「场景名」解析，用法与 RF 模式完全一致。

### MLP 子采样的命令行参数化（仅 `--mlp` 模式有效，RF 不受影响）

成员子采样的两个关键参数可直接在命令行覆盖，**无需改代码**（不传则用代码默认 `MLP_BOOTSTRAP=True` / `MLP_TRAIN_SUBSET=60`）：

```bash
# 关闭有放回重采样（无放回抽子集）
python code/calibration/unified_calibration.py --mlp --no-bootstrap XAM-N6

# 无放回 + 子集大小 40（消除重复样本、靠更小子集维持成员分歧）
python code/calibration/unified_calibration.py --mlp --no-bootstrap --subset=40 XAM-N6

# 改子集大小 / 用全部历史数据（不做子采样）
python code/calibration/unified_calibration.py --mlp --subset=80 XAM-N6
python code/calibration/unified_calibration.py --mlp --subset=none XAM-N6   # none/all/full/0 均表示用全部数据
```

| 参数 | 取值 | 作用（覆盖的常量） |
|------|------|------|
| `--bootstrap` / `--no-bootstrap` | 开 / 关 | `MLP_BOOTSTRAP`：成员训练子集是否有放回抽样 |
| `--subset=<N>` | 正整数 | `MLP_TRAIN_SUBSET`：每个成员训练子集大小 |
| `--subset=none` | `none`/`all`/`full`/`0` | `MLP_TRAIN_SUBSET=None`，用全部历史数据、不做子采样 |

> 提示：仅把 bootstrap 关掉而子集仍为 60，离线 CV 显示会令 σ 塌缩（成员重叠过高）；如要无放回，建议同时 `--subset=40` 维持成员分歧。

### 输出与 RF 并存（不会互相覆盖）

MLP 模式下，结果文件名追加网络结构标签 `__<tag>`（`tag` 由 `_mlp_tag()` 生成，形如 `mlp_h64-64_m10`，编码隐藏层与集成规模），因此 **RF 的既有结果不会被覆盖**。**非默认的子采样设置还会再追加标记，避免实验覆盖默认结果**：`--no-bootstrap` → `_nob`、`--subset=N` → `_sN`、`--subset=none` → `_sfull`（例：`--no-bootstrap --subset=40` 得 `mlp_h64-64_m10_nob_s40`）：

| 产物 | RF（默认） | MLP |
|------|-----------|-----|
| 标定 JSON / 汇总 | `{场景}_calibration.json` 等（无后缀） | 追加 `__mlp_h64-64_m10` 后缀 |
| 训练好的模型 | （RF 不落盘） | `data/processed_data/calibration/mlp_models/{场景}_mlp_h64-64_m10.joblib`（含归一化参数与各成员权重，便于复现/适配） |
| 网络超参记录 | 结果 JSON 中 `surrogate_config = null` | 结果 JSON 中 `surrogate_config` 记录隐藏层/集成规模/激活/求解器/σ 校准系数/子采样大小（`train_subset`）/重采样开关（`bootstrap`）/续训开关（`warm_start`）等 |

### MLP 集成配置（`unified_calibration.py` 顶部常量，可按需调整）

下表所列默认值经**日志离线交叉验证**标定（在已评估点上做 RepeatedKFold，比较 RMSE / R² / σ 校准），是当前数据支撑的最优组合：

| 项 | 默认值 | 说明 |
|----|--------|------|
| 实现 | `sklearn.neural_network.MLPRegressor` 的 **深度集成**（`MLPEnsemble`，`_fit_mlp()` 构建） | $M$ 个独立 MLP，$\sigma$ 取成员间分歧 |
| `MLP_HIDDEN_LAYERS` | `(64, 64)` | 2 隐藏层 × 64 神经元（CV 显示优于 `(32,32)`）|
| `MLP_ENSEMBLE_SIZE` | `10` | 集成成员数；由 5→10 使留出 RMSE 降约 6%、σ 校准 95% 覆盖率 0.85→0.91，再加到 15 收益递减 |
| `MLP_ACTIVATION` / `MLP_SOLVER` | `tanh` / `lbfgs` | 小样本回归收敛快、曲面平滑 |
| `MLP_ALPHA` / `MLP_MAX_ITER` | `1e-3` / `2000` | L2 正则强度 / 最大迭代 |
| `MLP_BOOTSTRAP` | `True` | 成员间有放回重采样以增加分歧（类比 RF 的 bagging）；CV 证实**关掉而子集仍为 60 会令 σ 塌缩、严重过自信**。可用命令行 `--bootstrap` / `--no-bootstrap` 覆盖 |
| `MLP_TRAIN_SUBSET` | `N_INIT`（=60） | 每个成员训练子集大小（抗过拟合子采样）；阶段 B 数据增至 100 时单网络仍只见 ≤60 点。可用命令行 `--subset=<N\|none>` 覆盖（`none` = 用全部数据） |
| `MLP_WARM_START` | `True` | 阶段 B 续训：每轮在上一轮成员权重上继续训练（`MLPEnsemble.update`），而非从零重训；`False` 则恢复每轮从零训练 |
| `MLP_SIGMA_CALIB` | `1.2` | σ 校准放大系数，在 `_mlp_lcb()` 中按此放大成员间标准差以修正过自信，使 LCB 探索更充分 |
| 随机种子 | **运行时随机抽取**（`np.random.default_rng()`，记于 `MLPEnsemble.seeds`） | 每个子 MLP 用不同的随机种子初始化网络参数，保证集成内/场景间初始化多样性 |
| 采集函数 | `_mlp_lcb()` → `_pick_lcb_point()` | 成员均值 $\mu$ 与（经 `MLP_SIGMA_CALIB` 校准的）成员标准差 $\sigma$ 给出 $\text{LCB}=\mu-\kappa\sigma$（`LC_KAPPA=1.96`），与 RF 块相同的选点接口但各自独立实现 |

> 提示：MLP 各子网络在初始化时使用**随机种子**，因此每次运行得到的网络互不相同（非完全可复现）；实际抽到的种子记录在 `MLPEnsemble.seeds` 中以便追溯。单次运行涉及 10 个子网络 × Phase B 41 轮 = 410 次子网络拟合：首轮 10 次为从零训练，其余 400 次为 warm-start 续训（在已有权重上继续优化）。SUMO 仿真仍为 101 次，是真正的耗时瓶颈。
