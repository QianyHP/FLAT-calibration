# 复现指南

从环境配置到全部图表的完整复现流程、实验矩阵与规模，以及可直接在算力机上运行的脚本。
结果解读见 [EXPERIMENT.md](EXPERIMENT.md)，数据准备见 [DATA.md](DATA.md)。

---

## 1. 环境

- Python 3.10+：`pip install -r requirements.txt`（含 `scikit-learn`、基线所需 `cma` / `optuna`）
- [Eclipse SUMO](https://eclipse.dev/sumo/)：`sumo` 与 `netconvert` 在 `PATH` 中；TraCI 随 SUMO 安装
- 自检：`python code/experiments/check_sumo_scenes.py`（六场景配置 + 单次 SUMO 试跑，退出码 0 即就绪）

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux: source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. 无需 SUMO：查看或重绘图表

仓库已提交 **对比 cache** 与 **四张 release 图**（PNG / PDF / SVG），合作者克隆即可查看你本地的实验结果，无需重跑 SUMO。需要重绘时：

```bash
python code/experiments/plot_all_figures.py
```

生成 `outputs/figures/` 下四张主图：标定收敛（图 1）、方法对比组合（图 2）、六场景多方法收敛（图 3）、N_init 样本效率（图 4）。

---

## 3. 完整复现（需要 SUMO）

### 3.1 实验总览与规模

统一预算 **100 次 SUMO / run**（40 LHS + 60 序贯 LCB，κ 2.0→0.5 退火）；多 seed 取 `42, 101, 202, 303, 404`。

| 阶段 | 命令 | 规模 | 产物 |
|------|------|------|------|
| 标定 | `unified_calibration.py`（RF + `--mlp`） | 6 场景 × 2 代理 × 100 ≈ **1,200** 次 SUMO | 图 1、表 1、`calibration_summary.csv` |
| 主对比 | `run_batch_parallel.py --phases main` | 6 方法 × 6 场景 × 5 seed × 100 = **18,000** 次 | 图 2 / 3、表 2 / 3 |
| N_init 消融 | `run_batch_parallel.py --phases sweep` | RF/MLP × {20,60,80,100}（n=40 复用主对比）× 6 场景 × 5 seed × 100 = **24,000** 次 | 图 4 |

合计约 **4.3 万次 SUMO 仿真**。三阶段均 **断点续跑**（已有 cache 自动跳过），可 12 路并行。
主对比六方法：`BF-SAC-RF`、`BF-SAC-MLP`、`SPSA`、`GA`、`CMA-ES`、`TPE`，覆盖代理序贯、局部梯度、种群进化、演化策略、贝叶斯密度五类机制。

### 3.2 一键脚本（算力机依次运行）

```bash
# 0) 环境自检
python code/experiments/check_sumo_scenes.py

# 1) BF-SAC 标定（图 1 + 表 1）：6 场景 × RF / MLP
python code/calibration/unified_calibration.py            # 全场景，RF
python code/calibration/unified_calibration.py --mlp      # 全场景，MLP 代理

# 2) 主对比 + N_init 消融（图 2/3/4 + 表 2/3）：12 路并行、断点续跑
python code/experiments/run_batch_parallel.py --workers 12 --phases main
python code/experiments/run_batch_parallel.py --workers 12 --phases sweep

# 3) 重建汇总 CSV + 多 seed 聚合 + 出图
python code/experiments/run_comparison.py --rebuild-csv --mode main
python code/experiments/run_comparison.py --rebuild-csv --mode sweep
python code/experiments/aggregate_multiseed.py --mode main
python code/experiments/plot_all_figures.py
```

把上述命令存为仓库内脚本即可一次跑完：

```bash
# Linux / macOS
bash scripts/run_all.sh
# Windows PowerShell
.\scripts\run_all.ps1
```

### 3.3 单独 / 并行运行

各方法的 cache 路径互不重叠，可在多终端按 `--methods` / `--scenes` 拆分：

```bash
# 单场景、多 seed、主对比
python code/experiments/run_comparison.py --scenes Tianjin --mode main --seeds 42,101,202,303,404 --resume
# 单场景、N_init 消融
python code/experiments/run_comparison.py --scenes Tianjin --mode sweep --seeds 42,101,202,303,404 --resume
```

缓存：`{场景}_{方法}_b100.json`（seed=42）、`{场景}_{方法}_b100_s{seed}.json`（其余 seed）。

---

## 4. 脚本一览

| 脚本 | 需要 SUMO | 作用 |
|------|-----------|------|
| `code/calibration/unified_calibration.py` | 是 | BF-SAC 标定主程序（RF / MLP 代理） |
| `code/experiments/run_comparison.py` | 是 | 主对比（6 方法）+ N_init 消融（`--mode main/sweep`），支持 `--seeds` |
| `code/experiments/run_batch_parallel.py` | 是 | 并行补跑缺失 cache（`--workers N --phases main,sweep`） |
| `code/experiments/check_sumo_scenes.py` | 是 | 数据 + 单次 SUMO 自检 |
| `code/experiments/experiment_io.py` | 否 | 多 seed cache 命名、解析与聚合 |
| `code/experiments/aggregate_multiseed.py` | 否 | 生成 `comparison_multiseed_*.csv`（置信带/误差棒数据） |
| `code/experiments/analyze_significance.py` | 否 | 跨场景配对 Wilcoxon → `comparison_significance.csv` |
| `code/experiments/plot_all_figures.py` | 否 | 一键生成四张主图（PNG + PDF + SVG）+ 显著性表 |
| `code/experiments/plot_calibration_convergence.py` | 否 | 图 1 |
| `code/experiments/plot_method_comparison.py` | 否 | 图 2 |
| `code/experiments/plot_multiscene_convergence.py` | 否 | 图 3 |
| `code/experiments/plot_n_init_sweep.py` | 否 | 图 4 |
| `code/experiments/plot_convergence_utils.py` | 否 | 作图公用：多 seed 均值 ± std 置信带 |

预处理脚本见 [DATA.md](DATA.md)。

---

## 5. 标定主程序与代理配置

`unified_calibration.py` 实现 BF-SAC（行为指纹 + 代理 + 序贯 LCB），代理二选一、接口一致：

| 代理 | 启用方式 | 特点 |
|------|----------|------|
| **RF**（默认） | `python code/calibration/unified_calibration.py [场景]` | 500 树，轻量、对小样本稳健 |
| **MLP 深度集成** | 加 `--mlp` 或 `BFSAC_SURROGATE=mlp` | 10 个 `(64,64)` 网络，容量更高 |

MLP 结果文件追加 `__mlp_h64-64_m10` 后缀、与 RF 并存。默认开启 bootstrap + 子采样以维持集成成员分歧——这是不确定度 σ 不塌缩、LCB 探索有效的前提。

### MLP 超参（仅 MLP 模式有意义；CLI 覆盖同名环境变量）

| 命令行 | 环境变量 | 含义（默认） |
|--------|----------|--------------|
| `--ensemble=N` | `BFSAC_MLP_ENSEMBLE_SIZE` | 集成成员数（10） |
| `--hidden=A,B` | `BFSAC_MLP_HIDDEN` | 隐藏层（64,64） |
| `--alpha=x` | `BFSAC_MLP_ALPHA` | L2 正则（1e-3） |
| `--sigma-calib=x` | `BFSAC_MLP_SIGMA_CALIB` | σ 放大系数（1.2） |
| `--bootstrap` / `--no-bootstrap` | `BFSAC_MLP_BOOTSTRAP` | 成员自助重采样（开） |
| `--subset=N` / `--subset=none` | `BFSAC_MLP_TRAIN_SUBSET` | 成员训练子集（40；none=全部） |
| `--warm-start` / `--no-warm-start` | `BFSAC_MLP_WARM_START` | Phase B 续训（开） |

非默认配置会在结果文件名追加紧凑标记（`_nob`/`_s40`/`_sfull` 等），各配置互不覆盖。序贯 LCB 的 κ 退火端点可经 `BFSAC_LC_KAPPA_START` / `BFSAC_LC_KAPPA_END` 调整。

---

## 6. 输出位置

| 类型 | 路径 |
|------|------|
| 标定结果 | `data/processed_data/calibration/{场景}_calibration[_with_history][__mlp_*].json`、`calibration_summary.csv` |
| MLP 模型 | `data/processed_data/calibration/mlp_models/*.joblib` |
| 对比缓存 | `outputs/results/comparison_cache/{场景}_{方法}_b100[_s{seed}].json` |
| 对比汇总 | `outputs/results/comparison_summary[_sweep].csv`、`comparison_convergence[_sweep].csv`、`comparison_multiseed_stats.csv`、`comparison_significance.csv` |
| 图表 | `outputs/figures/*.{png,pdf,svg}`（已提交 release 图；亦可 `plot_all_figures.py` 重绘） |
