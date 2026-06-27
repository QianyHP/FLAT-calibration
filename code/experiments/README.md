# 实验脚本（`code/experiments/`）

与 FLAT 标定、方法对比、消融及论文配图相关的脚本入口。完整复现见 [`docs/REPRODUCE.md`](../../docs/REPRODUCE.md)，结果解读见 [`docs/EXPERIMENT.md`](../../docs/EXPERIMENT.md)。

> **命名说明**：对外方法名 **FLAT-RF / FLAT-MLP**；代码与 `comparison_cache/` 中历史键 **`BF-SAC-RF` / `BF-SAC-MLP`** 与之对应，作图时经 `plot_style.METHOD_LABELS` 显示为 FLAT。

## 一键出图（无需 SUMO）

```bash
python code/experiments/plot_all_figures.py
```

生成四张 release 主对比图（标定收敛总览、方法对比组合、六场景多方法收敛、N_init 消融）→ `outputs/figures/`。

## 论文子图（`paper/Figures/`）

| 脚本 | 输出 |
|------|------|
| `plot_calibration_convergence.py` | `calib_conv_{scene}.pdf`（六场景标定收敛） |
| `plot_n_init_sweep.py` | `n_init_{scene}.pdf`（N_init 消融） |
| `plot_method_comparison.py` | `method_comparison_convergence.pdf`（主结果组合） |
| `plot_speed_distribution_violin.py` | `speed_distribution_violin.pdf`（速度分布 + ρ） |
| `plot_lcb_contour.py` | `lcb_contour.pdf` |
| `plot_landscape_dual.py` | `landscape_dual_XAM-N6.pdf` |

上述脚本同时写入 `outputs/figures/` 便于文档引用。

## 目标地形（rugged vs. smoothed）

| 脚本 | SUMO | 说明 |
|------|------|------|
| `run_landscape_dual.py` | 是 | $20\times20$ 网格扫描 `(accel, tau)`，每点记录 raw 逐秒速度 RMSE 与 $J_b$ |
| `plot_landscape_dual.py` | 否 | 双 3-D 曲面对比；`--gif` 生成旋转动画 |

```bash
python code/experiments/run_landscape_dual.py          # 400 runs → JSON
python code/experiments/plot_landscape_dual.py         # JSON → PDF/PNG
```

默认场景 **XAM-N6**；需先完成该场景标定（`unified_calibration.py`）。

## 扩展预算 / 样本效率

| 脚本 | SUMO | 作用 |
|------|------|------|
| `run_efficiency_extended.py` | 是 | 单任务：基线 @ B=500 |
| `run_efficiency_batch.py` | 是 | 并行补跑 120 任务 → `comparison_cache_ext/`（gitignore） |
| `analyze_efficiency.py` | 否 | 追平 FLAT@100 所需仿真数 → `efficiency_*.csv` |

## 其他常用脚本

| 脚本 | SUMO | 作用 |
|------|------|------|
| `run_comparison.py` | 是 | 主对比 / N_init 消融（`--mode main/sweep`） |
| `run_batch_parallel.py` | 是 | 并行补跑缺失 cache |
| `aggregate_multiseed.py` | 否 | 多 seed 聚合 CSV |
| `analyze_significance.py` | 否 | Wilcoxon 显著性 |
| `check_sumo_scenes.py` | 是 | 环境自检 |
| `plot_trajectory_compare.py` | 是* | 轨迹对比示意（需 SUMO 重仿真） |
| `export_fig1_teaser_assets.py` | 否 | 可选 poster 素材（非 ICTAI 正文） |

共享模块：`experiment_io.py`（cache 路径）、`plot_style.py`（配色与 paper panel 导出）、`plot_convergence_utils.py`。
