# 实验脚本说明

本目录包含 **对比/消融实验**、**作图** 与 **SUMO 自检** 脚本。标定主程序在 [`../calibration/unified_calibration.py`](../calibration/unified_calibration.py)。

## 脚本一览

| 脚本 | 需要 SUMO | 作用 |
|------|-----------|------|
| `run_comparison.py` | 是 | 六场景 × 五方法公平对比（预算 101） |
| `plot_all_figures.py` | 否 | 一键生成三张 release 图 + `ablation_summary.csv` |
| `plot_calibration_convergence.py` | 否 | 六场景 BF-SAC 标定收敛图 |
| `plot_method_comparison.py` | 否 | 方法对比组合图（六场景平均收敛 + 均值柱图） |
| `plot_multiscene_convergence.py` | 否 | 六场景 × 五方法收敛 2×3 面板 |
| `check_sumo_scenes.py` | 是 | 检查数据与单次 SUMO 是否可跑 |

## 对比实验

```bash
# 单场景试跑
python code/experiments/run_comparison.py --scenes XAM-N6 --methods BF-SAC,No-RF

# 全部场景与方法
python code/experiments/run_comparison.py

# 使用已有缓存（不重复仿真）
python code/experiments/run_comparison.py --resume
```

方法键：`BF-SAC`、`No-RF`、`No-LHS`、`SPSA`、`GA`（默认全部）。

缓存路径：`outputs/results/comparison_cache/{场景}_{方法}_b101.json`。

## 作图

```bash
python code/experiments/plot_all_figures.py
```

依赖已提交的 `outputs/results/comparison_summary.csv`、`comparison_convergence.csv` 及标定/缓存 JSON。

## SUMO 自检

```bash
python code/experiments/check_sumo_scenes.py
```

对六个场景依次检查：数据目录、`sumocfg`、`rou.xml`、真实指纹、一次中点参数的 SUMO 评估。全部通过则退出码 0。

## 输入输出

| 类型 | 路径 |
|------|------|
| 读入 | `outputs/results/comparison_*.csv`、`data/processed_data/calibration/` |
| 写出图 | `outputs/figures/*.png`（本地生成，默认 gitignore） |
| 写出表 | `outputs/results/ablation_summary.csv` |

完整协议见 [`docs/EXPERIMENT.md`](../../docs/EXPERIMENT.md)。
