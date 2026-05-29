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
