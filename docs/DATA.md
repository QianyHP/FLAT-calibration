# 数据准备说明

BF-SAC 标定与对比实验所需的原始轨迹、SUMO 场景配置及标定结果均存于 `data/`。大体积原始 CSV / 仿真输出默认由 `.gitignore` 排除，请按下列步骤在本地准备。复现流程见 [REPRODUCE.md](REPRODUCE.md)。

## 目录结构

| 路径 | 内容 |
|------|------|
| `data/raw_data/SIND/{城市}/` | 交叉口轨迹、`map.osm`、`sumo/*.xml`（天津、长春、西安） |
| `data/raw_data/UTE/{场景}/` | 快速路 `frenet.csv`、`sumo/*.xml`（YTDJ、RML、XAM-N6） |
| `data/processed_data/calibration/` | BF-SAC 标定 JSON、`calibration_summary.csv` |
| `data/processed_data/all_cities_od_stats.csv` | 预处理附属统计（可选） |
| `data/external_data/SinD/` | SinD 仓库克隆（含西安等发布包） |

上游镜像（可选，不提交 git）：`_local_archive/data_upstream/SinD_Data/` — 三城原始记录与 `map.osm`。

## 六场景与代码键名

| 场景键 | 类型 | 数据目录 |
|--------|------|----------|
| `Tianjin` | SIND 交叉口 | `data/raw_data/SIND/Tianjin/` |
| `Changchun` | SIND 交叉口 | `data/raw_data/SIND/Changchun/` |
| `Xian` | SIND 交叉口 | `data/raw_data/SIND/Xian/` |
| `YTDJ` | UTE 快速路 | `data/raw_data/UTE/YTDJ/` |
| `RML` | UTE 快速路 | `data/raw_data/UTE/RML/` |
| `XAM-N6` | UTE 快速路 | `data/raw_data/UTE/XAM-N6/` |

标定脚本读取：

- SIND：`veh_tracks.csv`（速度 / 加速度分量）
- UTE：`frenet.csv`（速度、纵向加速度列）
- SUMO：`sumo/{cfg}.sumocfg` 与同目录 `*.rou.xml`

## 首次准备 SIND 数据

若仅有 SinD 上游仓库，可先还原 `Data/` 到 `data/external_data/SinD/Data/`，或把原始包放到 `_local_archive/data_upstream/SinD_Data/`，再执行：

```bash
python code/preprocessing/organize_sind.py
python code/preprocessing/organize_sind.py Xian
python code/preprocessing/build_all_sumo.py
python code/preprocessing/build_all_sumo.py Xian
python code/preprocessing/clean_static_vehicle_tracks.py
```

`build_all_sumo.py` 会调用 SUMO 的 `netconvert` 等工具生成路网与路由；需保证 `netconvert` 在 `PATH` 中。

## 首次准备 UTE 数据

快速路场景需已有 `frenet.csv` 与场景几何信息，再生成 SUMO 配置：

```bash
python code/preprocessing/build_ytdj_sumo.py
python code/preprocessing/build_rml_sumo.py
python code/preprocessing/build_xam_sumo.py
```

## 标定结果（处理后）

运行 `python code/calibration/unified_calibration.py` 后写入 `data/processed_data/calibration/`：

| 文件 | 说明 |
|------|------|
| `{场景}_calibration.json` | 精简标定结果（无 `doe_history`） |
| `{场景}_calibration_with_history.json` | 含逐次仿真收敛历史 |
| `calibration_summary.csv` | 六场景汇总 |
| `calibration_all.json` | 合并 JSON |

## 对比实验结果（release 数据包）

见 `outputs/results/`（由 `run_comparison.py` 生成，本仓库 release 已提交部分结果）：

- `comparison_summary.csv` / `comparison_convergence.csv`
- `comparison_cache/{场景}_{方法}_b100.json`

## 检查数据是否就绪

```bash
python code/experiments/check_sumo_scenes.py
```

通过即表示各场景目录、`sumocfg`、真实指纹与单次 SUMO 试跑均正常。
