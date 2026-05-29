# 标定模块说明

## 主程序

`unified_calibration.py` — **BF-SAC**（行为指纹 + 随机森林代理 + 序贯 LCB），固定 **101 次 SUMO** 预算。

```bash
# 六场景（耗时长）
python code/calibration/unified_calibration.py

# 单场景
python code/calibration/unified_calibration.py Tianjin
python code/calibration/unified_calibration.py XAM-N6
```

## 流程概要

1. 从 `data/raw_data/` 读取真实轨迹，计算 8 维行为指纹。
2. **阶段 A**：60 次 LHS，每次一次 SUMO，训练/更新 RF 代理。
3. **阶段 B**：41 次序贯加点（信赖域候选 + LCB 选点 + 真实 SUMO 评估）。
4. 写出 JSON/CSV，并将最优参数写回 `sumo/*.rou.xml`。

## 场景配置

场景键与数据路径在 `SCENARIOS` 字典中定义（`Tianjin`、`Changchun`、`Xian`、`YTDJ`、`RML`、`XAM-N6`）。每场景对应 `raw_data/.../sumo/{cfg}.sumocfg`。

## 输出

| 文件 | 说明 |
|------|------|
| `data/processed_data/calibration/{场景}_calibration.json` | 无历史精简版 |
| `data/processed_data/calibration/{场景}_calibration_with_history.json` | 含 `doe_history` |
| `data/processed_data/calibration/calibration_summary.csv` | 汇总 |
| `data/processed_data/calibration/calibration_all.json` | 合并 |

## 依赖

- Python：`numpy`、`pandas`、`scipy`、`scikit-learn`
- 系统：SUMO + TraCI（`sumo` 在 `PATH`）

运行前建议：

```bash
python code/experiments/check_sumo_scenes.py
```

实验协议见 [`docs/EXPERIMENT.md`](../../docs/EXPERIMENT.md)。
