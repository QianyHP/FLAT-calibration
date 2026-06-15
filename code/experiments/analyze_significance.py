"""analyze_significance.py — 主对比配对显著性检验（读已有 CSV，不跑 SUMO）

对每个 BF-SAC 代理（RF / MLP），在 6 场景 × 5 seed = 30 个配对点上，
与各基线的 error_at_budget 做单侧 Wilcoxon 符号秩检验（BF-SAC 更小更好）。

用法:
  python code/experiments/analyze_significance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJ = Path(__file__).resolve().parents[2]
EXP = PROJ / "code" / "experiments"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

from experiment_io import REPLICATE_SEEDS  # noqa: E402
from plot_style import METHOD_ORDER, SCENE_ORDER  # noqa: E402

RES = PROJ / "outputs" / "results"
BFSAC_METHODS = ["BF-SAC-RF", "BF-SAC-MLP"]
BASELINES = [m for m in METHOD_ORDER if m not in BFSAC_METHODS]
ALPHA = 0.05
N_PAIRS = len(SCENE_ORDER) * len(REPLICATE_SEEDS)


def _load_per_run() -> pd.DataFrame:
    p = RES / "comparison_per_run.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run: python code/experiments/run_comparison.py --rebuild-csv --mode main"
        )
    df = pd.read_csv(p)
    need = {"scene", "method_key", "seed", "error_at_budget"}
    if not need.issubset(df.columns):
        raise ValueError(f"comparison_per_run.csv missing columns: {need - set(df.columns)}")
    return df


def _lookup(df: pd.DataFrame, scene: str, method: str, seed: int) -> float | None:
    row = df[(df["scene"] == scene) & (df["method_key"] == method) & (df["seed"] == seed)]
    if row.empty:
        return None
    return float(row.iloc[0]["error_at_budget"])


def _paired_30(
    df: pd.DataFrame, reference: str, baseline: str,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """30 个 (scene, seed) 配对：同场景同 seed 同预算，仅方法不同。"""
    ref_vals, base_vals = [], []
    wins = 0
    for scene in SCENE_ORDER:
        for seed in REPLICATE_SEEDS:
            rv = _lookup(df, scene, reference, seed)
            bv = _lookup(df, scene, baseline, seed)
            if rv is None or bv is None:
                continue
            ref_vals.append(rv)
            base_vals.append(bv)
            if rv < bv - 1e-9:
                wins += 1
    return np.asarray(ref_vals), np.asarray(base_vals), wins, len(ref_vals)


def _wilcoxon_less(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    if np.allclose(x - y, 0):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(x, y, alternative="less", zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        return float("nan"), float("nan")


def run_global(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for reference in BFSAC_METHODS:
        for baseline in BASELINES:
            ref_v, base_v, wins, n = _paired_30(df, reference, baseline)
            stat, p = _wilcoxon_less(ref_v, base_v)
            rows.append({
                "reference": reference,
                "baseline": baseline,
                "n_pairs": n,
                "pairs_won": wins,
                "win_rate": round(wins / n, 3) if n else None,
                "mean_diff": float(np.mean(ref_v - base_v)) if n else None,
                "wilcoxon_stat": stat,
                "p_value": p,
                "significant_005": bool(p < ALPHA) if np.isfinite(p) else False,
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = _load_per_run()
    methods = BFSAC_METHODS + BASELINES
    main_df = df[df["method_key"].isin(methods)]

    global_df = run_global(main_df)

    RES.mkdir(parents=True, exist_ok=True)
    global_path = RES / "comparison_significance.csv"
    global_df.to_csv(global_path, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print(f"Paired Wilcoxon @ {N_PAIRS} scene×seed points (one-sided: BF-SAC < baseline)")
    print("=" * 72)
    for reference in BFSAC_METHODS:
        print(f"\n  [{reference}]")
        sub = global_df[global_df["reference"] == reference]
        for _, row in sub.iterrows():
            sig = "yes" if row["significant_005"] else "no"
            print(
                f"    vs {row['baseline']:8s}  p={row['p_value']:.4g}  "
                f"won {int(row['pairs_won'])}/{int(row['n_pairs'])}  "
                f"sig={sig}"
            )
    print(f"\nSaved: {global_path}")


if __name__ == "__main__":
    main()
