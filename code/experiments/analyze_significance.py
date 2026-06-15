"""analyze_significance.py — 跨场景配对 Wilcoxon 检验（读已有 CSV，不跑 SUMO）

对每个 seed，在六个场景上对 BF-SAC-RF 与各基线的 error_at_budget 做配对 Wilcoxon
符号秩检验（单侧：BF-SAC 更小更好）。输出 per-seed 明细与汇总表。

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
REF = "BF-SAC-RF"
BASELINES = [m for m in METHOD_ORDER if m not in (REF, "BF-SAC-MLP")]
ALPHA = 0.05


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


def _paired_vectors(
    df: pd.DataFrame, seed: int, baseline: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    ref_vals, base_vals = [], []
    wins = 0
    for scene in SCENE_ORDER:
        r = df[(df["scene"] == scene) & (df["method_key"] == REF) & (df["seed"] == seed)]
        b = df[(df["scene"] == scene) & (df["method_key"] == baseline) & (df["seed"] == seed)]
        if r.empty or b.empty:
            continue
        rv, bv = float(r.iloc[0]["error_at_budget"]), float(b.iloc[0]["error_at_budget"])
        ref_vals.append(rv)
        base_vals.append(bv)
        if rv < bv - 1e-9:
            wins += 1
    return np.asarray(ref_vals), np.asarray(base_vals), wins


def run_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict] = []
    for seed in REPLICATE_SEEDS:
        for baseline in BASELINES:
            ref_v, base_v, wins = _paired_vectors(df, seed, baseline)
            n = len(ref_v)
            if n < 3:
                continue
            diffs = ref_v - base_v
            if np.allclose(diffs, 0):
                stat, p = 0.0, 1.0
            else:
                try:
                    stat, p = wilcoxon(ref_v, base_v, alternative="less", zero_method="wilcox")
                except ValueError:
                    stat, p = float("nan"), float("nan")
            detail_rows.append({
                "reference": REF,
                "baseline": baseline,
                "seed": int(seed),
                "n_scenes": n,
                "scenes_won": wins,
                "mean_diff": float(np.mean(ref_v - base_v)),
                "wilcoxon_stat": stat,
                "p_value": p,
                "significant_005": bool(p < ALPHA) if np.isfinite(p) else False,
            })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return detail, detail

    summary = (
        detail.groupby("baseline", as_index=False)
        .agg(
            p_median=("p_value", "median"),
            p_min=("p_value", "min"),
            p_max=("p_value", "max"),
            n_seeds_sig=("significant_005", "sum"),
            mean_scenes_won=("scenes_won", "mean"),
            mean_diff=("mean_diff", "mean"),
        )
        .sort_values("p_median")
    )
    summary["n_seeds"] = len(REPLICATE_SEEDS)
    return detail, summary


def main() -> None:
    df = _load_per_run()
    main_df = df[df["method_key"].isin([REF] + BASELINES)]
    detail, summary = run_analysis(main_df)

    RES.mkdir(parents=True, exist_ok=True)
    detail_path = RES / "comparison_significance_detail.csv"
    summary_path = RES / "comparison_significance.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print("Cross-scene paired Wilcoxon (BF-SAC-RF vs baselines, per seed)")
    print("=" * 72)
    if summary.empty:
        print("No results.")
        return

    for _, row in summary.iterrows():
        bl = row["baseline"]
        sig = int(row["n_seeds_sig"])
        print(
            f"  vs {bl:8s}  p median={row['p_median']:.4g}  "
            f"range=[{row['p_min']:.4g}, {row['p_max']:.4g}]  "
            f"sig seeds={sig}/{int(row['n_seeds'])}  "
            f"avg scenes won={row['mean_scenes_won']:.1f}/6"
        )
    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
