"""Compute scene-level exact Wilcoxon comparisons from cached runs.

The five stochastic runs are averaged within each scene before testing, so the
inferential unit is the scene rather than the nested scene-by-seed run.  No
SUMO simulation is executed.

Usage:
    python code/experiments/analyze_significance.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJ = Path(__file__).resolve().parents[2]
RES = PROJ / "outputs" / "results"
INPUT = RES / "comparison_summary_main.csv"
OUTPUT = RES / "comparison_scene_level_significance.csv"

SCENE_ORDER = ["Tianjin", "Changchun", "Xian", "YTDJ", "RML", "XAM-N6"]
REPLICATE_SEEDS = [42, 101, 202, 303, 404]
BFSAC_METHODS = ["BF-SAC-GP", "BF-SAC-RF", "BF-SAC-MLP"]
BASELINES = ["SPSA", "GA", "CMA-ES", "TPE"]
ALPHA = 0.05


def _load_runs() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}.")
    df = pd.read_csv(INPUT)
    seed_column = "run_seed" if "run_seed" in df.columns else "seed"
    required = {"scene", "method_key", seed_column, "error_at_budget"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{INPUT} missing columns: {sorted(missing)}")

    methods = BFSAC_METHODS + BASELINES
    df = df[df["method_key"].isin(methods)].copy()
    df[seed_column] = pd.to_numeric(df[seed_column], errors="raise")
    counts = (
        df.groupby(["scene", "method_key"])[seed_column]
        .nunique()
        .rename("n_seeds")
        .reset_index()
    )
    expected = len(REPLICATE_SEEDS)
    bad = counts[counts["n_seeds"] != expected]
    if len(bad) != len(SCENE_ORDER) * len(methods):
        expected_pairs = pd.MultiIndex.from_product(
            [SCENE_ORDER, methods], names=["scene", "method_key"]
        ).to_frame(index=False)
        bad = expected_pairs.merge(counts, how="left").fillna({"n_seeds": 0})
        bad = bad[bad["n_seeds"] != expected]
    if not bad.empty:
        raise ValueError(
            "Expected five seeds for every scene/method pair; found:\n"
            + bad.to_string(index=False)
        )
    return df


def _scene_means(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["scene", "method_key"], as_index=False)
        .agg(error_at_budget=("error_at_budget", "mean"))
    )


def _compare(scene_df: pd.DataFrame, reference: str, baseline: str) -> dict[str, object]:
    pivot = scene_df.pivot(index="scene", columns="method_key", values="error_at_budget")
    pair = pivot.reindex(SCENE_ORDER)[[reference, baseline]].dropna()
    differences = pair[reference].to_numpy() - pair[baseline].to_numpy()
    differences = differences[np.abs(differences) > 1e-12]
    if len(differences) < 3:
        raise ValueError(f"Too few non-zero scene differences for {reference} vs {baseline}")

    stat, p_value = wilcoxon(differences, alternative="two-sided", method="exact")
    return {
        "reference": reference,
        "baseline": baseline,
        "n_scenes": len(differences),
        "scenes_won": int(np.sum(differences < 0)),
        "win_rate": round(float(np.mean(differences < 0)), 3),
        "mean_diff": float(np.mean(differences)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
        "significant_005": bool(p_value < ALPHA),
        "scenes": ",".join(pair.index.tolist()),
    }


def main() -> None:
    scene_df = _scene_means(_load_runs())
    result = pd.DataFrame(
        [
            _compare(scene_df, reference, baseline)
            for reference in BFSAC_METHODS
            for baseline in BASELINES
        ]
    )
    RES.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("Scene-level exact two-sided Wilcoxon comparisons")
    print("Five seeds aggregated within each of six scenes; no SUMO run.")
    for reference in BFSAC_METHODS:
        print(f"\n[{reference}]")
        for row in result[result["reference"] == reference].itertuples(index=False):
            print(
                f"  vs {row.baseline:8s}  p={row.p_value:.4g}  "
                f"won {row.scenes_won}/{row.n_scenes}  "
                f"mean diff={row.mean_diff:.6f}"
            )
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
