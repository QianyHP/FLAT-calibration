"""analyze_efficiency.py
Computes sample-efficiency ratios: how many simulations does each baseline need
to match FLAT-RF / FLAT-MLP performance at budget 100?

Threshold: per-scene mean J_b of FLAT-RF (or FLAT-MLP) across 5 seeds at B=100.
Analysis unit: each of the 6 scenes × 5 seeds = 30 (scene, seed) pairs.

Outputs:
  outputs/results/efficiency_per_run.csv   -- per (scene, seed, method) rows
  outputs/results/efficiency_summary.csv   -- mean ± std per method (Table III)

Usage:
  python code/experiments/analyze_efficiency.py
  python code/experiments/analyze_efficiency.py --latex   # print LaTeX table row
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJ / "code" / "experiments"
CAL_ROOT = PROJ / "code" / "calibration"
for _p in (EXP_ROOT, CAL_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_io import REPLICATE_SEEDS  # noqa: E402
from run_efficiency_extended import (  # noqa: E402
    BASELINES_EXT,
    BUDGET_EXT,
    CACHE_EXT_DIR,
    SCENE_ORDER,
    _cache_path_ext,
)

RES = PROJ / "outputs" / "results"
CACHE_MAIN_DIR = RES / "comparison_cache"
BUDGET_FLAT = 100


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_flat_thresholds() -> dict[str, dict[str, float]]:
    """Return per-scene mean J_b of FLAT-RF and FLAT-MLP at B=100 across seeds."""
    from experiment_io import cache_path as main_cache_path

    thresholds: dict[str, dict[str, float]] = {"FLAT-RF": {}, "FLAT-MLP": {}}
    for variant, mkey in [("FLAT-RF", "BF-SAC-RF"), ("FLAT-MLP", "BF-SAC-MLP")]:
        for scene in SCENE_ORDER:
            vals = []
            for seed in REPLICATE_SEEDS:
                p = main_cache_path(scene, mkey, BUDGET_FLAT, seed)
                if p.exists():
                    d = json.loads(p.read_text(encoding="utf-8"))
                    vals.append(float(d.get("error_at_budget", d.get("final_error", 10.0))))
            if vals:
                thresholds[variant][scene] = float(np.mean(vals))
            else:
                print(f"[!] No FLAT cache for {variant} scene={scene}", flush=True)
    return thresholds


def _first_budget_to_match(
    conv_sims: list[int],
    conv_best: list[float],
    threshold: float,
    cap: int = BUDGET_EXT,
) -> int:
    """Return the first sim index where best_so_far <= threshold, else cap+1."""
    for s, b in zip(conv_sims, conv_best):
        if b <= threshold:
            return int(s)
    return cap + 1  # never reached within budget


# ── main analysis ──────────────────────────────────────────────────────────────
def compute_efficiency(
    flat_variant: str,
    thresholds: dict[str, float],
    cap: int = BUDGET_EXT,
) -> pd.DataFrame:
    """For a given FLAT reference (RF or MLP), compute budget-to-match per run."""
    rows = []
    for mkey in BASELINES_EXT:
        for scene in SCENE_ORDER:
            if scene not in thresholds:
                continue
            thresh = thresholds[scene]
            for seed in REPLICATE_SEEDS:
                p = _cache_path_ext(scene, mkey, seed)
                if not p.exists():
                    print(f"  [missing] {scene} {mkey} s={seed}", flush=True)
                    continue
                d = json.loads(p.read_text(encoding="utf-8"))
                conv_sims = d["convergence_sims"]
                conv_best = d["convergence_best"]
                budget_needed = _first_budget_to_match(conv_sims, conv_best, thresh, cap)
                rows.append({
                    "flat_variant": flat_variant,
                    "baseline": mkey,
                    "scene": scene,
                    "seed": seed,
                    "threshold": round(thresh, 4),
                    "budget_needed": budget_needed,
                    "reached": budget_needed <= cap,
                })
    return pd.DataFrame(rows)


def summarize_efficiency(df: pd.DataFrame, cap: int = BUDGET_EXT) -> pd.DataFrame:
    """Aggregate over 30 (scene × seed) pairs: mean ± std, ratio."""
    rows = []
    for baseline in BASELINES_EXT:
        sub = df[df["baseline"] == baseline]
        if sub.empty:
            continue
        n_total = len(sub)
        n_reached = int(sub["reached"].sum())
        n_unreached = n_total - n_reached

        # Use cap+1 for unreached runs in mean (conservative)
        vals = sub["budget_needed"].values.astype(float)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        # Ratio = mean_budget / BUDGET_FLAT
        ratio = mean_val / BUDGET_FLAT

        rows.append({
            "baseline": baseline,
            "n_pairs": n_total,
            "n_reached": n_reached,
            "n_unreached": n_unreached,
            "budget_mean": round(mean_val, 1),
            "budget_std": round(std_val, 1),
            "ratio": round(ratio, 2),
        })
    return pd.DataFrame(rows)


def print_latex_table(
    summary_rf: pd.DataFrame,
    summary_mlp: pd.DataFrame,
    cap: int = BUDGET_EXT,
) -> None:
    """Print a LaTeX tabular snippet for the paper."""
    order = ["TPE", "GA", "CMA-ES", "SPSA"]
    print()
    print("% Table III: Sample-efficiency ratios")
    print(r"\begin{table}[t]")
    print(r"  \caption{Simulations needed by each baseline to match FLAT@100 "
          r"(mean\,$\pm$\,std over $30$ scene$\times$seed pairs; "
          rf"cap\,${cap}$; "
          r"$\dagger$ $>$half pairs unreached).}")
    print(r"  \label{tab:efficiency}")
    print(r"  \centering\footnotesize\renewcommand{\arraystretch}{1.15}")
    print(r"  \begin{tabularx}{\columnwidth}{@{}l C C C C@{}}")
    print(r"    \toprule")
    print(r"    & \multicolumn{2}{c}{vs.\ FLAT-RF} & \multicolumn{2}{c}{vs.\ FLAT-MLP} \\")
    print(r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    print(r"    Baseline & Sims$\downarrow$ & Ratio & Sims$\downarrow$ & Ratio \\")
    print(r"    \midrule")

    rf_d = summary_rf.set_index("baseline")
    mlp_d = summary_mlp.set_index("baseline")

    for mkey in order:
        if mkey not in rf_d.index or mkey not in mlp_d.index:
            continue
        r = rf_d.loc[mkey]
        m = mlp_d.loc[mkey]

        def _fmt(row: pd.Series, c: int) -> tuple[str, str]:
            dagger = r"$\dagger$" if row["n_unreached"] > row["n_pairs"] // 2 else ""
            if row["budget_mean"] > c:
                sims_str = rf"$>{c}${dagger}"
                ratio_str = rf"$>{c/BUDGET_FLAT:.1f}\times$"
            else:
                sims_str = rf"{row['budget_mean']:.0f}\,$\pm$\,{row['budget_std']:.0f}{dagger}"
                ratio_str = rf"{row['ratio']:.2f}$\times$"
            return sims_str, ratio_str

        rs, rr = _fmt(r, cap)
        ms, mr = _fmt(m, cap)
        print(f"    {mkey:<8} & {rs} & {rr} & {ms} & {mr} \\\\")

    print(r"    \bottomrule")
    print(r"  \end{tabularx}")
    print(r"\end{table}")
    print()


# ── entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true",
                        help="Also print LaTeX table snippet")
    parser.add_argument("--cap", type=int, default=BUDGET_EXT)
    args = parser.parse_args()

    print("Loading FLAT@100 thresholds …")
    all_thresholds = _load_flat_thresholds()

    # Check how many extended cache files exist
    n_exist = sum(
        1 for scene in SCENE_ORDER
        for mkey in BASELINES_EXT
        for seed in REPLICATE_SEEDS
        if _cache_path_ext(scene, mkey, seed).exists()
    )
    n_total = len(SCENE_ORDER) * len(BASELINES_EXT) * len(REPLICATE_SEEDS)
    print(f"Extended cache: {n_exist}/{n_total} files found")

    if n_exist == 0:
        print("[!] No extended cache files found.")
        print("    Run: python code/experiments/run_efficiency_batch.py --workers 12")
        return

    all_rows: list[pd.DataFrame] = []
    summaries_rf: pd.DataFrame | None = None
    summaries_mlp: pd.DataFrame | None = None

    for variant in ("FLAT-RF", "FLAT-MLP"):
        thresh = all_thresholds[variant]
        if not thresh:
            print(f"[!] No thresholds for {variant}, skip.")
            continue
        df = compute_efficiency(variant, thresh, args.cap)
        if df.empty:
            continue
        df.to_csv(RES / f"efficiency_per_run_{variant.replace('-','_')}.csv",
                  index=False, encoding="utf-8-sig")
        summary = summarize_efficiency(df, args.cap)
        out = RES / f"efficiency_summary_{variant.replace('-','_')}.csv"
        summary.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n=== {variant} reference thresholds ===")
        for scene, t in sorted(thresh.items()):
            print(f"  {scene:<12}: J_b threshold = {t:.4f}")
        print(f"\n=== Efficiency summary vs. {variant} ===")
        print(summary.to_string(index=False))
        print(f"Saved: {out}")
        if variant == "FLAT-RF":
            summaries_rf = summary
        else:
            summaries_mlp = summary
        all_rows.append(df)

    # Combined per-run file
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_combined = RES / "efficiency_per_run.csv"
        combined.to_csv(out_combined, index=False, encoding="utf-8-sig")
        print(f"\nCombined per-run saved: {out_combined}")

    if args.latex and summaries_rf is not None and summaries_mlp is not None:
        print_latex_table(summaries_rf, summaries_mlp, args.cap)


if __name__ == "__main__":
    main()
