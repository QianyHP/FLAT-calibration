"""aggregate_multiseed.py — 从 comparison_cache 聚合多 seed 统计量。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJ / "code" / "experiments"
CAL_ROOT = PROJ / "code" / "calibration"
for p in (EXP_ROOT, CAL_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from unified_calibration import BUDGET_SUMO  # noqa: E402
from experiment_io import load_per_run_summary, write_multiseed_outputs  # noqa: E402
from run_comparison import methods_for_mode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="聚合多 seed 实验 cache")
    parser.add_argument("--mode", choices=["main", "sweep", "all"], default="main")
    args = parser.parse_args()

    mkeys = set(methods_for_mode(args.mode))
    per_run = load_per_run_summary(BUDGET_SUMO, mkeys)
    if per_run.empty:
        print("[!] No cache files found.")
        return
    p1, p2, p3 = write_multiseed_outputs(per_run, BUDGET_SUMO)
    print(f"Per-run:   {p1} ({len(per_run)} rows)")
    print(f"By-scene:  {p2}")
    print(f"Global:    {p3}")
    print(per_run.groupby("method_key")["seed"].nunique().to_string())


if __name__ == "__main__":
    main()
