#!/usr/bin/env bash
# Full reproduction pipeline (requires SUMO). See docs/REPRODUCE.md §3.2.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [0/7] Environment check ==="
python code/experiments/check_sumo_scenes.py

echo "=== [1/7] RF calibration (6 scenes) ==="
python code/calibration/unified_calibration.py

echo "=== [2/7] MLP calibration (6 scenes) ==="
python code/calibration/unified_calibration.py --mlp

echo "=== [3/7] Main comparison (parallel) ==="
python code/experiments/run_batch_parallel.py --workers 12 --phases main

echo "=== [4/7] N_init sweep (parallel) ==="
python code/experiments/run_batch_parallel.py --workers 12 --phases sweep

echo "=== [5/7] Rebuild CSV + multiseed aggregate ==="
python code/experiments/run_comparison.py --rebuild-csv --mode main
python code/experiments/run_comparison.py --rebuild-csv --mode sweep
python code/experiments/aggregate_multiseed.py --mode main

echo "=== [6/7] Significance analysis ==="
python code/experiments/analyze_significance.py

echo "=== [7/7] Figures (PNG + PDF + SVG) ==="
python code/experiments/plot_all_figures.py

echo "=== Done ==="
