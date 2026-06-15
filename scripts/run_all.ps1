# Full reproduction pipeline (requires SUMO). See docs/REPRODUCE.md §3.2.
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

Write-Host "=== [0/7] Environment check ==="
python code/experiments/check_sumo_scenes.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [1/7] RF calibration (6 scenes) ==="
python code/calibration/unified_calibration.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [2/7] MLP calibration (6 scenes) ==="
python code/calibration/unified_calibration.py --mlp
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [3/7] Main comparison (parallel) ==="
python code/experiments/run_batch_parallel.py --workers 12 --phases main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [4/7] N_init sweep (parallel) ==="
python code/experiments/run_batch_parallel.py --workers 12 --phases sweep
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [5/7] Rebuild CSV + multiseed aggregate ==="
python code/experiments/run_comparison.py --rebuild-csv --mode main
python code/experiments/run_comparison.py --rebuild-csv --mode sweep
python code/experiments/aggregate_multiseed.py --mode main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [6/7] Significance analysis ==="
python code/experiments/analyze_significance.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [7/7] Figures (PNG + PDF + SVG) ==="
python code/experiments/plot_all_figures.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Done ==="
