"""plot_all_figures.py — 一键生成 FLAT release 主对比图

顺序调用作图脚本的 main()，无需 SUMO。
输出（outputs/figures/）:
  bfsac_calibration_convergence.*   — 六场景标定收敛总览（历史文件名前缀）
  method_comparison_panel.*
  multiscene_method_convergence.*
  n_init_sample_efficiency.*        — 若存在 sweep CSV
  comparison_significance.csv         — Wilcoxon 显著性

论文子图（calib_conv_*, n_init_*, speed_distribution_violin 等）请单独运行
code/experiments/README.md 中列出的脚本 → paper/Figures/
"""
from __future__ import annotations

from pathlib import Path

from plot_calibration_convergence import main as plot_calibration
from plot_method_comparison import main as plot_comparison
from plot_multiscene_convergence import main as plot_multiscene

PROJ = Path(__file__).resolve().parents[2]
SWEEP_CSV = PROJ / "outputs" / "results" / "comparison_summary_sweep.csv"


def main() -> None:
    print("=" * 72)
    print("FLAT release figures")
    print("=" * 72)

    print("\n[1/6] Multiseed aggregate …")
    try:
        from aggregate_multiseed import main as agg_ms
        agg_ms()
    except Exception as e:
        print(f"  skip: {e}")

    print("\n[2/6] Significance (Wilcoxon) …")
    try:
        from analyze_significance import main as sig
        sig()
    except Exception as e:
        print(f"  skip: {e}")

    print("\n[3/6] Calibration convergence …")
    try:
        plot_calibration()
    except FileNotFoundError as e:
        print(f"  skip: {e}")

    print("\n[4/6] Method comparison panel …")
    plot_comparison()

    print("\n[5/6] Multiscene convergence …")
    plot_multiscene()

    print("\n[6/6] N_init sample-efficiency …")
    if SWEEP_CSV.exists():
        from plot_n_init_sweep import main as plot_sweep
        plot_sweep()
    else:
        print("  skip (no comparison_summary_sweep.csv)")

    print("\n" + "=" * 72)
    print("Done. See outputs/figures/ (PNG + PDF + SVG)")
    print("=" * 72)


if __name__ == "__main__":
    main()
