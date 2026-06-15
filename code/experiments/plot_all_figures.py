"""plot_all_figures.py — Generate all release figures (no SUMO required)."""
from __future__ import annotations

from pathlib import Path

from plot_calibration_convergence import main as plot_calibration
from plot_method_comparison import main as plot_comparison
from plot_multiscene_convergence import main as plot_multiscene

PROJ = Path(__file__).resolve().parents[2]
SWEEP_CSV = PROJ / "outputs" / "results" / "comparison_summary_sweep.csv"


def main() -> None:
    print("=" * 72)
    print("BF-SAC release figures")
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
