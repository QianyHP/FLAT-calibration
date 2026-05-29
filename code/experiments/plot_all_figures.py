"""plot_all_figures.py — 一键生成全部 release 图表

顺序调用三个作图脚本的 main()，无需 SUMO。
输出:
  outputs/figures/bfsac_calibration_convergence.png
  outputs/figures/method_comparison_panel.png
  outputs/figures/multiscene_method_convergence.png
"""
from __future__ import annotations

from plot_calibration_convergence import main as plot_calibration
from plot_method_comparison import main as plot_comparison
from plot_multiscene_convergence import main as plot_multiscene


def main() -> None:
    print("=" * 72)
    print("BF-SAC release figures (all)")
    print("=" * 72)

    print("\n[1/3] Calibration convergence …")
    plot_calibration()

    print("\n[2/3] Method comparison panel …")
    plot_comparison()

    print("\n[3/3] Multiscene convergence …")
    plot_multiscene()

    print("\n" + "=" * 72)
    print("Done. See outputs/figures/")
    print("=" * 72)


if __name__ == "__main__":
    main()
