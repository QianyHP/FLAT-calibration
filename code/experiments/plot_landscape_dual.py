#!/usr/bin/env python3
"""
plot_landscape_dual.py — Rugged raw objective vs. smoothed J_b (3-D surfaces).

Reads ``outputs/results/landscape_dual_<scene>.json`` from
``run_landscape_dual.py``, interpolates the 20×20 grid (cubic + light
Gaussian smooth for display only), min-max normalises both surfaces to
[0, 1] for shape comparison, and exports a dual-panel figure.

Usage
-----
  python code/experiments/plot_landscape_dual.py
  python code/experiments/plot_landscape_dual.py --azim 90 --elev 28
  python code/experiments/plot_landscape_dual.py --gif   # rotation GIF

Outputs (default scene XAM-N6, azim=135)
-----------------------------------------
  outputs/figures/landscape_dual_XAM-N6.pdf
  outputs/figures/landscape_dual_XAM-N6.png
  outputs/figures/landscape_rotate_XAM-N6.gif   (--gif only)

No SUMO required.
"""

import argparse, json, os, shutil, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      # noqa
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--azim",  type=float, default=135)
parser.add_argument("--elev",  type=float, default=28)
parser.add_argument("--scene", default="XAM-N6",
                    help="scene name (must match landscape_dual_<scene>.json)")
parser.add_argument("--gif",   action="store_true",
                    help="generate animated GIF rotating around z-axis")
args, _ = parser.parse_known_args()

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size":   12,
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
})

# ── load data ─────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data = json.load(open(os.path.join(ROOT, "outputs", "results",
                                   f"landscape_dual_{args.scene}.json")))

res       = np.array(data["results"])
accel_pts = res[:, 0]
tau_pts   = res[:, 1]
jb_pts    = res[:, 2]
raw_pts   = res[:, 3]

# ── interpolate ───────────────────────────────────────────────────────────────
N_INTERP = 80
accel_grid = np.linspace(data["accel_range"][0], data["accel_range"][1], N_INTERP)
tau_grid   = np.linspace(data["tau_range"][0],   data["tau_range"][1],   N_INTERP)
ACCEL, TAU = np.meshgrid(accel_grid, tau_grid)
pts        = np.column_stack([accel_pts, tau_pts])

def interp(z_pts, smooth=0.6):
    good = z_pts < 9
    Z = griddata(pts[good], z_pts[good], (ACCEL, TAU), method="cubic")
    mask = np.isnan(Z)
    if mask.any():
        Z[mask] = griddata(pts[good], z_pts[good],
                           (ACCEL[mask], TAU[mask]), method="nearest")
    return gaussian_filter(Z, sigma=smooth)

Z_raw = interp(raw_pts, smooth=0.5)
Z_jb  = interp(jb_pts,  smooth=0.5)


# ── single-frame render ───────────────────────────────────────────────────────
def make_frame(azim, elev, figsize=(8.2, 3.6)):
    fig = plt.figure(figsize=figsize, facecolor="white")

    ax_L = fig.add_axes([0.01, 0.04, 0.47, 0.88], projection="3d")
    ax_R = fig.add_axes([0.50, 0.04, 0.47, 0.88], projection="3d")

    CMAP = "RdBu_r"   # blue=low (good), red=high (bad)

    def norm01(Z, pts_z):
        good = pts_z < 9
        lo, hi = pts_z[good].min(), pts_z[good].max()
        return (Z - lo) / (hi - lo + 1e-9)

    Z_raw_n = norm01(Z_raw, raw_pts)
    Z_jb_n  = norm01(Z_jb,  jb_pts)

    panels = [
        (ax_L, Z_raw_n, raw_pts,
         "Rugged raw objective", "(trajectory matching)", False),
        (ax_R, Z_jb_n,  jb_pts,
         "Smoothed objective",   "(fingerprint)", True),
    ]
    FS_AXIS = 12
    FS_TICK = 10.5
    FS_SUB = 12.5
    FS_SUB2 = 12
    # Narrative titles/tagline: serif (axis labels stay sans-serif via rcParams)
    FONT_SERIF = {"fontfamily": "serif"}

    for ax, Z_n, _pts_z, subtitle, subtitle2, show_z_ticks in panels:
        ax.plot_surface(ACCEL, TAU, np.clip(Z_n, 0, 1),
                        cmap=CMAP, linewidth=0, antialiased=True,
                        alpha=0.90, vmin=0, vmax=1)

        ax.set_xlabel(r"$a_{\max}$", fontsize=FS_AXIS, labelpad=0)
        ax.set_ylabel(r"$\tau$", fontsize=FS_AXIS, labelpad=0)
        ax.set_zlabel(r"$\tilde{J}(\theta)$", fontsize=FS_AXIS, labelpad=4)
        ax.set_zlim(0, 1)

        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("0.82")
        ax.grid(False)
        ax.tick_params(labelsize=FS_TICK, pad=0)
        ax.xaxis.set_tick_params(pad=-1)
        ax.yaxis.set_tick_params(pad=-1)
        ax.zaxis.set_tick_params(pad=3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        if show_z_ticks:
            ax.zaxis.set_major_locator(plt.MaxNLocator(4))
        else:
            ax.set_zticks([])
            ax.set_zticklabels([])

        ax.text2D(0.5, 1.02, subtitle,
                  transform=ax.transAxes, ha="center", va="bottom",
                  fontsize=FS_SUB, fontweight="bold", **FONT_SERIF)
        ax.text2D(0.5, 0.97, subtitle2,
                  transform=ax.transAxes, ha="center", va="bottom",
                  fontsize=FS_SUB2, color="0.45", **FONT_SERIF)

    # ── Bottom tagline ────────────────────────────────────────────────────────
    fig.text(0.5, 0.005,
             "Smoother, comparable, learnable target",
             ha="center", va="bottom", fontsize=12,
             color="#2b7bba", fontweight="bold", **FONT_SERIF)

    return fig


# ── static figure ─────────────────────────────────────────────────────────────
if not args.gif:
    fig = make_frame(azim=args.azim, elev=args.elev)
    out_dir = os.path.join(ROOT, "outputs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    az_tag = f"_az{int(args.azim):+04d}" if args.azim != 135 else ""
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"landscape_dual_{args.scene}{az_tag}.{ext}")
        fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    if args.scene == "XAM-N6" and args.azim == 135:
        paper_pdf = os.path.join(ROOT, "paper", "Figures", f"landscape_dual_{args.scene}.pdf")
        os.makedirs(os.path.dirname(paper_pdf), exist_ok=True)
        shutil.copy2(os.path.join(out_dir, f"landscape_dual_{args.scene}.pdf"), paper_pdf)
        print(f"Copied: {paper_pdf}")
    plt.close(fig)

# ── animated GIF ──────────────────────────────────────────────────────────────
else:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not found — installing …")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "-q"])
        from PIL import Image

    import io

    out_dir  = os.path.join(ROOT, "outputs", "figures")
    gif_path = os.path.join(out_dir, f"landscape_rotate_{args.scene}.gif")
    os.makedirs(out_dir, exist_ok=True)

    STEP  = 5          # degrees per frame
    ELEV  = args.elev  # elevation stays fixed (default 28)
    azims = list(range(-180, 180, STEP))  # 72 frames
    frames = []

    print(f"Rendering {len(azims)} frames …")
    for i, az in enumerate(azims):
        fig = make_frame(azim=az, elev=ELEV, figsize=(8.2, 3.6))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        if (i + 1) % 18 == 0:
            print(f"  {i+1}/{len(azims)} frames done")

    # Save GIF (loop=0 → infinite loop)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=60,   # ms per frame  ≈ ~16 fps for smooth rotation
        loop=0,
        optimize=False,
    )
    print(f"GIF saved: {gif_path}")
