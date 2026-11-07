#!/usr/bin/env python3
"""
Learnability curves on 10-D LHS landscapes (v3 protocol — Appendix A2).

Differs from Fig. 8 in-sample R^2_fit on sequential calibration samples:
  - Fixed 25% external holdout (100/400), same test set for all N
  - Training sizes N from the remaining 300 pool
  - FLAT-native GP (same surrogate family as main GP experiments)
  - Separate response surfaces J_b and J_s on identical theta

Metric: R^2_ext (external holdout). R^2_fit on train for audit only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import r2_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "experiments"))
from plot_style import APPENDIX_SCENES  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "code", "calibration"))
from unified_calibration import (  # noqa: E402
    PARAM_BOUNDS,
    PARAM_NAMES,
    _fit_gp,
    _fit_rf,
    _surrogate_valid_mask,
)

FAIL_VALUE = 10.0
TEST_SIZE = 100
N_REPEATS = 5
BASE_SEED = 42
N_GRID = [40, 60, 80, 100, 120, 160, 200, 240, 280, 300]
DEFAULT_SCENES = APPENDIX_SCENES


def load_xy(scene: str, objective: str) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(ROOT, "outputs", "results", f"landscape_full10d_{scene}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lo = PARAM_BOUNDS[:, 0]
    span = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]
    X, Y = [], []
    key = "J_b" if objective == "jb" else "J_s"
    for item in data["results"]:
        y = float(item[key])
        if y >= FAIL_VALUE:
            continue
        p = item["params"]
        vec = np.array([(p[n] - lo[i]) / span[i] for i, n in enumerate(PARAM_NAMES)])
        X.append(vec)
        Y.append(y)
    return np.vstack(X), np.array(Y)


def split_fixed(n: int, test_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    test_idx = rng.choice(n, size=test_size, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[test_idx] = False
    train_idx = np.where(mask)[0]
    return train_idx, test_idx


def fit_flat(scene: str, surrogate: str, X_tr: np.ndarray, y_tr: np.ndarray, run_seed: int):
    valid = _surrogate_valid_mask(y_tr)
    if int(valid.sum()) < 10:
        return None
    Xv, yv = X_tr[valid], y_tr[valid]
    if surrogate == "gp":
        return _fit_gp(Xv, yv, scene, run_seed=run_seed)
    return _fit_rf(Xv, yv, scene, run_seed=run_seed)


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) < 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def curve_scene(
    scene: str,
    surrogate: str,
    objectives: list[str],
    n_grid: list[int],
    n_repeats: int,
    test_size: int,
) -> dict:
    out: dict = {"surrogate": surrogate, "objectives": {}}
    X_jb, Y_jb = load_xy(scene, "jb")
    _, Y_js = load_xy(scene, "js")
    train_idx, test_idx = split_fixed(len(X_jb), test_size, BASE_SEED)
    train_pool = train_idx
    X_test = X_jb[test_idx]

    for obj in objectives:
        Y_all = Y_jb if obj == "jb" else Y_js
        Y_test = Y_all[test_idx]
        rows = []
        for n in n_grid:
            if n > len(train_pool):
                continue
            ext_scores, fit_scores = [], []
            for rep in range(n_repeats):
                rng = np.random.default_rng(BASE_SEED + n * 17 + rep)
                sub = rng.choice(train_pool, size=n, replace=False)
                X_tr, y_tr = X_jb[sub], Y_all[sub]
                model = fit_flat(scene, surrogate, X_tr, y_tr, run_seed=BASE_SEED + rep)
                if model is None:
                    ext_scores.append(float("nan"))
                    fit_scores.append(float("nan"))
                    continue
                pred_ext = model.predict(X_test)
                pred_fit = model.predict(X_tr)
                ext_scores.append(r2_safe(Y_test, pred_ext))
                fit_scores.append(r2_safe(y_tr, pred_fit))
            ext = np.array(ext_scores, dtype=float)
            fit = np.array(fit_scores, dtype=float)
            rows.append({
                "n_train": n,
                "r2_ext_mean": float(np.nanmean(ext)),
                "r2_ext_std": float(np.nanstd(ext)),
                "r2_fit_mean": float(np.nanmean(fit)),
                "r2_fit_std": float(np.nanstd(fit)),
                "n_test": int(len(test_idx)),
            })
        out["objectives"][obj] = rows
    return out


def main(
    scenes: list[str],
    surrogates: list[str],
    objectives: list[str],
    test_size: int = TEST_SIZE,
    n_grid: list[int] | None = None,
):
    n_grid = n_grid or [n for n in N_GRID if n <= 400 - test_size]
    payload = {
        "protocol": "v3",
        "metric_primary": "r2_ext",
        "metric_fig8": "r2_fit on sequential calibration samples (Fig. percalib)",
        "test_size": test_size,
        "train_pool": 400 - test_size,
        "n_grid": n_grid,
        "n_repeats": N_REPEATS,
        "surrogates": "FLAT _fit_gp / _fit_rf (unified_calibration)",
        "scenes": {},
    }

    for scene in scenes:
        payload["scenes"][scene] = {}
        for surr in surrogates:
            payload["scenes"][scene][surr] = curve_scene(
                scene, surr, objectives, n_grid, N_REPEATS, test_size,
            )
            jb = payload["scenes"][scene][surr]["objectives"]["jb"]
            js = payload["scenes"][scene][surr]["objectives"]["js"]
            for n_ref in (40, 100):
                rb = next((r for r in jb if r["n_train"] == n_ref), None)
                rs = next((r for r in js if r["n_train"] == n_ref), None)
                if rb and rs:
                    print(
                        f"{scene} {surr} N={n_ref}: "
                        f"R2_ext Jb={rb['r2_ext_mean']:.3f} Js={rs['r2_ext_mean']:.3f} | "
                        f"R2_fit Jb={rb['r2_fit_mean']:.3f} Js={rs['r2_fit_mean']:.3f}"
                    )

    out_path = os.path.join(ROOT, "outputs", "results", "landscape_learnability_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_path}")
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    p.add_argument("--test-size", type=int, default=TEST_SIZE)
    p.add_argument("--surrogates", nargs="*", default=["gp"])
    p.add_argument("--objectives", nargs="*", default=["jb", "js"])
    args = p.parse_args()
    main(args.scenes, args.surrogates, args.objectives, args.test_size)
