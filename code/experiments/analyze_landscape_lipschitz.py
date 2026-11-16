#!/usr/bin/env python3
"""
Response-standardized empirical local Lipschitz proxies for J_s vs J_b on landscape grid samples.

Uses existing landscape_dual JSON (no extra SUMO). Each grid point is embedded
in the full 10-D parameter vector (theta* with accel/tau swept). k-NN slopes
are computed in min-max normalized parameter space, with each response divided
by its population standard deviation on the same valid design.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "calibration"))
from unified_calibration import PARAM_BOUNDS, PARAM_NAMES  # noqa: E402

DEFAULT_SCENE = "XAM-N6"
K_NEIGHBORS = 5
PERCENTILE = 95
FAIL_VALUE = 10.0


def load_landscape(scene: str, mode: str = "slice") -> tuple[dict, str]:
    if mode == "full10d":
        rel = f"outputs/results/landscape_full10d_{scene}.json"
    else:
        rel = f"outputs/results/landscape_dual_{scene}.json"
    path = os.path.join(ROOT, rel)
    with open(path, encoding="utf-8") as f:
        return json.load(f), rel


def load_theta_star(scene: str) -> dict[str, float]:
    path = os.path.join(
        ROOT, "data", "processed_data", "calibration",
        f"{scene}_calibration_with_history.json",
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)["calibrated_params"]


def params_to_vector(params: dict[str, float]) -> np.ndarray:
    return np.array([params[name] for name in PARAM_NAMES], dtype=float)


def normalize_theta(theta: np.ndarray) -> np.ndarray:
    lo = PARAM_BOUNDS[:, 0]
    span = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]
    return (theta - lo) / span


def build_dataset_slice(data: dict, theta_star: dict[str, float]):
    thetas = []
    jb_list = []
    js_list = []
    meta = []

    for accel, tau, jb, js in data["results"]:
        if jb >= FAIL_VALUE or js >= FAIL_VALUE:
            continue
        p = dict(theta_star)
        p["accel"] = float(accel)
        p["tau"] = float(tau)
        thetas.append(params_to_vector(p))
        jb_list.append(float(jb))
        js_list.append(float(js))
        meta.append({"accel": accel, "tau": tau})

    theta = np.vstack(thetas)
    theta_n = normalize_theta(theta)
    return theta_n, np.array(jb_list), np.array(js_list), meta


def build_dataset_full10d(data: dict):
    thetas = []
    jb_list = []
    js_list = []

    for item in data["results"]:
        jb = float(item["J_b"])
        js = float(item["J_s"])
        if jb >= FAIL_VALUE or js >= FAIL_VALUE:
            continue
        thetas.append(params_to_vector(item["params"]))
        jb_list.append(jb)
        js_list.append(js)

    theta = np.vstack(thetas)
    theta_n = normalize_theta(theta)
    return theta_n, np.array(jb_list), np.array(js_list), []


def standardize_response(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale response differences by their design SD (centering cancels)."""
    scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Response standard deviation must be finite and positive")
    return values / scale, scale


def local_slopes(theta_n: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    n = len(values)
    tree = cKDTree(theta_n)
    # query k+1 to drop self (distance 0)
    dists, idx = tree.query(theta_n, k=min(k + 1, n))
    if n == 1:
        return np.array([0.0])

    point_slopes = []
    for i in range(n):
        slopes_i = []
        for d, j in zip(dists[i], idx[i]):
            if j == i or d <= 1e-12:
                continue
            slopes_i.append(abs(values[i] - values[j]) / d)
            if len(slopes_i) >= k:
                break
        if slopes_i:
            point_slopes.append(max(slopes_i))
    return np.array(point_slopes)


def summarize(name: str, slopes: np.ndarray, percentile: float) -> dict:
    if len(slopes) == 0:
        return {
            "objective": name,
            "n_points": 0,
            "mean_local_lipschitz": float("nan"),
            "median_local_lipschitz": float("nan"),
            f"p{int(percentile)}_local_lipschitz": float("nan"),
            "max_local_lipschitz": float("nan"),
        }
    return {
        "objective": name,
        "n_points": int(len(slopes)),
        "mean_local_lipschitz": float(np.mean(slopes)),
        "median_local_lipschitz": float(np.median(slopes)),
        f"p{int(percentile)}_local_lipschitz": float(np.percentile(slopes, percentile)),
        "max_local_lipschitz": float(np.max(slopes)),
    }


def main(
    scene: str = DEFAULT_SCENE,
    k: int = K_NEIGHBORS,
    percentile: float = PERCENTILE,
    mode: str = "slice",
):
    data, source = load_landscape(scene, mode)
    if mode == "full10d":
        theta_n, jb, js, _ = build_dataset_full10d(data)
        embedding = "10-D LHS full PARAM_BOUNDS; min-max normalized kNN"
        out_name = f"landscape_lipschitz_full10d_{scene}.json"
    else:
        theta_star = load_theta_star(scene)
        theta_n, jb, js, _ = build_dataset_slice(data, theta_star)
        embedding = "10-D theta* with accel/tau swept; min-max normalized by PARAM_BOUNDS"
        out_name = f"landscape_lipschitz_{scene}.json"

    jb_scaled, sd_b = standardize_response(jb)
    js_scaled, sd_s = standardize_response(js)
    slopes_b = local_slopes(theta_n, jb_scaled, k)
    slopes_s = local_slopes(theta_n, js_scaled, k)
    n_anchors = len(data["results"])
    n_valid = len(theta_n)

    p95_b = float(np.percentile(slopes_b, percentile)) if len(slopes_b) else float("nan")
    p95_s = float(np.percentile(slopes_s, percentile)) if len(slopes_s) else float("nan")
    ratio = p95_s / p95_b if p95_b > 0 else float("inf")

    out = {
        "scene": scene,
        "mode": mode,
        "source": source,
        "n_grid_total": n_anchors,
        "n_valid_points": n_valid,
        "k_neighbors": k,
        "percentile": percentile,
        "embedding": embedding,
        "response_scale": "per-objective population standard deviation on the valid design",
        "response_std": {"J_b": sd_b, "J_s": sd_s},
        "J_b": summarize("J_b", slopes_b, percentile),
        "J_s": summarize("J_s", slopes_s, percentile),
        "ratio_Js_over_Jb_p95": ratio,
    }

    out_path = os.path.join(ROOT, "outputs", "results", out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Local Lipschitz analysis on landscape grid")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--k", type=int, default=K_NEIGHBORS)
    parser.add_argument("--percentile", type=float, default=PERCENTILE)
    parser.add_argument(
        "--mode",
        choices=("slice", "full10d"),
        default="slice",
        help="slice: 2-D grid at theta*; full10d: kNN on 10-D LHS",
    )
    args = parser.parse_args()
    main(args.scene, args.k, args.percentile, args.mode)
