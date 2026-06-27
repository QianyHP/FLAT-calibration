#!/usr/bin/env python3
"""
run_landscape_dual.py — 2D objective landscape sweep (real SUMO).

Purpose
-------
Collect grid data for the rugged-vs-smoothed landscape figure: at each
(accel, tau) point run one SUMO simulation and record BOTH
  • raw  — RMSE of per-second mean speed (sim vs. observed frenet.csv)
  • J_b  — standard 8-d behavioural-fingerprint error (same as calibration)

All other eight parameters stay fixed at the scene's calibrated θ* from
``{scene}_calibration_with_history.json``.

Default scene: XAM-N6 (UTE expressway). Grid: 20×20 = 400 SUMO runs.

Usage
-----
  python code/experiments/run_landscape_dual.py
  python code/experiments/run_landscape_dual.py --scene XAM-N6 --workers 6

Requires SUMO + TraCI. Prerequisite: ``unified_calibration.py`` for the scene.

Output JSON schema (``outputs/results/landscape_dual_<SCENE>.json``)
--------------------------------------------------------------------
  scene, accel_range, tau_range, n_grid,
  results: list of [accel, tau, J_b, raw_rmse]
  Failed runs are stored as 10.0 for both metrics.
"""

import argparse, json, os, sys, time, multiprocessing as mp
import numpy as np
import pandas as pd

_parser = argparse.ArgumentParser(description="2D landscape grid sweep (accel × tau)")
_parser.add_argument("--scene", default="XAM-N6",
                     help="scene key in SCENARIOS (default: XAM-N6)")
_parser.add_argument("--workers", type=int, default=6,
                     help="parallel SUMO workers (default: 6)")
_args, _ = _parser.parse_known_args()

# ── project path ─────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code", "calibration"))
from unified_calibration import (
    SCENARIOS, FEATURE_WEIGHTS, FEATURE_NAMES,
    compute_features, feature_error, extract_real_features,
)

SCENE     = _args.scene
N_GRID    = 20
N_WORKERS = _args.workers

ACCEL_RANGE = (0.5, 4.0)
TAU_RANGE   = (0.5, 2.5)

# ── pre-compute real per-second mean speed (UTE: frenet.csv) ─────────────────
def build_real_speed_profile(sc):
    df = pd.read_csv(sc["data"] / "frenet.csv")
    cols = {c.lower(): c for c in df.columns}
    spd_col = cols.get("speed(km/h)", cols.get("velocity(km/h)"))
    time_col = cols.get("time(s)", cols.get("timestamp(s)", cols.get("time")))
    speed = df[spd_col].values / 3.6   # km/h → m/s
    warmup, sim_end = sc["warmup"], sc["sim_end"]
    if time_col:
        t_s = df[time_col].values
        mask = (t_s >= warmup) & (t_s < sim_end)
        sub_t = t_s[mask]
        sub_v = speed[mask]
        per_sec = pd.Series(sub_v, name="speed")\
                    .groupby(sub_t.astype(int)).mean()
    else:
        # No time column: use row index at 0.1s steps
        t_s = np.arange(len(df)) * 0.1
        mask = (t_s >= warmup) & (t_s < sim_end)
        per_sec = pd.Series(speed[mask]).groupby((t_s[mask]).astype(int)).mean()
    return per_sec.sort_index()


# ── single SUMO run returning (accel, tau, J_b, raw_rmse) ────────────────────
def evaluate_point(args):
    accel, tau, opt_params, real_feats, real_speed_series = args
    import traci

    sc = SCENARIOS[SCENE]
    params = dict(opt_params)   # copy θ★ for all 10 dims
    params["accel"] = float(accel)
    params["tau"]   = float(tau)

    cfg = str(sc["data"] / "sumo" / f"{sc['cfg']}.sumocfg")
    seed = np.random.randint(1, 99999)

    try:
        traci.start(["sumo", "-c", cfg,
                     "--no-step-log", "true",
                     "--no-warnings", "true",
                     "--seed", str(seed)])
        # set vehicle type params (skip non-car types)
        skip_types = {"DEFAULT_BIKETYPE", "DEFAULT_PEDTYPE", "DEFAULT_RAILTYPE",
                      "DEFAULT_CONTAINERTYPE", "DEFAULT_TAXITYPE"}
        for vtype in traci.vehicletype.getIDList():
            if vtype in skip_types:
                continue
            traci.vehicletype.setAccel(vtype,        float(params["accel"]))
            traci.vehicletype.setDecel(vtype,        float(params["decel"]))
            traci.vehicletype.setImperfection(vtype, float(params["sigma"]))
            traci.vehicletype.setTau(vtype,          float(params["tau"]))
            traci.vehicletype.setMinGap(vtype,       float(params["minGap"]))
            traci.vehicletype.setSpeedFactor(vtype,  float(params["speedFactor"]))
            for lc in ("lcStrategic", "lcCooperative", "lcAssertive", "lcSpeedGain"):
                try:
                    traci.vehicletype.setParameter(vtype, lc, str(float(params[lc])))
                except Exception:
                    pass

        warmup   = sc["warmup"]
        sim_end  = sc["sim_end"]

        all_speeds  = []
        all_accels  = []
        per_sec_sim = {}   # second -> list of speeds

        step = 0
        while traci.simulation.getTime() < sim_end:
            traci.simulationStep()
            step += 1
            t = traci.simulation.getTime()
            if t < warmup:
                continue
            vids = traci.vehicle.getIDList()
            if not vids:
                continue
            speeds_now = [traci.vehicle.getSpeed(vid) for vid in vids]
            accels_now = [traci.vehicle.getAcceleration(vid) for vid in vids]
            t_int = int(t)
            per_sec_sim.setdefault(t_int, []).extend(speeds_now)
            if step % 3 == 0:
                all_speeds.extend(speeds_now)
                all_accels.extend(accels_now)

    except Exception as exc:
        try: traci.close()
        except Exception: pass
        return (accel, tau, 10.0, 10.0)
    finally:
        try: traci.close()
        except Exception: pass

    # ── J_b ──────────────────────────────────────────────────────────────────
    if len(all_speeds) < 20:
        return (accel, tau, 10.0, 10.0)
    sim_feats = compute_features(np.array(all_speeds), np.array(all_accels))
    jb = feature_error(real_feats, sim_feats)

    # ── raw: per-second mean speed RMSE ──────────────────────────────────────
    common_secs = sorted(set(per_sec_sim.keys()) & set(real_speed_series.index))
    if len(common_secs) < 10:
        return (accel, tau, jb, 10.0)
    sim_means  = np.array([np.mean(per_sec_sim[s]) for s in common_secs])
    real_means = np.array([real_speed_series[s]    for s in common_secs])
    raw_rmse   = float(np.sqrt(np.mean((sim_means - real_means) ** 2)))

    return (accel, tau, float(jb), raw_rmse)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    sc           = SCENARIOS[SCENE]
    real_feats   = extract_real_features(SCENE)
    real_speed   = build_real_speed_profile(sc)
    opt_params   = json.load(open(
        os.path.join(ROOT,
                     "data", "processed_data", "calibration",
                     f"{SCENE}_calibration_with_history.json")
    ))["calibrated_params"]

    accel_vals = np.linspace(*ACCEL_RANGE, N_GRID)
    tau_vals   = np.linspace(*TAU_RANGE,   N_GRID)

    args_list = [
        (accel, tau, opt_params, real_feats, real_speed)
        for accel in accel_vals
        for tau   in tau_vals
    ]

    print(f"Launching {len(args_list)} SUMO runs on {N_WORKERS} workers ...")
    t0 = time.time()

    results = []
    done = 0
    with mp.Pool(N_WORKERS) as pool:
        for res in pool.imap_unordered(evaluate_point, args_list):
            results.append(res)
            done += 1
            if done % 12 == 0:
                elapsed = time.time() - t0
                eta     = elapsed / done * (len(args_list) - done)
                print(f"  {done}/{len(args_list)} done  ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} min")

    # Save
    out = {
        "scene": SCENE,
        "accel_range": list(ACCEL_RANGE),
        "tau_range":   list(TAU_RANGE),
        "n_grid":    N_GRID,
        "results":   [list(r) for r in results],   # [accel, tau, jb, raw]
    }
    out_path = os.path.join(ROOT, "outputs", "results", f"landscape_dual_{SCENE}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
