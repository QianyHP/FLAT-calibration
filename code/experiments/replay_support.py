"""Shared fixed-parameter SUMO evaluation; no optimization or parameter selection."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/calibration"))
import unified_calibration as cal

OBS = {}

def key(job):
    return f"{job['scene']}_{job['arm']}_opt{job['opt_seed']}_sim{job['sim_seed']}"


def observed(scene):
    if scene not in OBS:
        sc = cal.SCENARIOS[scene]
        if sc['type'] == 'SIND':
            df = pd.read_csv(sc['data'] / 'veh_tracks.csv')
            t = df['timestamp_ms'].to_numpy() / 1000
            v = np.hypot(df['vx'].to_numpy(), df['vy'].to_numpy())
        else:
            df = pd.read_csv(sc['data'] / 'frenet.csv')
            cols = {c.lower(): c for c in df.columns}
            t = df[cols['time(s)']].to_numpy()
            v = df[cols.get('speed(km/h)', cols.get('velocity(km/h)'))].to_numpy() / 3.6
        mask = ((t >= sc['warmup']) & (t < sc['sim_end']) & np.isfinite(v)
                & (v >= 0) & (v <= 40))
        speeds = v[mask]
        if len(speeds) < 20:
            raise ValueError(f'Too few observed speed samples: {scene}')
        OBS[scene] = (speeds, cal.extract_real_features(scene),
                      cal.build_real_speed_profile(scene))
    return OBS[scene]


def replay(job):
    start = time.perf_counter()
    row = {k: v for k, v in job.items() if k != 'params'}
    try:
        real, reference, profile = observed(job['scene'])
        # TraCI is isolated per worker process. No optimizer or global RNG reseeding.
        feat, per_sec, speeds = cal._run_sumo_sim(
            job['scene'], job['params'], seed=job['sim_seed'], return_speed_samples=True)
        # Optional samples exclude the terminal step; legacy J_b is unchanged.
        speeds = speeds[np.isfinite(speeds) & (speeds >= 0) & (speeds <= 40)]
        if len(speeds) < 20 or not np.all(np.isfinite(feat)):
            raise ValueError('Insufficient/invalid simulated observations')
        common = set(per_sec).intersection(profile.index)
        if len(common) < 10:
            raise ValueError('Insufficient time-profile overlap')
        row.update(status='ok', wasserstein_mps=float(wasserstein_distance(real, speeds)),
                   jb=cal.feature_error(reference, feat, cal.get_feature_weights(job['scene'])),
                   js_mps=cal.per_second_speed_rmse(per_sec, profile),
                   n_real=int(real.size), n_sim=int(speeds.size), n_common_seconds=len(common))
    except Exception as exc:
        row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    row['elapsed_s'] = time.perf_counter() - start
    return row
