"""Compute reference-only, per-vehicle inverse-variance fingerprint weights.

This utility writes a weight file without changing calibration defaults.
Run from the repository root: python code/calibration/compute_ivw_weights.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from unified_calibration import SCENARIOS, FEATURE_NAMES, compute_features, get_feature_weights


def inverse_variance_weights(features, *, variance_floor=0.0):
    """Sample variance across vehicles; normalize inverse variances to sum one."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError('Need at least two complete, finite vehicle fingerprints')
    if not np.isfinite(variance_floor) or variance_floor < 0:
        raise ValueError('variance_floor must be finite and nonnegative')
    variance = x.var(axis=0, ddof=1)
    denominator = np.maximum(variance, variance_floor)
    if np.any(denominator <= 0):
        raise ValueError('Zero reference variance: explicitly choose a positive variance floor')
    # Equivalent to 1/variance followed by normalization, avoiding overflow.
    precision = denominator.min() / denominator
    return precision / precision.sum(), variance


def scene_weights(scene, *, variance_floor=0.0):
    sc = SCENARIOS[scene]
    path = sc['data'] / ('veh_tracks.csv' if sc['type']=='SIND' else 'frenet.csv')
    df = pd.read_csv(path)
    if sc['type']=='SIND':
        ids = df['track_id'].to_numpy()
        speed = np.hypot(df.vx.to_numpy(), df.vy.to_numpy())
        accel = (df.vx.to_numpy()*df.ax.to_numpy()+df.vy.to_numpy()*df.ay.to_numpy()) / np.maximum(speed,0.01)
        accel[speed < 0.3] = 0.0
    else:
        ids = df['vehicleID'].to_numpy()
        cols = {c.lower():c for c in df.columns}
        speed = df[cols.get('speed(km/h)',cols.get('velocity(km/h)'))].to_numpy()/3.6
        accel = df[cols['acceleration(m/s^2)']].to_numpy()
    # Full observed records, matching extract_real_features; no time-window filtering.
    samples = pd.DataFrame({'id':ids,'speed':speed,'accel':accel})
    complete=[]; excluded_short=0; excluded_nonfinite=0
    if samples.id.isna().any():
        raise ValueError(f'{scene}: missing vehicle identifiers')
    for _, group in samples.groupby('id',sort=True):
        if len(group)<20:
            excluded_short+=1;continue
        if not np.isfinite(group[['speed','accel']].to_numpy()).all():
            excluded_nonfinite+=1;continue
        complete.append(compute_features(group.speed.to_numpy(),group.accel.to_numpy()))
    weights,variance=inverse_variance_weights(complete,variance_floor=variance_floor)
    legacy=np.asarray(get_feature_weights(scene))
    return {'scene':scene,'source':path.relative_to(Path(__file__).resolve().parents[2]).as_posix(),
            'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'vehicles_used':len(complete),'vehicles_excluded_short':excluded_short,
            'vehicles_excluded_nonfinite':excluded_nonfinite,
            'sample_variance':variance.tolist(),'weights':weights.tolist(),
            'legacy_weights':legacy.tolist(),'legacy_weight_sum':float(legacy.sum()),
            'max_abs_difference_from_legacy':float(np.max(np.abs(weights-legacy)))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes',nargs='+',choices=list(SCENARIOS),default=list(SCENARIOS))
    p.add_argument('--variance-floor',type=float,default=0.0)
    p.add_argument('--output',type=Path,default=Path('outputs/results/ivw_reference_weights.json'))
    args=p.parse_args()
    if args.output.exists():
        p.error('Output already exists; select a new path or deliberately remove the old derived file')
    payload={'protocol':'per_vehicle_reference_ivw_v1','feature_names':FEATURE_NAMES,
             'variance':'sample variance (ddof=1) across equally weighted vehicle fingerprints',
             'scale':'raw feature units: m/s, m/s^2, dimensionless stop fraction; unit-dependent',
             'observation_window':'full observed file, matching extract_real_features',
             'variance_floor':args.variance_floor,'zero_variance_policy':'fail unless positive floor explicitly supplied',
             'calibration_defaults_changed':False,
             'scenes':[scene_weights(s,variance_floor=args.variance_floor) for s in args.scenes]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for s in payload['scenes']:
        print(s['scene'],s['vehicles_used'],np.round(s['weights'],6).tolist(), 'legacy difference',round(s['max_abs_difference_from_legacy'],6))
    print(args.output)

if __name__=='__main__':
    main()
