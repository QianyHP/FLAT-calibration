"""Matched replay of three FLAT backends: six scenes, four original runs, ten seeds.

The original four common parameter sets (101,202,303,404) are used for every
backend and scene. Missing original external-baseline parameters are not replaced.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import replay_support as pilot

RELEASE_CALIBRATION_LF_SHA256 = '86e8ce48888261c9f777e4c26a5f9f78e29370c0cc33791a8a5e684d5c1ffef8'
ROOT = pilot.ROOT
OUT = ROOT / 'outputs/results/replay_surrogates'
SCENES = ('Tianjin', 'Changchun', 'Xian', 'YTDJ', 'RML', 'XAM-N6')
ARMS = ('BF-SAC-GP', 'BF-SAC-RF', 'BF-SAC-MLP')
OPT_SEEDS = (101, 202, 303, 404)
SIM_SEEDS = tuple(range(10001, 10011))


def prepare():
    """Load the published, frozen parameter vectors; no raw traffic data needed."""
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    jobs = []
    for p in manifest['parameter_sets']:
        assert set(p['params']) == set(pilot.cal.PARAM_NAMES)
        for seed in SIM_SEEDS:
            jobs.append({**{k:v for k,v in p.items() if k != 'original_best_jb'}, 'sim_seed':seed})
    assert len(jobs) == 720 and len({pilot.key(j) for j in jobs}) == 720
    assert len(manifest['parameter_sets']) == 72
    return jobs, manifest


def validate_inputs(manifest):
    """Stop on missing or changed source data instead of silently changing a replay."""
    for rel, digest in manifest['input_hashes'].items():
        file = ROOT / rel.replace('\\', '/')
        if not file.exists() or pilot.sha(file) != digest:
            raise ValueError(f'Missing or changed original input: {rel}')
    normalized = hashlib.sha256(Path(pilot.cal.__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if pilot.sha(Path(pilot.cal.__file__)) != manifest['calibration_module_sha256'] and normalized != RELEASE_CALIBRATION_LF_SHA256:
        raise ValueError('Calibration implementation differs from the recorded experiment')
    version = subprocess.check_output(['sumo','--version'],text=True).splitlines()[0]
    if version != manifest['sumo_version']:
        raise ValueError('SUMO version differs from the recorded experiment')


def aggregate(rows):
    df = pd.DataFrame(rows).sort_values(['scene','arm','opt_seed','sim_seed'])
    assert len(df) == 720 and (df.status == 'ok').all()
    assert not df.duplicated(['scene','arm','opt_seed','sim_seed']).any()
    metrics = ['jb','js_mps','wasserstein_mps']
    assert np.isfinite(df[metrics].to_numpy()).all()
    grouped = df.groupby(['scene','arm','opt_seed'])
    assert len(grouped) == 72 and (grouped.size() == 10).all()
    per_opt = grouped[metrics].mean().reset_index()
    per_opt = per_opt.merge(grouped.jb.std().rename('jb_sim_seed_sd').reset_index())
    df.to_csv(OUT/'per_replay.csv',index=False)
    per_opt.to_csv(OUT/'per_optimization_seed.csv',index=False)
    summary = per_opt.groupby(['scene','arm'])[metrics].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(OUT/'summary.csv')
    print(summary.to_string(),flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    jobs,manifest = prepare()
    validate_inputs(manifest)
    imported = {}
    print('PREFLIGHT: 72 frozen parameter sets; original inputs verified; 720 replays',flush=True)
    for scene in SCENES:
        speeds,_,_ = pilot.observed(scene)
        print(f'PREFLIGHT {scene}: {len(speeds)} observed speeds',flush=True)
    if args.preflight:
        return
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'cache').mkdir(exist_ok=True)
    file = OUT/'manifest.json'
    if file.exists():
        if not args.resume:
            raise RuntimeError('Existing experiment requires explicit --resume')
        assert json.loads(file.read_text(encoding='utf-8')) == manifest
    else:
        file.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    saved_file = OUT / 'per_replay.csv'
    saved = {}
    if args.resume and saved_file.exists():
        for row in pd.read_csv(saved_file).to_dict('records'):
            if row['status'] != 'ok' or pilot.key(row) in saved:
                raise ValueError('Failed or duplicate published replay row')
            saved[pilot.key(row)] = row
    rows,pending = [],[]
    for job in jobs:
        key = pilot.key(job)
        file = OUT/'cache'/(key+'.json')
        if args.resume and file.exists():
            row = json.loads(file.read_text(encoding='utf-8'))
            assert row['status']=='ok' and row['source_sha256']==job['source_sha256']
            rows.append(row)
        elif key in saved:
            row = saved[key]
            assert row['source_sha256'] == job['source_sha256']
            rows.append(row)
        elif key in imported:
            row = imported[key]
            file.write_text(json.dumps(row,indent=2),encoding='utf-8')
            rows.append(row)
        else:
            pending.append(job)
    workers = max(1,min(args.workers,4))
    start = time.perf_counter()
    print(f'START: {len(rows)} imported/cached, {len(pending)} new; {workers} workers',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(pilot.replay,j):j for j in pending}
        for future in as_completed(futures):
            job = futures[future]
            row = future.result()
            (OUT/'cache'/(pilot.key(job)+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
            rows.append(row)
            print(f"{len(rows)}/720 {pilot.key(job)} {row['status']} Jb={row.get('jb')} {row['elapsed_s']:.1f}s",flush=True)
    failures = [r for r in rows if r['status']!='ok']
    accounting = dict(completed=len(rows),failures=failures,workers=workers,
                      new_replays_this_invocation=len(pending),imported_rows=len(imported),
                      invocation_wall_s=time.perf_counter()-start,
                      new_replay_elapsed_sum_s=sum(r['elapsed_s'] for r in rows if pilot.key(r) not in imported))
    (OUT/'run_accounting.json').write_text(json.dumps(accounting,indent=2),encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} failures; no automatic retries')
    aggregate(rows)


if __name__ == '__main__':
    main()
