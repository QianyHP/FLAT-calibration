"""Measure serial SUMO--TraCI evaluation cost using fixed published parameters.

python code/experiments/measure_evaluation_cost.py
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/calibration'))
import unified_calibration as cal
OUT = ROOT / 'outputs/results/evaluation_cost'
SCENES = ('Tianjin', 'Changchun', 'Xian', 'YTDJ', 'RML', 'XAM-N6')
SEEDS = (20001, 20002, 20003)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def cost(output):
    manifest = ROOT / "outputs/results/replay_surrogates/manifest.json"
    parameters = json.loads(manifest.read_text(encoding="utf-8"))["parameter_sets"]
    rows = []
    for scene in SCENES:
        matching = [p for p in parameters if p['scene'] == scene and p['arm'] == 'BF-SAC-GP' and p['opt_seed'] == 101]
        if len(matching) != 1:
            raise ValueError(f'Expected one fixed parameter set for {scene}')
        source = manifest
        params = matching[0]['params']
        assert set(params) == set(cal.PARAM_NAMES)
        reference = cal.extract_real_features(scene)
        for seed in SEEDS:
            start = time.perf_counter()
            feat, _ = cal._run_sumo_sim(scene, params, seed=seed)
            jb = cal.feature_error(reference, feat, cal.get_feature_weights(scene))
            elapsed = time.perf_counter() - start
            assert np.isfinite(jb)
            rows.append(dict(scene=scene, simulation_seed=seed, elapsed_s=elapsed,
                             jb=float(jb), parameter_source=str(source.relative_to(ROOT)),
                             parameter_source_sha256=digest(source)))
            pd.DataFrame(rows).to_csv(output / 'evaluation_cost.csv', index=False)
            print(f'COST {scene} {seed}: {elapsed:.3f} s', flush=True)
    df = pd.DataFrame(rows)
    df.groupby('scene').elapsed_s.agg(['mean', 'median', 'min', 'max']).to_csv(output / 'evaluation_cost_summary.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output directory already exists; choose a new --output for a new measurement')
    args.output.mkdir(parents=True)
    manifest = dict(scenes=SCENES, simulation_seeds=SEEDS, optimization_seed=101,
                    parameters='fixed GP final parameters from published replay manifest',
                    timing='serial evaluation including startup, TraCI collection, fingerprint and scoring; reference extraction excluded',
                    simulation_count=18, platform=platform.platform(), processor=platform.processor(),
                    sumo_version=subprocess.check_output(['sumo','--version'],text=True).splitlines()[0])
    (args.output/'protocol.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    cost(args.output)
