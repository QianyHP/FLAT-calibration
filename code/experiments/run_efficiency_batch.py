"""run_efficiency_batch.py — 12-way parallel launcher for efficiency experiments.

Finds all missing extended-budget cache files and dispatches them in parallel.

Usage:
  python code/experiments/run_efficiency_batch.py --workers 12
  python code/experiments/run_efficiency_batch.py --workers 12 --scenes Tianjin,RML
  python code/experiments/run_efficiency_batch.py --workers 12 --methods TPE,GA
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
EXP = PROJ / "code" / "experiments"
CAL = PROJ / "code" / "calibration"
for _p in (EXP, CAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_io import REPLICATE_SEEDS  # noqa: E402
from run_efficiency_extended import (  # noqa: E402
    BASELINES_EXT,
    BUDGET_EXT,
    CACHE_EXT_DIR,
    SCENE_ORDER,
    _cache_path_ext,
)

RUN = PROJ / "code" / "experiments" / "run_efficiency_extended.py"
LOG_DIR = PROJ / "outputs" / "results" / "_batch_logs_ext"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _missing_jobs(
    scenes: list[str] | None = None,
    methods: list[str] | None = None,
) -> list[tuple[str, str, int]]:
    scenes = scenes or SCENE_ORDER
    methods = methods or BASELINES_EXT
    jobs: list[tuple[str, str, int]] = []
    for scene in scenes:
        for mkey in methods:
            for seed in REPLICATE_SEEDS:
                if not _cache_path_ext(scene, mkey, seed).exists():
                    jobs.append((scene, mkey, seed))
    return jobs


def _run_one(scene: str, mkey: str, seed: int) -> dict:
    log = LOG_DIR / f"ext_{scene}_{mkey}_s{seed}.log"
    cmd = [
        sys.executable,
        str(RUN),
        "--scene", scene,
        "--method", mkey,
        "--seed", str(seed),
        "--resume",
    ]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(PROJ), stdout=f, stderr=subprocess.STDOUT)
    return {
        "scene": scene,
        "method": mkey,
        "seed": seed,
        "ok": proc.returncode == 0,
        "elapsed_s": round(time.time() - t0, 1),
        "log": str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel launcher for extended-budget efficiency experiments."
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--scenes", type=str, default="",
                        help="Comma-separated scenes (default: all 6)")
    parser.add_argument("--methods", type=str, default="",
                        help="Comma-separated methods (default: SPSA,GA,CMA-ES,TPE)")
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()] or None
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or None

    jobs = _missing_jobs(scenes, methods)

    total = len(SCENE_ORDER) * len(BASELINES_EXT) * len(REPLICATE_SEEDS)
    if not jobs:
        print(f"All {total} extended-budget caches present. Nothing to run.")
        print("Run analyze_efficiency.py to compute ratios.")
        return

    print(f"Extended-budget jobs: {len(jobs)}/{total}  workers: {args.workers}")
    print(f"Budget per run: {BUDGET_EXT} SUMO simulations")
    print(f"Cache dir: {CACHE_EXT_DIR}")
    for scene, mkey, seed in jobs[:15]:
        print(f"  {scene:<12} {mkey:<8} seed={seed}")
    if len(jobs) > 15:
        print(f"  ... +{len(jobs) - 15} more")
    print()

    ok, fail = 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_run_one, *j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            status = "OK" if r["ok"] else "FAIL"
            if r["ok"]:
                ok += 1
            else:
                fail += 1
            print(
                f"[{i}/{len(jobs)}] {status}  {r['scene']:<12} {r['method']:<8} "
                f"s={r['seed']}  {r['elapsed_s']}s",
                flush=True,
            )
            if not r["ok"]:
                print(f"       log: {r['log']}", flush=True)

    elapsed_min = (time.time() - t0) / 60
    print(f"\nDone in {elapsed_min:.1f} min  ok={ok}  fail={fail}")
    if fail:
        print(f"Check logs in {LOG_DIR}")
        sys.exit(1)
    else:
        print("\nAll jobs complete. Now run:")
        print("  python code/experiments/analyze_efficiency.py")


if __name__ == "__main__":
    main()
