"""run_batch_parallel.py — 补跑缺失 cache，维持 max_workers 路并行。

用法:
  python code/experiments/run_batch_parallel.py --workers 12
  python code/experiments/run_batch_parallel.py --workers 12 --phases main,sweep
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
for p in (EXP, CAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiment_io import REPLICATE_SEEDS, cache_path  # noqa: E402
from run_comparison import (  # noqa: E402
    BUDGET_FAIR,
    MAIN_METHODS,
    SCENE_ORDER,
    sweep_multiseed_method_keys,
)

RUN = PROJ / "code" / "experiments" / "run_comparison.py"
LOG_DIR = PROJ / "outputs" / "results" / "_batch_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _missing_main() -> list[tuple[str, str, str, int]]:
    jobs: list[tuple[str, str, str, int]] = []
    for scene in SCENE_ORDER:
        for mkey in MAIN_METHODS:
            for seed in REPLICATE_SEEDS:
                if not cache_path(scene, mkey, BUDGET_FAIR, seed).exists():
                    jobs.append(("main", scene, mkey, seed))
    return jobs


def _missing_sweep() -> list[tuple[str, str, str, int]]:
    """n_init∈{20,60,80,100} × 6 场景 × RF/MLP × 5 seed；n_init=40 用主对比 cache。"""
    jobs: list[tuple[str, str, str, int]] = []
    for scene in SCENE_ORDER:
        for mkey in sweep_multiseed_method_keys():
            for seed in REPLICATE_SEEDS:
                if not cache_path(scene, mkey, BUDGET_FAIR, seed).exists():
                    jobs.append(("sweep", scene, mkey, seed))
    return jobs


def _run_one(phase: str, scene: str, mkey: str, seed: int) -> dict:
    log = LOG_DIR / f"{phase}_{scene}_{mkey}_s{seed}.log"
    cmd = [
        sys.executable,
        str(RUN),
        "--scenes", scene,
        "--methods", mkey,
        "--mode", phase,
        "--seed", str(seed),
        "--resume",
    ]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(PROJ), stdout=f, stderr=subprocess.STDOUT)
    return {
        "phase": phase,
        "scene": scene,
        "method": mkey,
        "seed": seed,
        "ok": proc.returncode == 0,
        "elapsed_s": round(time.time() - t0, 1),
        "log": str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--phases", type=str, default="main,sweep")
    args = parser.parse_args()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    jobs: list[tuple[str, str, str, int]] = []
    if "main" in phases:
        jobs.extend(_missing_main())
    if "sweep" in phases:
        jobs.extend(_missing_sweep())

    if not jobs:
        print("All caches present. Nothing to run.")
        return

    print(f"Jobs: {len(jobs)}  workers: {args.workers}")
    for phase, scene, mkey, seed in jobs[:20]:
        print(f"  {phase} {scene} {mkey} seed={seed}")
    if len(jobs) > 20:
        print(f"  ... +{len(jobs) - 20} more")

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
                f"[{i}/{len(jobs)}] {status} {r['phase']} {r['scene']} {r['method']} "
                f"s={r['seed']}  {r['elapsed_s']}s",
                flush=True,
            )
            if not r["ok"]:
                print(f"       log: {r['log']}", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min  ok={ok} fail={fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
