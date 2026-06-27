"""run_efficiency_extended.py
Runs the four baselines (SPSA / GA / CMA-ES / TPE) with an extended budget
(default 500) to compute sample-efficiency ratios against FLAT@100.

Each job saves a convergence cache to outputs/results/comparison_cache_ext/.

Single-job usage:
  python code/experiments/run_efficiency_extended.py \\
      --scene Tianjin --method SPSA --seed 42 --resume

Rebuild CSV from existing cache:
  python code/experiments/run_efficiency_extended.py --rebuild-csv

Launch all 120 jobs in parallel via:
  python code/experiments/run_efficiency_batch.py --workers 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
EXP_ROOT = PROJ / "code" / "experiments"
CAL_ROOT = PROJ / "code" / "calibration"
for _p in (EXP_ROOT, CAL_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from experiment_io import REPLICATE_SEEDS  # noqa: E402
from unified_calibration import (  # noqa: E402
    PARAM_NAMES,
    SCENARIOS,
    feature_error,
    get_feature_weights,
    get_param_bounds,
    extract_real_features,
    _run_sumo_traci,
)

# ── constants ─────────────────────────────────────────────────────────────────
BUDGET_EXT = 500           # extended budget cap for baselines
GA_POP = 10                # keep consistent with run_comparison.py
FAIL_ERROR = 10.0
BASELINES_EXT = ["SPSA", "GA", "CMA-ES", "TPE"]
SCENE_ORDER = list(SCENARIOS.keys())

RES = PROJ / "outputs" / "results"
CACHE_EXT_DIR = RES / "comparison_cache_ext"
RES.mkdir(parents=True, exist_ok=True)
CACHE_EXT_DIR.mkdir(parents=True, exist_ok=True)


# ── extended evaluator ────────────────────────────────────────────────────────
@dataclass
class SumoEvaluatorExt:
    scene: str
    real_feat: np.ndarray
    weights: np.ndarray
    bounds: np.ndarray
    budget: int = BUDGET_EXT
    sim_count: int = 0
    best_error: float = field(default=float("inf"))
    sim_history: list[int] = field(default_factory=list)
    best_history: list[float] = field(default_factory=list)

    def _log(self, err: float) -> None:
        self.sim_count += 1
        if err < self.best_error:
            self.best_error = err
        self.sim_history.append(self.sim_count)
        self.best_history.append(self.best_error)

    def evaluate_vector(self, x: np.ndarray) -> float:
        if self.sim_count >= self.budget:
            raise RuntimeError(f"Budget exhausted ({self.budget})")
        params = {k: float(v) for k, v in zip(PARAM_NAMES, x)}
        try:
            sim_feat = _run_sumo_traci(self.scene, params)
            err = feature_error(self.real_feat, sim_feat, self.weights)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            # TraCI calls sys.exit(-1) when SUMO crashes/connection refused;
            # that raises SystemExit which is BaseException, not Exception.
            print(f"       [!] SUMO failed ({type(exc).__name__}): {exc}", flush=True)
            # Best-effort TraCI cleanup so the next traci.start() can succeed.
            try:
                import traci as _traci
                _traci.close()
            except Exception:
                pass
            err = FAIL_ERROR
        self._log(err)
        return err


def _make_evaluator_ext(scene: str, budget: int = BUDGET_EXT) -> SumoEvaluatorExt:
    return SumoEvaluatorExt(
        scene,
        extract_real_features(scene),
        get_feature_weights(scene),
        get_param_bounds(scene),
        budget=budget,
    )


def _pack_result_ext(method_key: str, ev: SumoEvaluatorExt, seed: int) -> dict:
    return {
        "method_key": method_key,
        "scene": ev.scene,
        "budget_ext": ev.budget,
        "n_sims": ev.sim_count,
        "final_error": round(ev.best_error, 4),
        "run_seed": int(seed),
        "convergence_sims": [int(s) for s in ev.sim_history],
        "convergence_best": [float(b) for b in ev.best_history],
    }


# ── extended baseline implementations ─────────────────────────────────────────
def method_spsa_ext(scene: str, seed: int, budget: int = BUDGET_EXT) -> dict:
    ev = _make_evaluator_ext(scene, budget)
    rng = np.random.default_rng(seed + abs(hash(scene)) % 9999)
    lo, hi = ev.bounds[:, 0], ev.bounds[:, 1]
    dim = len(PARAM_NAMES)
    theta = rng.random(dim) * (hi - lo) + lo
    ev.evaluate_vector(theta)

    a, c, A = 0.22, 0.15, 20.0
    ae, ge = 0.602, 0.101
    k = 1
    while ev.sim_count + 2 <= budget:
        ak = a / (k + A) ** ae
        ck = c / k ** ge
        delta = rng.choice([-1.0, 1.0], size=dim)
        tp = np.clip(theta + ck * delta, lo, hi)
        tm = np.clip(theta - ck * delta, lo, hi)
        fp = ev.evaluate_vector(tp)
        fm = ev.evaluate_vector(tm)
        g = (fp - fm) / (2.0 * ck) * delta
        theta = np.clip(theta - ak * g, lo, hi)
        k += 1
        if k % 30 == 0:
            print(f"       SPSA {ev.sim_count}/{budget}  best={ev.best_error:.4f}", flush=True)
    while ev.sim_count < budget:
        ev.evaluate_vector(theta)
    return _pack_result_ext("SPSA", ev, seed)


def _tournament(pop: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(pop), 3, replace=False)
    return pop[idx[np.argmin(fitness[idx])]]


def method_ga_ext(scene: str, seed: int, budget: int = BUDGET_EXT) -> dict:
    ev = _make_evaluator_ext(scene, budget)
    rng = np.random.default_rng(seed + abs(hash(scene)) % 9999)
    lo, hi = ev.bounds[:, 0], ev.bounds[:, 1]
    dim = len(PARAM_NAMES)
    n_waves = budget // GA_POP
    pop = rng.random((GA_POP, dim)) * (hi - lo) + lo
    fitness = np.zeros(GA_POP)

    for wave in range(n_waves):
        if wave == 0:
            for i in range(GA_POP):
                fitness[i] = ev.evaluate_vector(pop[i])
        else:
            new_pop = np.zeros_like(pop)
            for i in range(GA_POP):
                p1 = _tournament(pop, fitness, rng)
                p2 = _tournament(pop, fitness, rng)
                mask = rng.random(dim) < 0.5
                child = np.where(mask, p1, p2) + rng.normal(0, (hi - lo) * 0.05)
                new_pop[i] = np.clip(child, lo, hi)
                fitness[i] = ev.evaluate_vector(new_pop[i])
            pop = new_pop
        if (wave + 1) % 10 == 0:
            print(f"       GA wave={wave+1}/{n_waves}  best={ev.best_error:.4f}", flush=True)
    return _pack_result_ext("GA", ev, seed)


def method_cmaes_ext(scene: str, seed: int, budget: int = BUDGET_EXT) -> dict:
    import cma  # lazy import
    ev = _make_evaluator_ext(scene, budget)
    lo, hi = ev.bounds[:, 0], ev.bounds[:, 1]
    span = hi - lo
    dim = len(PARAM_NAMES)
    rng = np.random.default_rng(seed + abs(hash(scene)) % 9999)

    def scale(z) -> np.ndarray:
        return np.clip(lo + np.asarray(z, dtype=float) * span, lo, hi)

    x0 = rng.random(dim)
    es = cma.CMAEvolutionStrategy(
        list(x0), 0.3,
        {"bounds": [0.0, 1.0], "seed": int(seed) % (2 ** 31 - 2) + 1,
         "verbose": -9, "verb_log": 0, "verb_disp": 0, "maxfevals": budget},
    )
    best_x, best_f = scale(x0), float("inf")
    while ev.sim_count < budget and not es.stop():
        sols = es.ask()
        fits, complete = [], True
        for z in sols:
            if ev.sim_count >= budget:
                complete = False
                break
            xv = scale(z)
            f = ev.evaluate_vector(xv)
            fits.append(f)
            if f < best_f:
                best_f, best_x = f, xv
        if not complete:
            break
        es.tell(sols, fits)
        if es.countiter % 10 == 0:
            print(f"       CMA-ES {ev.sim_count}/{budget}  best={ev.best_error:.4f}", flush=True)
    while ev.sim_count < budget:
        ev.evaluate_vector(best_x)
    return _pack_result_ext("CMA-ES", ev, seed)


def method_tpe_ext(scene: str, seed: int, budget: int = BUDGET_EXT) -> dict:
    import optuna  # lazy import
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ev = _make_evaluator_ext(scene, budget)
    lo, hi = ev.bounds[:, 0], ev.bounds[:, 1]

    def objective(trial: "optuna.Trial") -> float:
        x = np.array([
            trial.suggest_float(PARAM_NAMES[i], float(lo[i]), float(hi[i]))
            for i in range(len(PARAM_NAMES))
        ])
        return ev.evaluate_vector(x)

    sampler = optuna.samplers.TPESampler(seed=int(seed) + abs(hash(scene)) % 9999)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return _pack_result_ext("TPE", ev, seed)


# ── cache helpers ─────────────────────────────────────────────────────────────
def _cache_path_ext(scene: str, method_key: str, seed: int) -> Path:
    if seed == 42:
        return CACHE_EXT_DIR / f"{scene}_{method_key}_b{BUDGET_EXT}.json"
    return CACHE_EXT_DIR / f"{scene}_{method_key}_b{BUDGET_EXT}_s{seed}.json"


def _load_cached_ext(scene: str, method_key: str, seed: int) -> dict | None:
    p = _cache_path_ext(scene, method_key, seed)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _save_cached_ext(result: dict, scene: str, seed: int) -> None:
    p = _cache_path_ext(scene, result["method_key"], seed)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one_ext(scene: str, method_key: str, seed: int) -> dict:
    dispatch = {
        "SPSA": method_spsa_ext,
        "GA": method_ga_ext,
        "CMA-ES": method_cmaes_ext,
        "TPE": method_tpe_ext,
    }
    if method_key not in dispatch:
        raise ValueError(f"Unknown method: {method_key}")
    return dispatch[method_key](scene, seed, BUDGET_EXT)


# ── CSV rebuild ───────────────────────────────────────────────────────────────
def rebuild_csv() -> Path:
    rows = []
    for p in sorted(CACHE_EXT_DIR.glob(f"*_b{BUDGET_EXT}*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        mkey = d.get("method_key", "")
        if mkey not in BASELINES_EXT:
            continue
        # parse scene and seed from filename
        stem = p.stem  # e.g. Tianjin_SPSA_b500  or  Tianjin_SPSA_b500_s101
        # strip _b500_s{seed} or _b500
        import re
        m = re.search(rf"_b{BUDGET_EXT}(?:_s(\d+))?$", stem)
        if not m:
            continue
        seed_val = int(m.group(1)) if m.group(1) else 42
        prefix = stem[:m.start()]
        scene_guess = prefix[: -len(f"_{mkey}")]
        for s, b in zip(d["convergence_sims"], d["convergence_best"]):
            rows.append({
                "scene": scene_guess,
                "method_key": mkey,
                "seed": seed_val,
                "sim_count": s,
                "best_error": b,
            })
    if not rows:
        print("[!] No extended cache files found.")
        return CACHE_EXT_DIR / "convergence_ext.csv"
    df = pd.DataFrame(rows)
    out = RES / "convergence_ext.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Rebuilt: {out} ({len(df)} rows)")
    return out


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baselines with extended budget for efficiency analysis."
    )
    parser.add_argument("--scene", type=str, default="")
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="",
                        help="Comma-separated seeds; overrides --seed")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-csv", action="store_true")
    args = parser.parse_args()

    if args.rebuild_csv:
        rebuild_csv()
        return

    scenes = [s.strip() for s in args.scene.split(",") if s.strip()] or SCENE_ORDER
    methods = [m.strip() for m in args.method.split(",") if m.strip()] or BASELINES_EXT
    seeds = ([int(s.strip()) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else [args.seed])

    print(f"Extended efficiency run  budget={BUDGET_EXT}")
    print(f"  scenes : {scenes}")
    print(f"  methods: {methods}")
    print(f"  seeds  : {seeds}")

    for scene in scenes:
        if scene not in SCENARIOS:
            print(f"[!] Unknown scene '{scene}', skip.")
            continue
        for mkey in methods:
            if mkey not in BASELINES_EXT:
                print(f"[!] Unknown method '{mkey}', skip.")
                continue
            for seed in seeds:
                tag = f"[{mkey} s={seed}]"
                if args.resume:
                    cached = _load_cached_ext(scene, mkey, seed)
                    if cached is not None:
                        print(f"  {scene} {tag} loaded cache", flush=True)
                        continue
                print(f"  {scene} {tag} running …", flush=True)
                t0 = time.time()
                res = run_one_ext(scene, mkey, seed)
                _save_cached_ext(res, scene, seed)
                print(f"  {scene} {tag} done in {time.time()-t0:.0f}s  "
                      f"best={res['final_error']:.4f}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
