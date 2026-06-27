"""统一对比实验：主对比 + N_init 样本效率消融。

主对比（--mode main，默认）：FLAT-RF / FLAT-MLP（cache 键 BF-SAC-RF / BF-SAC-MLP；
n_init=40，κ 2→0.5 退火）对四个异构基线 SPSA / GA / CMA-ES / TPE，统一 100 次 SUMO 预算。

样本效率消融（--mode sweep）：FLAT-RF / FLAT-MLP 在 n_init ∈ {20,40,60,80,100}
下扫描（总预算恒为 100，n_init=100 即纯 LHS、无序贯加点）。

多 seed：seed=42 写入无后缀 cache，其余 seed 为 *_s{seed}.json；所有方法共用同一
--seed / --seeds 列表，FLAT 的 LHS/序贯/代理随机源均随 run_seed 变化。各方法的
cache 路径互不重叠，可在多终端按 --methods 拆分并行。

示例：
  python code/experiments/run_comparison.py --scenes Tianjin --mode main --resume
  python code/experiments/run_comparison.py --seeds 42,101,202,303,404 --mode sweep --resume
  python code/experiments/run_comparison.py --rebuild-csv --mode main
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
for p in (EXP_ROOT, CAL_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiment_io import (  # noqa: E402
    LEGACY_DEFAULT_SEED,
    cache_path as seeded_cache_path,
    write_multiseed_outputs,
    load_per_run_summary,
)

from unified_calibration import (  # noqa: E402
    BUDGET_SUMO,
    N_INIT_MAIN,
    PARAM_NAMES,
    SCENARIOS,
    SWEEP_N_INIT,
    calibrate_one,
    feature_error,
    get_feature_weights,
    get_param_bounds,
    extract_real_features,
    _run_sumo_traci,
)

RES = PROJ / "outputs" / "results"
CACHE_DIR = RES / "comparison_cache"
RES.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SCENE_ORDER = list(SCENARIOS.keys())
BUDGET_FAIR = BUDGET_SUMO
FAIL_ERROR = 10.0
GA_POP = 10

MAIN_METHODS = ["BF-SAC-RF", "BF-SAC-MLP", "SPSA", "GA", "CMA-ES", "TPE"]

if BUDGET_FAIR % GA_POP != 0:
    raise RuntimeError(f"BUDGET_FAIR={BUDGET_FAIR} 须能被 GA_POP={GA_POP} 整除")


def _bfsac_key(surrogate: str, n_init: int) -> str:
    base = "BF-SAC-RF" if surrogate == "rf" else "BF-SAC-MLP"
    if n_init == N_INIT_MAIN:
        return base
    return f"{base}-n{n_init}"


def _parse_bfsac_key(mkey: str) -> tuple[str, int] | None:
    if mkey == "BF-SAC-RF":
        return "rf", N_INIT_MAIN
    if mkey == "BF-SAC-MLP":
        return "mlp", N_INIT_MAIN
    if mkey.startswith("BF-SAC-RF-n"):
        return "rf", int(mkey.split("-n")[-1])
    if mkey.startswith("BF-SAC-MLP-n"):
        return "mlp", int(mkey.split("-n")[-1])
    return None


def _method_label(mkey: str) -> str:
    parsed = _parse_bfsac_key(mkey)
    if parsed:
        surr, n_init = parsed
        tag = "RF" if surr == "rf" else "MLP"
        if n_init == BUDGET_FAIR:
            return f"BF-SAC-{tag}（纯 LHS {n_init}）"
        if n_init == N_INIT_MAIN:
            return f"BF-SAC-{tag}"
        return f"BF-SAC-{tag}（n_init={n_init}）"
    return {
        "SPSA": "SPSA",
        "GA": "遗传算法（GA）",
        "CMA-ES": "CMA-ES",
        "TPE": "TPE",
    }.get(mkey, mkey)


def sweep_method_keys() -> list[str]:
    keys: list[str] = []
    for n_init in SWEEP_N_INIT:
        keys.append(_bfsac_key("rf", n_init))
        keys.append(_bfsac_key("mlp", n_init))
    return keys


def sweep_multiseed_method_keys() -> list[str]:
    """sweep 中需独立补跑多 seed 的键（n_init=40 与主对比共用 BF-SAC-RF/MLP cache）。"""
    return [k for k in sweep_method_keys() if k not in ("BF-SAC-RF", "BF-SAC-MLP")]


def methods_for_mode(mode: str) -> list[str]:
    if mode == "main":
        return MAIN_METHODS.copy()
    if mode == "sweep":
        return sweep_method_keys()
    if mode == "all":
        return MAIN_METHODS + [k for k in sweep_method_keys() if k not in MAIN_METHODS]
    raise ValueError(f"未知 mode: {mode}")


ALL_METHOD_KEYS = methods_for_mode("all")


def jb_to_score_proxy(jb: float) -> float:
    return round(float(np.clip(100.0 - 45.0 * jb, 0.0, 100.0)), 2)


def _make_evaluator(scene: str) -> "SumoEvaluator":
    return SumoEvaluator(
        scene,
        extract_real_features(scene),
        get_feature_weights(scene),
        get_param_bounds(scene),
        budget=BUDGET_FAIR,
    )


def _trim_convergence(sims: list[int], bests: list[float]) -> tuple[list[int], list[float]]:
    out_s, out_b = [], []
    for s, b in zip(sims, bests):
        if s > BUDGET_FAIR:
            break
        out_s.append(int(s))
        out_b.append(float(b))
    return out_s, out_b


def _error_at_budget(sims: list[int], bests: list[float]) -> float:
    if not sims:
        return FAIL_ERROR
    idx = next((i for i, s in enumerate(sims) if s >= BUDGET_FAIR), len(sims) - 1)
    return float(bests[idx])


@dataclass
class SumoEvaluator:
    scene: str
    real_feat: np.ndarray
    weights: np.ndarray
    bounds: np.ndarray
    budget: int = BUDGET_FAIR
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
            raise RuntimeError(f"仿真预算已用尽（{self.budget}）")
        params = {k: float(v) for k, v in zip(PARAM_NAMES, x)}
        try:
            sim_feat = _run_sumo_traci(self.scene, params)
            err = feature_error(self.real_feat, sim_feat, self.weights)
        except Exception as exc:
            print(f"       [!] SUMO failed: {exc}", flush=True)
            err = FAIL_ERROR
        self._log(err)
        return err


def _pack_result(method_key: str, ev: SumoEvaluator, extra: dict | None = None) -> dict:
    if ev.sim_count != BUDGET_FAIR:
        raise RuntimeError(f"{method_key}: 期望 {BUDGET_FAIR} 次仿真，实际 {ev.sim_count}")
    sims, bests = _trim_convergence(ev.sim_history, ev.best_history)
    err_b = _error_at_budget(sims, bests)
    out = {
        "method": _method_label(method_key),
        "method_key": method_key,
        "budget_fair": BUDGET_FAIR,
        "final_error": round(ev.best_error, 4),
        "error_at_budget": round(err_b, 4),
        "score_proxy_at_budget": jb_to_score_proxy(err_b),
        "score_proxy_final": jb_to_score_proxy(ev.best_error),
        "n_sims": BUDGET_FAIR,
        "convergence_sims": sims,
        "convergence_best": bests,
    }
    if extra:
        out.update(extra)
    return out


def _result_from_calibration(
    method_key: str, result: dict, elapsed: float, n_init: int,
) -> dict:
    n_sims = int(result.get("n_sumo_runs", 0))
    if n_sims != BUDGET_FAIR:
        raise RuntimeError(f"{method_key}: n_sumo_runs={n_sims}，期望 {BUDGET_FAIR}")
    hist = result.get("doe_history") or []
    sims = [int(h["sim_index"]) for h in hist]
    bests = [float(h["best_so_far"]) for h in hist]
    sims, bests = _trim_convergence(sims, bests)
    err_b = _error_at_budget(sims, bests)
    final = float(result["feature_error"])
    return {
        "method": _method_label(method_key),
        "method_key": method_key,
        "n_init": n_init,
        "budget_fair": BUDGET_FAIR,
        "final_error": round(final, 4),
        "error_at_budget": round(err_b, 4),
        "doe_best_error": round(float(result.get("doe_best_error", final)), 4),
        "score_proxy_at_budget": jb_to_score_proxy(err_b),
        "score_proxy_final": jb_to_score_proxy(final),
        "n_sims": BUDGET_FAIR,
        "convergence_sims": sims,
        "convergence_best": bests,
        "elapsed_s": round(elapsed, 1),
        "backend": "sumo",
        "calibration_mode": result.get("calibration_mode"),
        "proxy_beats_doe": result.get("proxy_beats_doe"),
        "best_params": result.get("calibrated_params"),
        "run_seed": result.get("run_seed"),
    }


def method_bfsac(scene: str, surrogate: str, n_init: int, seed: int) -> dict:
    mkey = _bfsac_key(surrogate, n_init)
    print(f"       [{mkey}] n_init={n_init} seed={seed} …", flush=True)
    t0 = time.time()
    result = calibrate_one(scene, surrogate_kind=surrogate, n_init=n_init, run_seed=seed)
    out = _result_from_calibration(mkey, result, time.time() - t0, n_init)
    out["run_seed"] = int(seed)
    if surrogate == "mlp":
        out["surrogate"] = result.get("surrogate")
        out["surrogate_config"] = result.get("surrogate_config")
    return out


def method_spsa(scene: str, seed: int) -> dict:
    ev = _make_evaluator(scene)
    rng = np.random.default_rng(seed + abs(hash(scene)) % 9999)
    lo, hi = ev.bounds[:, 0], ev.bounds[:, 1]
    dim = len(PARAM_NAMES)
    theta = rng.random(dim) * (hi - lo) + lo
    ev.evaluate_vector(theta)

    a, c, A = 0.22, 0.15, 20.0
    ae, ge = 0.602, 0.101
    k = 1
    while ev.sim_count + 2 <= BUDGET_FAIR:
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
        if k % 15 == 0:
            print(f"       SPSA {ev.sim_count}/{BUDGET_FAIR}  best={ev.best_error:.4f}", flush=True)

    while ev.sim_count < BUDGET_FAIR:
        ev.evaluate_vector(theta)
    return _pack_result("SPSA", ev, {"run_seed": int(seed)})


def _tournament(pop: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(pop), 3, replace=False)
    return pop[idx[np.argmin(fitness[idx])]]


def method_ga(scene: str, seed: int) -> dict:
    ev = _make_evaluator(scene)
    rng = np.random.default_rng(seed + abs(hash(scene)) % 9999)
    lo, hi = ev.bounds[:, 0], ev.bounds[:, 1]
    dim = len(PARAM_NAMES)
    n_waves = BUDGET_FAIR // GA_POP
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
        if (wave + 1) % 3 == 0:
            print(f"       GA wave={wave+1}/{n_waves}  best={ev.best_error:.4f}", flush=True)
    return _pack_result("GA", ev, {"run_seed": int(seed)})


def method_cmaes(scene: str, seed: int) -> dict:
    """CMA-ES（演化策略）：在归一化 [0,1] 空间搜索，固定 100 次仿真预算。"""
    import cma  # 延迟导入

    ev = _make_evaluator(scene)
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
         "verbose": -9, "verb_log": 0, "verb_disp": 0, "maxfevals": BUDGET_FAIR},
    )
    best_x, best_f = scale(x0), float("inf")
    while ev.sim_count < BUDGET_FAIR and not es.stop():
        sols = es.ask()
        fits, complete = [], True
        for z in sols:
            if ev.sim_count >= BUDGET_FAIR:
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
        if es.countiter % 3 == 0:
            print(f"       CMA-ES {ev.sim_count}/{BUDGET_FAIR}  best={ev.best_error:.4f}", flush=True)
    while ev.sim_count < BUDGET_FAIR:
        ev.evaluate_vector(best_x)
    return _pack_result("CMA-ES", ev, {"run_seed": int(seed)})


def method_tpe(scene: str, seed: int) -> dict:
    """TPE（贝叶斯密度估计，Optuna）：固定 100 次试验 = 100 次仿真。"""
    import optuna  # 延迟导入

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ev = _make_evaluator(scene)
    lo, hi = ev.bounds[:, 0], ev.bounds[:, 1]

    def objective(trial: "optuna.Trial") -> float:
        x = np.array([
            trial.suggest_float(PARAM_NAMES[i], float(lo[i]), float(hi[i]))
            for i in range(len(PARAM_NAMES))
        ])
        return ev.evaluate_vector(x)

    sampler = optuna.samplers.TPESampler(seed=int(seed) + abs(hash(scene)) % 9999)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=BUDGET_FAIR, show_progress_bar=False)
    return _pack_result("TPE", ev, {"run_seed": int(seed)})


def run_one(scene: str, method_key: str, seed: int = LEGACY_DEFAULT_SEED) -> dict:
    parsed = _parse_bfsac_key(method_key)
    if parsed:
        return method_bfsac(scene, parsed[0], parsed[1], seed)
    if method_key == "SPSA":
        return method_spsa(scene, seed)
    if method_key == "GA":
        return method_ga(scene, seed)
    if method_key == "CMA-ES":
        return method_cmaes(scene, seed)
    if method_key == "TPE":
        return method_tpe(scene, seed)
    raise ValueError(f"未知方法: {method_key}")


def cache_path(scene: str, method_key: str, seed: int = LEGACY_DEFAULT_SEED) -> Path:
    return seeded_cache_path(scene, method_key, BUDGET_FAIR, seed)


def load_cached(scene: str, method_key: str, seed: int = LEGACY_DEFAULT_SEED) -> dict | None:
    p = cache_path(scene, method_key, seed)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_cached(scene: str, result: dict, seed: int = LEGACY_DEFAULT_SEED) -> None:
    mkey = result["method_key"]
    p = cache_path(scene, mkey, seed)
    result = {**result, "run_seed": int(seed)}
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _rows_from_result(scene: str, mkey: str, res: dict) -> tuple[dict, list[dict]]:
    eb = res["error_at_budget"]
    summary = {
        "scene": scene,
        "method": _method_label(mkey),
        "method_key": mkey,
        "budget_fair": BUDGET_FAIR,
        "error_at_budget": eb,
        "final_error": res["final_error"],
        "score_proxy_at_budget": res.get("score_proxy_at_budget", jb_to_score_proxy(eb)),
        "score_proxy_final": res.get("score_proxy_final", jb_to_score_proxy(res["final_error"])),
        "n_sims": BUDGET_FAIR,
        "n_init": res.get("n_init"),
        "run_seed": res.get("run_seed"),
        "backend": res.get("backend", "sumo"),
    }
    label = _method_label(mkey)
    sims, bests = _trim_convergence(res["convergence_sims"], res["convergence_best"])
    run_seed = res.get("run_seed")
    conv = [
        {
            "scene": scene,
            "method": label,
            "method_key": mkey,
            "sim_count": s,
            "best_error": b,
            "run_seed": run_seed,
        }
        for s, b in zip(sims, bests)
    ]
    return summary, conv


def _parse_seeds_arg(text: str, fallback: int) -> list[int]:
    if not text.strip():
        return [fallback]
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def rebuild_csv_from_cache(mode: str = "all") -> None:
    mkeys = set(methods_for_mode(mode))
    summary_rows: list[dict] = []
    conv_rows: list[dict] = []
    from experiment_io import iter_cache_files, parse_cache_file

    keys_sorted = sorted(mkeys, key=len, reverse=True)
    for p in sorted(CACHE_DIR.glob(f"*_b{BUDGET_FAIR}*.json")):
        meta = parse_cache_file(p, keys_sorted)
        if meta is None or meta["method_key"] not in mkeys:
            continue
        res = json.loads(p.read_text(encoding="utf-8"))
        mkey = res.get("method_key") or meta["method_key"]
        scene = meta["scene"]
        res["run_seed"] = meta["seed"]
        srow, crows = _rows_from_result(scene, mkey, res)
        srow["run_seed"] = meta["seed"]
        summary_rows.append(srow)
        conv_rows.extend(crows)
    if not summary_rows:
        print("[!] No cache files found.")
        return
    tag = "main" if mode == "main" else ("sweep" if mode == "sweep" else "all")
    sum_path = RES / f"comparison_summary_{tag}.csv"
    conv_path = RES / f"comparison_convergence_{tag}.csv"
    pd.DataFrame(summary_rows).to_csv(sum_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(conv_rows).to_csv(conv_path, index=False, encoding="utf-8-sig")
    if mode == "main":
        pd.DataFrame(summary_rows).to_csv(RES / "comparison_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(conv_rows).to_csv(RES / "comparison_convergence.csv", index=False, encoding="utf-8-sig")
    print(f"Rebuilt: {sum_path} ({len(summary_rows)} rows)")
    print(f"Rebuilt: {conv_path} ({len(conv_rows)} rows)")
    per_run = load_per_run_summary(BUDGET_FAIR, mkeys)
    if not per_run.empty:
        p1, p2, p3 = write_multiseed_outputs(per_run, BUDGET_FAIR)
        print(f"Multiseed: {p1.name} ({len(per_run)} runs)")
        print(f"Multiseed: {p2.name}, {p3.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BF-SAC ICTAI 对比实验（SUMO）")
    parser.add_argument("--scenes", type=str, default="", help="逗号分隔场景名")
    parser.add_argument("--methods", type=str, default="", help="逗号分隔方法键")
    parser.add_argument(
        "--mode", choices=["main", "sweep", "all"], default="main",
        help="main=主对比六方法；sweep=N_init 消融；all=全部",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=LEGACY_DEFAULT_SEED)
    parser.add_argument("--seeds", type=str, default="", help="逗号分隔多 seed，覆盖 --seed")
    parser.add_argument("--rebuild-csv", action="store_true")
    args = parser.parse_args()

    if args.rebuild_csv:
        rebuild_csv_from_cache(args.mode)
        return

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()] or SCENE_ORDER
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or methods_for_mode(args.mode)
    seeds = _parse_seeds_arg(args.seeds, args.seed)

    print("=" * 72)
    print("BF-SAC ICTAI 对比实验")
    print(f"  mode: {args.mode}")
    print(f"  场景: {', '.join(scenes)}")
    print(f"  方法: {', '.join(methods)}")
    print(f"  seeds: {seeds}")
    print(f"  预算: {BUDGET_FAIR} 次 SUMO / 方法 / seed")
    print("=" * 72)

    summary_rows: list[dict] = []
    conv_rows: list[dict] = []

    for scene in scenes:
        if scene not in SCENARIOS:
            print(f"[!] Unknown scene '{scene}', skip.")
            continue
        print(f"\n{'=' * 72}\n  Scene: {scene}\n{'=' * 72}")

        for mkey in methods:
            for run_seed in seeds:
                tag = f"[{mkey} s={run_seed}]"
                if args.resume:
                    cached = load_cached(scene, mkey, run_seed)
                    if cached is not None:
                        print(f"  {tag} loaded cache", flush=True)
                        res = cached
                    else:
                        print(f"  {tag} running …", flush=True)
                        t0 = time.time()
                        res = run_one(scene, mkey, run_seed)
                        save_cached(scene, res, run_seed)
                        print(f"  {tag} done in {time.time()-t0:.0f}s", flush=True)
                else:
                    print(f"  {tag} running …", flush=True)
                    t0 = time.time()
                    res = run_one(scene, mkey, run_seed)
                    save_cached(scene, res, run_seed)
                    print(f"  {tag} done in {time.time()-t0:.0f}s", flush=True)

                eb = res["error_at_budget"]
                print(
                    f"       {res['method']:32s}  err@{BUDGET_FAIR}={eb:.4f}  "
                    f"sims={res['n_sims']}  seed={run_seed}",
                    flush=True,
                )
                res["run_seed"] = run_seed
                srow, crows = _rows_from_result(scene, mkey, res)
                srow["run_seed"] = run_seed
                summary_rows.append(srow)
                conv_rows.extend(crows)

    if not summary_rows:
        print("[!] No results.")
        return

    tag = args.mode
    sum_path = RES / (f"comparison_summary_{tag}.csv" if tag != "main" else "comparison_summary.csv")
    conv_path = RES / (f"comparison_convergence_{tag}.csv" if tag != "main" else "comparison_convergence.csv")

    new_sum = pd.DataFrame(summary_rows)
    new_conv = pd.DataFrame(conv_rows)
    if sum_path.exists():
        old_sum = pd.read_csv(sum_path)
        if "run_seed" in new_sum.columns:
            keys = list(zip(new_sum["scene"], new_sum["method_key"], new_sum["run_seed"]))
            old_sum = old_sum[
                ~old_sum.apply(
                    lambda r: (r["scene"], r["method_key"], r.get("run_seed", 42)) in keys,
                    axis=1,
                )
            ]
        else:
            keys = list(zip(new_sum["scene"], new_sum["method_key"]))
            old_sum = old_sum[
                ~old_sum.apply(lambda r: (r["scene"], r["method_key"]) in keys, axis=1)
            ]
        new_sum = pd.concat([old_sum, new_sum], ignore_index=True)
    if conv_path.exists():
        old_conv = pd.read_csv(conv_path)
        if "run_seed" in new_conv.columns:
            keys = list(zip(new_conv["scene"], new_conv["method_key"], new_conv["run_seed"]))
            old_conv = old_conv[
                ~old_conv.apply(
                    lambda r: (r["scene"], r["method_key"], r.get("run_seed", 42)) in keys,
                    axis=1,
                )
            ]
        else:
            keys = list(zip(new_conv["scene"], new_conv["method_key"]))
            old_conv = old_conv[
                ~old_conv.apply(lambda r: (r["scene"], r["method_key"]) in keys, axis=1)
            ]
        new_conv = pd.concat([old_conv, new_conv], ignore_index=True)

    new_sum.to_csv(sum_path, index=False, encoding="utf-8-sig")
    new_conv.to_csv(conv_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {sum_path} ({len(new_sum)} rows)")
    print(f"Saved: {conv_path} ({len(new_conv)} rows)")
    mkeys = set(methods)
    per_run = load_per_run_summary(BUDGET_FAIR, mkeys)
    if not per_run.empty:
        p1, p2, p3 = write_multiseed_outputs(per_run, BUDGET_FAIR)
        print(f"Multiseed: {p1.name}, {p2.name}, {p3.name}")


if __name__ == "__main__":
    main()
