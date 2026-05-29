"""run_comparison.py — SUMO 上的 BF-SAC 消融与 GA/SPSA 基线对比

  - 目标函数：J_b（feature_error，与 unified_calibration 一致）
  - 每次评估：一次 SUMO–TraCI 仿真
  - error_at_budget：累计 BUDGET_FAIR 次仿真时的最优 J_b（101，与 BF-SAC 一致）
  - 消融：No-RF（无代理，纯 LHS 101）/ No-LHS

运行（耗时较长，建议先单场景试跑）：
  python code/experiments/run_comparison.py --scenes XAM-N6 --methods BF-SAC,No-RF
  python code/experiments/run_comparison.py              # 六场景 × 全部方法
  python code/experiments/run_comparison.py --resume     # 跳过已有缓存（同预算后缀）

输出：
  outputs/results/comparison_summary.csv
  outputs/results/comparison_convergence.csv
  outputs/results/comparison_cache/<scene>_<method_key>.json
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
CAL_ROOT = PROJ / "code" / "calibration"
if str(CAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CAL_ROOT))

from unified_calibration import (  # noqa: E402
    BUDGET_SUMO,
    N_DOE,
    N_INIT,
    PARAM_NAMES,
    SCENARIOS,
    calibrate_one,
    feature_error,
    get_feature_weights,
    get_param_bounds,
    lhs_samples_for_scene,
    extract_real_features,
    _lhs_seed_for_scene,
    _run_sumo_traci,
)

RES = PROJ / "outputs" / "results"
CACHE_DIR = RES / "comparison_cache"
RES.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SCENE_ORDER = list(SCENARIOS.keys())
BUDGET_FAIR = BUDGET_SUMO  # 101：与 BF-SAC 总预算对齐
FAIL_ERROR = 10.0

METHOD_LABELS = {
    "BF-SAC": "BF-SAC（序贯LCB）",
    "No-RF": f"无RF（LHS {BUDGET_FAIR}）",
    "No-LHS": f"无LHS（随机 {BUDGET_FAIR}）",
    "SPSA": "SPSA",
    "GA": "遗传算法（GA）",
}

# 默认对比：BF-SAC（RF+序贯） vs 无代理 DoE vs 无 LHS vs GA/SPSA
DEFAULT_METHODS = ["BF-SAC", "No-RF", "No-LHS", "SPSA", "GA"]


def jb_to_score_proxy(jb: float) -> float:
    """J_b 线性映射为评分 proxy：max(0, 100 − 45·J_b)。"""
    return round(float(np.clip(100.0 - 45.0 * jb, 0.0, 100.0)), 2)


@dataclass
class SumoEvaluator:
    scene: str
    real_feat: np.ndarray
    weights: np.ndarray
    bounds: np.ndarray
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
        if self.sim_count >= 2000:
            return FAIL_ERROR
        params = {k: float(v) for k, v in zip(PARAM_NAMES, x)}
        try:
            sim_feat = _run_sumo_traci(self.scene, params)
            err = feature_error(self.real_feat, sim_feat, self.weights)
        except Exception as exc:
            print(f"       [!] SUMO failed: {exc}", flush=True)
            err = FAIL_ERROR
        self._log(err)
        return err

    def evaluate_dict(self, params: dict) -> float:
        x = np.array([params[k] for k in PARAM_NAMES], dtype=float)
        return self.evaluate_vector(x)

    def snapshot(self, max_sims: int | None = None) -> tuple[float, int]:
        if max_sims is None:
            return self.best_error, self.sim_count
        idx = next(
            (i for i, s in enumerate(self.sim_history) if s >= max_sims),
            len(self.best_history) - 1,
        )
        if idx < 0:
            return self.best_error, self.sim_count
        return float(self.best_history[idx]), int(self.sim_history[idx])


def _pack_result(
    method_key: str,
    ev: SumoEvaluator,
    budget_error: float,
    final_error: float,
    extra: dict | None = None,
) -> dict:
    out = {
        "method": METHOD_LABELS[method_key],
        "method_key": method_key,
        "budget_fair": BUDGET_FAIR,
        "final_error": round(final_error, 4),
        "error_at_budget": round(budget_error, 4),
        "score_proxy_at_budget": jb_to_score_proxy(budget_error),
        "score_proxy_final": jb_to_score_proxy(final_error),
        "n_sims": ev.sim_count,
        "convergence_sims": ev.sim_history,
        "convergence_best": ev.best_history,
    }
    if extra:
        out.update(extra)
    return out


def _result_from_calibration(
    method_key: str,
    result: dict,
    elapsed: float,
) -> dict:
    final = float(result["feature_error"])
    hist = result.get("doe_history") or []
    sims = [int(h["sim_index"]) for h in hist]
    bests = [float(h["best_so_far"]) for h in hist]
    err_b, _ = _error_at_budget(sims, bests, BUDGET_FAIR, final)
    n_sims = int(result.get("n_sumo_runs", len(sims) or BUDGET_FAIR))
    return {
        "method": METHOD_LABELS[method_key],
        "method_key": method_key,
        "budget_fair": BUDGET_FAIR,
        "final_error": round(final, 4),
        "error_at_budget": round(err_b, 4),
        "score_proxy_at_budget": jb_to_score_proxy(err_b),
        "score_proxy_final": jb_to_score_proxy(final),
        "n_sims": n_sims,
        "convergence_sims": sims,
        "convergence_best": bests,
        "elapsed_s": round(elapsed, 1),
        "backend": "sumo",
        "calibration_mode": result.get("calibration_mode"),
        "proxy_beats_doe": result.get("proxy_beats_doe"),
    }


def method_bfsac(scene: str) -> dict:
    """BF-SAC：序贯 LCB（calibrate_one）。"""
    print("       [BF-SAC] sequential LCB …", flush=True)
    t0 = time.time()
    result = calibrate_one(scene)
    return _result_from_calibration("BF-SAC", result, time.time() - t0)


def _error_at_budget(
    sims: list[int], bests: list[float], budget: int, fallback: float
) -> tuple[float, int]:
    if not sims:
        return fallback, 0
    idx = next((i for i, s in enumerate(sims) if s >= budget), len(sims) - 1)
    return float(bests[idx]), int(sims[idx])


def method_no_rf(scene: str, seed: int) -> dict:
    """无 RF：仅 LHS，预算内全部用于真仿真。"""
    n = BUDGET_FAIR
    ev = SumoEvaluator(
        scene,
        extract_real_features(scene),
        get_feature_weights(scene),
        get_param_bounds(scene),
    )
    X = lhs_samples_for_scene(n, ev.bounds, _lhs_seed_for_scene(scene))
    for i in range(n):
        ev.evaluate_vector(X[i])
        if (i + 1) % 10 == 0:
            print(f"       No-RF {i+1}/{n}  best={ev.best_error:.4f}", flush=True)
    err_b, _ = ev.snapshot(BUDGET_FAIR)
    return _pack_result("No-RF", ev, err_b, ev.best_error)


def method_no_lhs(scene: str, seed: int) -> dict:
    """无 LHS：均匀随机采样（故意降低空间覆盖）。"""
    n = BUDGET_FAIR
    ev = SumoEvaluator(
        scene,
        extract_real_features(scene),
        get_feature_weights(scene),
        get_param_bounds(scene),
    )
    rng = np.random.default_rng(seed + abs(hash(scene)) % 9999)
    lo, hi = ev.bounds[:, 0], ev.bounds[:, 1]
    scale = hi - lo
    X = rng.random((n, len(PARAM_NAMES))) * scale + lo
    for i in range(n):
        ev.evaluate_vector(X[i])
        if (i + 1) % 10 == 0:
            print(f"       No-LHS {i+1}/{n}  best={ev.best_error:.4f}", flush=True)
    err_b, _ = ev.snapshot(BUDGET_FAIR)
    return _pack_result("No-LHS", ev, err_b, ev.best_error)


def method_spsa(scene: str, seed: int, max_sims: int = 300) -> dict:
    """SPSA：每轮 2 次真仿真，记录至 max_sims。"""
    ev = SumoEvaluator(
        scene,
        extract_real_features(scene),
        get_feature_weights(scene),
        get_param_bounds(scene),
    )
    rng = np.random.default_rng(seed + abs(hash(scene)) % 9999)
    lo, hi = ev.bounds[:, 0], ev.bounds[:, 1]
    dim = len(PARAM_NAMES)

    theta = rng.random(dim) * (hi - lo) + lo
    ev.evaluate_vector(theta)

    a, c, A = 0.22, 0.15, 20.0
    ae, ge = 0.602, 0.101

    k = 1
    while ev.sim_count < max_sims:
        ak = a / (k + A) ** ae
        ck = c / k ** ge
        delta = rng.choice([-1.0, 1.0], size=dim)
        tp = np.clip(theta + ck * delta, lo, hi)
        tm = np.clip(theta - ck * delta, lo, hi)
        fp = ev.evaluate_vector(tp)
        fm = ev.evaluate_vector(tm)
        if ev.sim_count >= max_sims:
            break
        g = (fp - fm) / (2.0 * ck) * delta
        theta = np.clip(theta - ak * g, lo, hi)
        k += 1
        if k % 15 == 0:
            print(f"       SPSA sims={ev.sim_count}  best={ev.best_error:.4f}", flush=True)

    err_b, _ = ev.snapshot(BUDGET_FAIR)
    return _pack_result("SPSA", ev, err_b, ev.best_error)


def method_ga(scene: str, seed: int, n_pop: int = 15, n_gen: int = 50) -> dict:
    """GA：种群 n_pop，每代 n_pop 次评估，默认最多 15×51=765 次。"""
    ev = SumoEvaluator(
        scene,
        extract_real_features(scene),
        get_feature_weights(scene),
        get_param_bounds(scene),
    )
    rng = np.random.default_rng(seed + abs(hash(scene)) % 9999)
    lo, hi = ev.bounds[:, 0], ev.bounds[:, 1]
    dim = len(PARAM_NAMES)
    max_sims = n_pop * (n_gen + 1)

    pop = rng.random((n_pop, dim)) * (hi - lo) + lo
    fitness = np.array([ev.evaluate_vector(pop[i]) for i in range(n_pop)])

    for gen in range(n_gen):
        if ev.sim_count >= max_sims:
            break
        new_pop = np.zeros_like(pop)
        new_fit = np.zeros(n_pop)
        for i in range(n_pop):
            idx = rng.choice(n_pop, 3, replace=False)
            p1 = pop[idx[np.argmin(fitness[idx])]]
            idx2 = rng.choice(n_pop, 3, replace=False)
            p2 = pop[idx2[np.argmin(fitness[idx2])]]
            mask = rng.random(dim) < 0.5
            child = np.where(mask, p1, p2) + rng.normal(0, (hi - lo) * 0.05)
            new_pop[i] = np.clip(child, lo, hi)
            if ev.sim_count >= max_sims:
                new_fit[i] = FAIL_ERROR
                continue
            new_fit[i] = ev.evaluate_vector(new_pop[i])
        all_p = np.vstack([pop, new_pop])
        all_f = np.concatenate([fitness, new_fit])
        keep = np.argsort(all_f)[:n_pop]
        pop, fitness = all_p[keep], all_f[keep]
        if (gen + 1) % 5 == 0:
            print(f"       GA gen={gen+1} sims={ev.sim_count}  best={ev.best_error:.4f}", flush=True)

    err_b, _ = ev.snapshot(BUDGET_FAIR)
    return _pack_result("GA", ev, err_b, ev.best_error)


METHOD_FUNCS = {
    "BF-SAC": lambda scene, seed: method_bfsac(scene),
    "No-RF": method_no_rf,
    "No-LHS": method_no_lhs,
    "SPSA": method_spsa,
    "GA": method_ga,
}


def cache_path(scene: str, method_key: str) -> Path:
    return CACHE_DIR / f"{scene}_{method_key}_b{BUDGET_FAIR}.json"


def load_cached(scene: str, method_key: str) -> dict | None:
    p = cache_path(scene, method_key)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_cached(scene: str, result: dict) -> None:
    p = cache_path(scene, result["method_key"])
    slim = {k: v for k, v in result.items() if k not in ("convergence_sims", "convergence_best")}
    slim["convergence_sims"] = result["convergence_sims"]
    slim["convergence_best"] = result["convergence_best"]
    p.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")


def run_one(scene: str, method_key: str, seed: int = 42) -> dict:
    func = METHOD_FUNCS[method_key]
    if method_key == "BF-SAC":
        return func(scene, seed)
    return func(scene, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="BF-SAC 对比与消融（SUMO）")
    parser.add_argument(
        "--scenes", type=str, default="",
        help="逗号分隔场景名，默认全部六场景",
    )
    parser.add_argument(
        "--methods", type=str, default="",
        help="逗号分隔方法键，默认: " + ",".join(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="若缓存存在则跳过该场景-方法",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()] or SCENE_ORDER
    methods = [m.strip() for m in args.methods.split(",") if m.strip()] or DEFAULT_METHODS

    print("=" * 72)
    print("BF-SAC 对比与消融（SUMO）")
    print(f"  场景: {', '.join(scenes)}")
    print(f"  方法: {', '.join(methods)}")
    print(f"  公平预算: {BUDGET_FAIR} 次仿真时的最优 J_b")
    print("=" * 72)

    summary_rows: list[dict] = []
    conv_rows: list[dict] = []

    for scene in scenes:
        if scene not in SCENARIOS:
            print(f"[!] Unknown scene '{scene}', skip.")
            continue
        print(f"\n{'=' * 72}\n  Scene: {scene}\n{'=' * 72}")

        for mkey in methods:
            if mkey not in METHOD_FUNCS:
                print(f"  [!] Unknown method '{mkey}', skip.")
                continue

            if args.resume:
                cached = load_cached(scene, mkey)
                if cached is not None:
                    print(f"  [{mkey}] loaded cache", flush=True)
                    res = cached
                else:
                    print(f"  [{mkey}] running …", flush=True)
                    t0 = time.time()
                    res = run_one(scene, mkey, args.seed)
                    save_cached(scene, res)
                    print(f"  [{mkey}] done in {time.time()-t0:.0f}s", flush=True)
            else:
                print(f"  [{mkey}] running …", flush=True)
                t0 = time.time()
                res = run_one(scene, mkey, args.seed)
                save_cached(scene, res)
                print(f"  [{mkey}] done in {time.time()-t0:.0f}s", flush=True)

            eb = res["error_at_budget"]
            print(
                f"       {res['method']:28s}  err@{BUDGET_FAIR}={eb:.4f}  "
                f"final={res['final_error']:.4f}  sims={res['n_sims']}",
                flush=True,
            )
            summary_rows.append({
                "scene": scene,
                "method": res["method"],
                "method_key": mkey,
                "budget_fair": BUDGET_FAIR,
                "error_at_budget": eb,
                "final_error": res["final_error"],
                "score_proxy_at_budget": res.get(
                    "score_proxy_at_budget",
                    jb_to_score_proxy(eb),
                ),
                "score_proxy_final": res.get(
                    "score_proxy_final",
                    jb_to_score_proxy(res["final_error"]),
                ),
                "n_sims": res["n_sims"],
                "backend": "sumo",
            })
            for s, b in zip(res["convergence_sims"], res["convergence_best"]):
                conv_rows.append({
                    "scene": scene,
                    "method": res["method"],
                    "method_key": mkey,
                    "sim_count": s,
                    "best_error": b,
                })

    if not summary_rows:
        print("[!] No results.")
        return

    sum_path = RES / "comparison_summary.csv"
    conv_path = RES / "comparison_convergence.csv"
    pd.DataFrame(summary_rows).to_csv(sum_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(conv_rows).to_csv(conv_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {sum_path}")
    print(f"Saved: {conv_path}")

    df = pd.DataFrame(summary_rows)
    avg = df.groupby("method")[["error_at_budget", "score_proxy_at_budget", "n_sims"]].mean()
    avg = avg.sort_values("error_at_budget")
    print(f"\n场景平均（{BUDGET_FAIR} 次预算快照）：")
    print(avg.to_string())


if __name__ == "__main__":
    main()
