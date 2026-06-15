"""experiment_io.py — 多 seed 实验 cache 命名、解析与聚合工具。"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
RES = PROJ / "outputs" / "results"
CACHE_DIR = RES / "comparison_cache"

# 默认 seed=42 沿用无后缀 cache（向后兼容）；其余 seed 带 _s{seed}
LEGACY_DEFAULT_SEED = 42
# 正式 5 次重复实验计划使用的 seed 列表
REPLICATE_SEEDS = [42, 101, 202, 303, 404]
# 胶着场景试点（全量 5 seed 之前）
PILOT_SCENES = ["YTDJ", "Tianjin", "XAM-N6"]
PILOT_SEEDS = [42, 101, 202]

_CACHE_TAIL_RE = re.compile(r"_b(\d+)(?:_s(\d+))?\.json$")


def cache_filename(scene: str, method_key: str, budget: int, seed: int = LEGACY_DEFAULT_SEED) -> str:
    if seed == LEGACY_DEFAULT_SEED:
        return f"{scene}_{method_key}_b{budget}.json"
    return f"{scene}_{method_key}_b{budget}_s{seed}.json"


def cache_path(scene: str, method_key: str, budget: int, seed: int = LEGACY_DEFAULT_SEED) -> Path:
    return CACHE_DIR / cache_filename(scene, method_key, budget, seed)


def parse_cache_file(path: Path, method_keys: list[str] | None = None) -> dict | None:
    """从 cache 文件名解析 scene / method_key / budget / seed。"""
    m = _CACHE_TAIL_RE.search(path.name)
    if not m:
        return None
    budget = int(m.group(1))
    seed = int(m.group(2)) if m.group(2) else LEGACY_DEFAULT_SEED
    stem = path.name[: m.start()]
    if method_keys is None:
        method_keys = _guess_method_keys(stem)
    for mkey in sorted(method_keys, key=len, reverse=True):
        suffix = f"_{mkey}"
        if stem.endswith(suffix):
            return {
                "scene": stem[: -len(suffix)],
                "method_key": mkey,
                "budget": budget,
                "seed": seed,
                "path": path,
            }
    return None


def _guess_method_keys(stem: str) -> list[str]:
    """从 cache 目录推断 method_key 列表。"""
    keys: set[str] = set()
    for p in CACHE_DIR.glob("*.json"):
        meta = parse_cache_file(p, method_keys=[])
        if meta:
            continue
        m = _CACHE_TAIL_RE.search(p.name)
        if not m:
            continue
        s = p.name[: m.start()]
        for part in ("BF-SAC-MLP-n100", "BF-SAC-RF-n100", "BF-SAC-MLP-n80", "BF-SAC-RF-n80",
                     "BF-SAC-MLP-n60", "BF-SAC-RF-n60", "BF-SAC-MLP-n40", "BF-SAC-RF-n40",
                     "BF-SAC-MLP-n20", "BF-SAC-RF-n20", "BF-SAC-MLP", "BF-SAC-RF", "SPSA", "GA", "CMA-ES", "TPE"):
            if s.endswith(f"_{part}"):
                keys.add(part)
    return sorted(keys, key=len, reverse=True)


def load_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_cache_files(budget: int, method_keys: set[str] | None = None) -> list[dict]:
    """枚举 cache 目录，返回带 meta 的条目。"""
    if not CACHE_DIR.exists():
        return []
    all_keys: set[str] = set()
    for p in CACHE_DIR.glob(f"*_b{budget}*.json"):
        m = _CACHE_TAIL_RE.search(p.name)
        if not m or int(m.group(1)) != budget:
            continue
        stem = p.name[: m.start()]
        for suffix in ("BF-SAC-MLP-n100", "BF-SAC-RF-n100", "BF-SAC-MLP-n80", "BF-SAC-RF-n80",
                       "BF-SAC-MLP-n60", "BF-SAC-RF-n60", "BF-SAC-MLP-n40", "BF-SAC-RF-n40",
                       "BF-SAC-MLP-n20", "BF-SAC-RF-n20", "BF-SAC-MLP", "BF-SAC-RF", "SPSA", "GA", "CMA-ES", "TPE"):
            if stem.endswith(f"_{suffix}"):
                all_keys.add(suffix)
    keys_sorted = sorted(all_keys, key=len, reverse=True)
    rows: list[dict] = []
    for p in sorted(CACHE_DIR.glob(f"*_b{budget}*.json")):
        meta = parse_cache_file(p, keys_sorted)
        if meta is None:
            continue
        if method_keys and meta["method_key"] not in method_keys:
            continue
        data = load_cache(p)
        rows.append({**meta, "data": data})
    return rows


def load_per_run_summary(budget: int, method_keys: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for item in iter_cache_files(budget, method_keys):
        d = item["data"]
        eb = float(d.get("error_at_budget", d.get("final_error", 10.0)))
        rows.append({
            "scene": item["scene"],
            "method_key": item["method_key"],
            "seed": item["seed"],
            "error_at_budget": eb,
            "final_error": float(d.get("final_error", eb)),
            "n_init": d.get("n_init"),
            "score_proxy_at_budget": d.get("score_proxy_at_budget"),
            "cache_path": str(item["path"]),
        })
    return pd.DataFrame(rows)


def aggregate_multiseed_stats(
    df: pd.DataFrame,
    *,
    ci: float = 0.95,
) -> pd.DataFrame:
    """按 scene × method_key 聚合多 seed：mean / std / sem / ci_half。"""
    if df.empty:
        return df
    z = 1.96 if abs(ci - 0.95) < 1e-6 else 1.0

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        mean = float(g["error_at_budget"].mean())
        std = float(g["error_at_budget"].std(ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        return pd.Series({
            "n_seeds": n,
            "error_mean": mean,
            "error_std": std,
            "error_sem": sem,
            "error_ci_half": z * sem,
            "error_min": float(g["error_at_budget"].min()),
            "error_max": float(g["error_at_budget"].max()),
        })

    rows = []
    for (scene, mkey), g in df.groupby(["scene", "method_key"]):
        row = _agg(g)
        row["scene"] = scene
        row["method_key"] = mkey
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_global_method_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    """六场景算术平均 ± 跨场景标准差（用于表/柱图）。"""
    if stats_df.empty:
        return stats_df
    return (
        stats_df.groupby("method_key", as_index=False)
        .agg(
            n_scenes=("scene", "count"),
            error_mean=("error_mean", "mean"),
            error_std_across_scenes=("error_mean", "std"),
            error_sem=("error_mean", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
        )
    )


def aggregate_convergence_per_scene(
    conv_df: pd.DataFrame,
) -> pd.DataFrame:
    """按 scene × method × sim_count 聚合多 seed：均值 / seed 内 std。"""
    if conv_df.empty:
        return conv_df
    return (
        conv_df.groupby(["scene", "method_key", "sim_count"], as_index=False)
        .agg(
            best_error_mean=("best_error", "mean"),
            best_error_std=("best_error", "std"),
            n_runs=("best_error", "count"),
        )
        .sort_values(["scene", "method_key", "sim_count"])
    )


def aggregate_convergence_global(
    conv_df: pd.DataFrame,
    scenes: list[str] | None = None,
) -> pd.DataFrame:
    """六场景平均收敛曲线：先对每 seed 做场景算术平均，再对 seed 求 mean±std。"""
    if conv_df.empty:
        return conv_df
    scoped = conv_df.copy()
    if scenes is not None:
        scoped = scoped[scoped["scene"].isin(scenes)]
    if scoped.empty:
        return scoped

    if "run_seed" in scoped.columns:
        seed_curves = (
            scoped.groupby(["method_key", "run_seed", "sim_count"], as_index=False)
            .agg(scene_mean=("best_error", "mean"))
        )
        return (
            seed_curves.groupby(["method_key", "sim_count"], as_index=False)
            .agg(
                best_error_mean=("scene_mean", "mean"),
                best_error_std=("scene_mean", "std"),
                n_runs=("scene_mean", "count"),
            )
            .sort_values(["method_key", "sim_count"])
        )

    return (
        scoped.groupby(["method_key", "sim_count"], as_index=False)
        .agg(
            best_error_mean=("best_error", "mean"),
            best_error_std=("best_error", "std"),
            n_runs=("best_error", "count"),
        )
        .sort_values(["method_key", "sim_count"])
    )


def write_multiseed_outputs(
    per_run: pd.DataFrame,
    budget: int,
    out_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    out_dir = out_dir or RES
    out_dir.mkdir(parents=True, exist_ok=True)
    per_path = out_dir / "comparison_per_run.csv"
    stats_path = out_dir / "comparison_multiseed_stats.csv"
    global_path = out_dir / "comparison_multiseed_global.csv"

    per_run.to_csv(per_path, index=False, encoding="utf-8-sig")
    stats = aggregate_multiseed_stats(per_run)
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
    global_stats = aggregate_global_method_stats(stats)
    global_stats.to_csv(global_path, index=False, encoding="utf-8-sig")
    return per_path, stats_path, global_path
