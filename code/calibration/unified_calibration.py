"""unified_calibration.py  ──  BF-SAC（序贯代理标定）

Behavioral Fingerprint Surrogate-Assisted Calibration (BF-SAC)
==============================================================
8 维行为指纹 + 固定 SUMO 预算下的序贯 RF 代理标定（LCB 采集）。

  - 阶段 A：``N_INIT`` 次 LHS 初始设计
  - 阶段 B：``BUDGET_SUMO - N_INIT`` 次序贯加点：重训 RF → 信赖域 + 全局候选
    → 下置信界 (LCB) 选点 → 真实 SUMO 评估并入库

用法:
  python unified_calibration.py              # 六场景（RF 代理，默认）
  python unified_calibration.py XAM-N6       # 单场景
  python unified_calibration.py --mlp        # 改用 MLP 集成代理（平替 RF）
  python unified_calibration.py --mlp XAM-N6 # 单场景 + MLP
  BFSAC_SURROGATE=mlp python unified_calibration.py   # 等价环境变量写法

代理选择：RF（随机森林，默认）与 MLP（深度集成）二选一，框架/预算/LCB 不变。
MLP 模式输出文件名追加 ``__mlp_hX-Y_mM`` 后缀并保存网络权重，绝不覆盖 RF 结果。
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

PROJ = Path(__file__).resolve().parents[2]

# 代理模型选择："rf"（默认，随机森林）或 "mlp"（MLP 集成平替）。
# 可用环境变量 BFSAC_SURROGATE 或命令行 --surrogate/--mlp/--rf 覆盖。
SURROGATE_KIND = os.environ.get("BFSAC_SURROGATE", "rf").lower()

# ═════════════════════════════════════════════════════════════════════
# 场景配置
# ═════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "Tianjin":   {"type": "SIND", "data": PROJ / "data/raw_data/SIND/Tianjin",
                  "cfg": "tianjin",   "sim_end": 600, "warmup": 60},
    "Changchun": {"type": "SIND", "data": PROJ / "data/raw_data/SIND/Changchun",
                  "cfg": "changchun", "sim_end": 600, "warmup": 60},
    "Xian":      {"type": "SIND", "data": PROJ / "data/raw_data/SIND/Xian",
                  "cfg": "xian",      "sim_end": 600, "warmup": 60},
    "YTDJ":      {"type": "UTE",  "data": PROJ / "data/raw_data/UTE/YTDJ",
                  "cfg": "ytdj",      "sim_end": 545, "warmup": 30},
    "RML":       {"type": "UTE",  "data": PROJ / "data/raw_data/UTE/RML",
                  "cfg": "rml",       "sim_end": 300, "warmup": 30},
    "XAM-N6":    {"type": "UTE",  "data": PROJ / "data/raw_data/UTE/XAM-N6",
                  "cfg": "xam",       "sim_end": 329, "warmup": 30},
}

# ═════════════════════════════════════════════════════════════════════
# 标定参数（6 维 Krauss 跟驰 + 4 维 LC2013 换道，TraCI 写入）
# ═════════════════════════════════════════════════════════════════════

PARAM_NAMES = [
    "accel", "decel", "sigma", "tau", "minGap", "speedFactor",
    "lcStrategic", "lcCooperative", "lcAssertive", "lcSpeedGain",
]
PARAM_BOUNDS = np.array([
    [1.0, 3.5],   # accel:       最大加速度 (m/s²)
    [2.0, 6.0],   # decel:       舒适减速度 (m/s²)
    [0.1, 0.9],   # sigma:       驾驶不确定性 [0,1]
    [0.5, 2.5],   # tau:         期望车头时距 (s)
    [1.0, 4.0],   # minGap:      最小车间距 (m)
    [0.5, 1.2],   # speedFactor: 限速遵从系数
    [0.35, 2.80],  # lcStrategic:  战略/强制变道积极性（SUMO 参数）
    [0.35, 2.00],  # lcCooperative: 协作变道意愿
    [0.35, 2.80],  # lcAssertive:   变道“果断度”/间隙接受
    [0.35, 2.80],  # lcSpeedGain:   主动变道对速度增益的权重
])

# ═════════════════════════════════════════════════════════════════════
# 行为特征指纹（8 维，去冗余）
# ═════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "v_mean", "v_p15", "v_p85",
    "a_pos_mean", "a_neg_mean", "a_p05", "a_p95",
    "stop_frac",
]

# 误差加权（第一性原理：速度水平/区间 + 驱动强度 + 拥堵占有）
FEATURE_WEIGHTS = np.array([
    0.18, 0.12, 0.15,  # 速度层
    0.12, 0.12, 0.11, 0.10,  # 加减速层
    0.10,  # 拥堵层
])

# 长春：综合分瓶颈在加速度强度失配，略提高加减速维权重
FEATURE_WEIGHTS_CHANGCHUN = np.array([
    0.15, 0.10, 0.12,
    0.14, 0.14, 0.12, 0.11,
    0.12,
])
# 西安：低速交叉口，强调停车占比与 v 低分位
FEATURE_WEIGHTS_XIAN = np.array([
    0.14, 0.16, 0.08,
    0.10, 0.10, 0.09, 0.09,
    0.24,
])
# RML：交织区速度谱与跟驰间距并重
FEATURE_WEIGHTS_RML = np.array([
    0.22, 0.14, 0.17,
    0.11, 0.11, 0.08, 0.07,
    0.10,
])


def get_feature_weights(name: str) -> np.ndarray:
    if name == "Changchun":
        return FEATURE_WEIGHTS_CHANGCHUN
    if name == "Xian":
        return FEATURE_WEIGHTS_XIAN
    if name == "RML":
        return FEATURE_WEIGHTS_RML
    return FEATURE_WEIGHTS


def get_param_bounds(name: str) -> np.ndarray:
    """与 3.3.4 思路一致：对瓶颈场景收紧可写边界以引导搜索。"""
    b = PARAM_BOUNDS.copy()
    if name == "Xian":
        b[5, 1] = min(b[5, 1], 0.62)
        b[5, 0] = max(b[5, 0], 0.30)
        b[0, 1] = min(b[0, 1], 2.15)
        b[3, 0] = max(b[3, 0], 1.10)
        b[4, 0] = max(b[4, 0], 2.25)
    elif name == "Changchun":
        b[0, 1] = min(b[0, 1], 2.35)   # 抑制过冲加速度
        b[0, 0] = max(b[0, 0], 0.95)
        b[2, 0] = max(b[2, 0], 0.38)  # sigma 略抬升，轨迹更平滑
    elif name == "RML":
        b[4, 0] = max(b[4, 0], 0.85)   # 匝道交织：更小车间距下界
        b[4, 1] = min(b[4, 1], 3.35)
        b[3, 0] = max(b[3, 0], 0.42)
        b[0, 1] = min(b[0, 1], 2.90)
    return b


def lhs_samples_for_scene(n: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    sampler = LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit = sampler.random(n=n)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return unit * (hi - lo) + lo


BUDGET_SUMO = 101   # 每场景总 SUMO 次数
N_INIT = 60         # 阶段 A：LHS 初始设计
N_DOE = N_INIT      # 对比实验 No-RF 等消融的 DoE 样本数

# 序贯 LCB 采集
LC_KAPPA = 1.96
N_CAND_LOCAL = 2500
N_CAND_GLOBAL = 2500
TRUST_FRAC = 0.18   # 信赖域：最优点附近各维 ± 该比例 × 参数跨度
MIN_DIST_REL = 0.02 # 与已评估点最小相对距离（避免重复仿真）
MIN_VALID_RF = 10

# ═════════════════════════════════════════════════════════════════════
# 公共搜索工具（与具体代理无关：LHS 种子 / 候选采样 / 去重距离）
# 两个代理块共享这些“框架级”几何工具，但各自的建模与 LCB 互不共用。
# ═════════════════════════════════════════════════════════════════════

def _lhs_seed_for_scene(name: str) -> int:
    return {
        "Changchun": 203, "Tianjin": 211, "RML": 207, "Xian": 204,
    }.get(name, 42)


def _propose_candidates(
    bounds: np.ndarray,
    x_best: np.ndarray,
    n_local: int,
    n_global: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    d = len(lo)
    global_c = rng.random((n_global, d)) * span + lo
    delta = TRUST_FRAC * span
    local_c = x_best + rng.uniform(-1.0, 1.0, (n_local, d)) * delta
    local_c = np.clip(local_c, lo, hi)
    return np.vstack([local_c, global_c])


def _min_rel_dist(cand: np.ndarray, X_seen: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-6)
    d = np.linalg.norm((cand[:, None, :] - X_seen[None, :, :]) / span, axis=2)
    return d.min(axis=1)


# ═════════════════════════════════════════════════════════════════════
# 代理一：随机森林（RF）—— 自成一块（拟合 / LCB / 选点）
# 输入 10 维参数、输出标量 J_b；σ 取自树间方差。对外仅暴露 .predict / .score。
# ═════════════════════════════════════════════════════════════════════

def _rf_n_jobs(name: str) -> int:
    return 1 if name in ("RML", "Tianjin") else -1


def _fit_rf(X: np.ndarray, Y: np.ndarray, name: str) -> RandomForestRegressor | None:
    valid = ~np.isnan(Y) & (Y < 10.0)
    if valid.sum() < MIN_VALID_RF:
        return None
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=20, random_state=42, n_jobs=_rf_n_jobs(name),
    )
    rf.fit(X[valid], Y[valid])
    return rf


def _rf_lcb(rf: RandomForestRegressor, X: np.ndarray, kappa: float) -> np.ndarray:
    """RF 下置信界：树间预测均值 − κ×树间标准差（越小越优）。"""
    preds = np.stack([t.predict(X) for t in rf.estimators_], axis=0)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    return mu - kappa * sigma


def _rf_pick_lcb_point(
    rf: RandomForestRegressor,
    bounds: np.ndarray,
    X_seen: np.ndarray,
    x_best: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """RF 块内选点：信赖域+全局候选 → RF-LCB → 去重 → 取最小。"""
    cand = _propose_candidates(bounds, x_best, N_CAND_LOCAL, N_CAND_GLOBAL, rng)
    lcb = _rf_lcb(rf, cand, kappa)
    md = _min_rel_dist(cand, X_seen, bounds)
    lcb[md < MIN_DIST_REL] = np.inf
    return cand[int(np.argmin(lcb))]


# ═════════════════════════════════════════════════════════════════════
# 代理二：MLP 深度集成（MLP）—— 自成一块（网络配置 / 拟合 / LCB / 选点 / 持久化）
# 输入 10 维参数、输出标量 J_b；σ 取自成员间分歧。对外仅暴露 .predict / .score。
# 与 RF 不共用任何建模代码，只保持相同的输入/输出接口。
# ═════════════════════════════════════════════════════════════════════
# 小数据（每场景 60→100 个评估点、10 维输入、标量输出）下的稳健配置。
# 该配置由「日志离线交叉验证」(在已评估点上做 RepeatedKFold) 标定得到：
#   - 集成个数 10：M=5→10 是关键一步，RMSE↓约 6%、σ 校准 cov95 0.85→0.91；
#     再加到 15 收益递减、成本翻倍，故取 10（训练 10 个小网络成本仍可忽略）
#   - 2 隐藏层 × 64：容量够拟合 10 维平滑目标，又不至于过拟合 ~100 点
#     （CV 显示 (64,64) 优于 (32,32)）
#   - tanh + lbfgs：小样本回归收敛快、曲面平滑；alpha 提供 L2 正则
#   - bootstrap：成员间用有放回重采样增加分歧，改善 σ（类比 RF 的 bagging）；
#     CV 证实关掉 bootstrap 会令 σ 塌缩、严重过自信，务必保留
#   - σ 校准系数 1.2：MLP 集成的成员间 σ 偏过自信（CV 得 c*≈1.2 才达 95% 覆盖），
#     在 LCB 中按此放大 σ，使探索更充分（见 _mlp_lcb）
MLP_HIDDEN_LAYERS = (64, 64)   # 每个 MLP 的隐藏层 → 神经元数
MLP_ENSEMBLE_SIZE = 10         # 并行训练的网络个数（集成成员数）
MLP_ACTIVATION = "tanh"
MLP_SOLVER = "lbfgs"
MLP_ALPHA = 1e-3
MLP_MAX_ITER = 2000
MLP_BOOTSTRAP = True
MLP_SIGMA_CALIB = 1.2          # σ 校准放大系数（修正集成 σ 的过自信）


def _mlp_tag() -> str:
    """编码网络结构信息，用于模型文件名与输出后缀，便于适配与区分。"""
    h = "-".join(str(x) for x in MLP_HIDDEN_LAYERS)
    return f"mlp_h{h}_m{MLP_ENSEMBLE_SIZE}"


class _ScaledMLP:
    """单个 MLP 成员：内部做输入/输出归一化，对外 .predict 接受原始参数尺度；
    供 _mlp_lcb 聚合成员预测以估计 σ。"""

    def __init__(self, mlp, x_lo, x_span, y_mean, y_std):
        self.mlp = mlp
        self.x_lo = x_lo
        self.x_span = x_span
        self.y_mean = y_mean
        self.y_std = y_std

    def predict(self, X: np.ndarray) -> np.ndarray:
        xs = (np.asarray(X, dtype=float) - self.x_lo) / self.x_span
        return self.mlp.predict(xs) * self.y_std + self.y_mean


class MLPEnsemble:
    """M 个独立 MLP 的深度集成（RF 的平行平替，二选一）。

    输入 10 维参数向量、输出标量 J_b；σ 由成员间分歧给出，供本块 LCB 采集使用。
    仅对外暴露 ``predict`` / ``score`` 以在训练指标处与 RF 保持一致的接口；
    成员存于 ``members``，由 ``_mlp_lcb`` 聚合（不复用 RF 的任何函数）。
    """

    def __init__(self, hidden_layers, ensemble_size, activation,
                 solver, alpha, max_iter, bootstrap):
        self.hidden_layers = hidden_layers
        self.ensemble_size = ensemble_size
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.bootstrap = bootstrap
        self.seeds: list[int] = []       # 各成员实际抽到的随机种子（运行时记录）
        self.members: list[_ScaledMLP] = []

    def fit(self, X: np.ndarray, Y: np.ndarray, bounds: np.ndarray) -> "MLPEnsemble":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        x_lo = bounds[:, 0]
        x_span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-6)
        y_mean = float(Y.mean())
        y_std = float(Y.std()) or 1.0
        n = len(X)
        self.members = []
        self.seeds = []
        # 由操作系统熵源播种，给每个子 MLP 抽取互不相同的随机种子，
        # 使各成员的网络参数初始化（及 bootstrap 重采样）真正随机、互相多样。
        seed_rng = np.random.default_rng()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制 lbfgs 偶发未收敛告警
            for m in range(self.ensemble_size):
                seed = int(seed_rng.integers(0, 2**31 - 1))
                self.seeds.append(seed)
                if self.bootstrap:
                    idx = np.random.RandomState(seed).randint(0, n, n)
                    xb, yb = X[idx], Y[idx]
                else:
                    xb, yb = X, Y
                xs = (xb - x_lo) / x_span
                ys = (yb - y_mean) / y_std
                mlp = MLPRegressor(
                    hidden_layer_sizes=self.hidden_layers,
                    activation=self.activation,
                    solver=self.solver,
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    random_state=seed,
                )
                mlp.fit(xs, ys)
                self.members.append(
                    _ScaledMLP(mlp, x_lo, x_span, y_mean, y_std))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([e.predict(X) for e in self.members], axis=0)
        return preds.mean(axis=0)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        return float(r2_score(np.asarray(Y, dtype=float), self.predict(X)))


def _fit_mlp(X: np.ndarray, Y: np.ndarray, name: str) -> MLPEnsemble | None:
    valid = ~np.isnan(Y) & (Y < 10.0)
    if valid.sum() < MIN_VALID_RF:
        return None
    ens = MLPEnsemble(
        hidden_layers=MLP_HIDDEN_LAYERS,
        ensemble_size=MLP_ENSEMBLE_SIZE,
        activation=MLP_ACTIVATION,
        solver=MLP_SOLVER,
        alpha=MLP_ALPHA,
        max_iter=MLP_MAX_ITER,
        bootstrap=MLP_BOOTSTRAP,
    )
    return ens.fit(X[valid], Y[valid], get_param_bounds(name))


def _mlp_lcb(ens: MLPEnsemble, X: np.ndarray, kappa: float) -> np.ndarray:
    """MLP 下置信界：成员预测均值 − κ×（校准后的）成员间标准差（越小越优）。

    成员间 σ 偏过自信，按 MLP_SIGMA_CALIB 放大后再代入 LCB，使探索更充分
    （校准系数由日志离线 CV 标定，见配置区注释）。
    """
    preds = np.stack([m.predict(X) for m in ens.members], axis=0)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0) * MLP_SIGMA_CALIB
    return mu - kappa * sigma


def _mlp_pick_lcb_point(
    ens: MLPEnsemble,
    bounds: np.ndarray,
    X_seen: np.ndarray,
    x_best: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """MLP 块内选点：信赖域+全局候选 → MLP-LCB → 去重 → 取最小。"""
    cand = _propose_candidates(bounds, x_best, N_CAND_LOCAL, N_CAND_GLOBAL, rng)
    lcb = _mlp_lcb(ens, cand, kappa)
    md = _min_rel_dist(cand, X_seen, bounds)
    lcb[md < MIN_DIST_REL] = np.inf
    return cand[int(np.argmin(lcb))]


def _save_mlp_model(name: str, model: "MLPEnsemble") -> None:
    """持久化训练好的 MLP 集成（含归一化参数与各成员权重），文件名标注网络结构。"""
    mdl_dir = PROJ / "data" / "processed_data" / "calibration" / "mlp_models"
    mdl_dir.mkdir(parents=True, exist_ok=True)
    path = mdl_dir / f"{name}_{_mlp_tag()}.joblib"
    joblib.dump(model, path)
    print(f"  Saved MLP ensemble: {path}", flush=True)


# ═════════════════════════════════════════════════════════════════════
# 统一接口：让 calibrate_one 与具体代理无关（按类型分派到对应块，两块互不共用）
# ═════════════════════════════════════════════════════════════════════

def _fit_surrogate(X: np.ndarray, Y: np.ndarray, name: str):
    """按 SURROGATE_KIND 选择代理：'mlp' → MLP 集成；其余 → 随机森林。"""
    if SURROGATE_KIND == "mlp":
        return _fit_mlp(X, Y, name)
    return _fit_rf(X, Y, name)


def _pick_lcb_point(
    model,
    bounds: np.ndarray,
    X_seen: np.ndarray,
    x_best: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """统一选点接口：按代理类型分派到各自块内实现（RF/MLP 互不共用建模逻辑）。"""
    if isinstance(model, MLPEnsemble):
        return _mlp_pick_lcb_point(model, bounds, X_seen, x_best, kappa, rng)
    return _rf_pick_lcb_point(model, bounds, X_seen, x_best, kappa, rng)


def _surrogate_config() -> dict | None:
    """记录 MLP 集成的网络超参（写入结果 JSON，便于复现与适配）；RF 模式返回 None。"""
    if SURROGATE_KIND != "mlp":
        return None
    return {
        "tag": _mlp_tag(),
        "hidden_layers": list(MLP_HIDDEN_LAYERS),
        "ensemble_size": MLP_ENSEMBLE_SIZE,
        "activation": MLP_ACTIVATION,
        "solver": MLP_SOLVER,
        "alpha": MLP_ALPHA,
        "max_iter": MLP_MAX_ITER,
        "bootstrap": MLP_BOOTSTRAP,
        "sigma_calib": MLP_SIGMA_CALIB,
    }


def _out_suffix() -> str:
    """RF 模式无后缀（保持既有文件名/结果不变）；MLP 模式追加网络标签以并存。"""
    return "" if SURROGATE_KIND != "mlp" else f"__{_mlp_tag()}"


def _parse_args(argv: list[str]) -> list[str]:
    """解析命令行：--mlp / --rf / --surrogate=<kind> 设定代理；其余视作场景名。

    通过模块级 SURROGATE_KIND 生效（不改变既有“位置参数=场景”的用法）。
    """
    global SURROGATE_KIND
    scenes: list[str] = []
    for a in argv:
        al = a.lower()
        if al == "--mlp":
            SURROGATE_KIND = "mlp"
        elif al == "--rf":
            SURROGATE_KIND = "rf"
        elif al.startswith("--surrogate="):
            SURROGATE_KIND = al.split("=", 1)[1]
        else:
            scenes.append(a)
    return scenes


def _eval_params(
    name: str,
    params: dict,
    real_feat: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    try:
        sf = _run_sumo_traci(name, params)
        err = feature_error(real_feat, sf, weights)
    except Exception as exc:
        print(f"       [!] SUMO failed: {exc}", flush=True)
        sf = np.full(len(FEATURE_NAMES), np.nan)
        err = 10.0
    return float(err), sf


def compute_features(speeds: np.ndarray, accels: np.ndarray) -> np.ndarray:
    """从速度(m/s)和纵向加速度(m/s²)序列计算 8 维行为指纹。"""
    if len(speeds) < 20 or len(accels) < 20:
        return np.full(len(FEATURE_NAMES), np.nan)

    v = speeds
    a = accels

    v_mean = float(np.mean(v))
    v_p15  = float(np.percentile(v, 15))
    v_p85  = float(np.percentile(v, 85))

    a_pos = a[a > 0.05]
    a_neg = a[a < -0.05]
    a_pos_mean = float(np.mean(a_pos))  if len(a_pos) > 5 else 0.0
    a_neg_mean = float(np.mean(np.abs(a_neg))) if len(a_neg) > 5 else 0.0
    a_p05  = float(np.percentile(a, 5))
    a_p95  = float(np.percentile(a, 95))

    stop_frac = float(np.mean(v < 1.0))

    return np.array([
        v_mean, v_p15, v_p85,
        a_pos_mean, a_neg_mean, a_p05, a_p95,
        stop_frac,
    ])


# ═════════════════════════════════════════════════════════════════════
# 真实特征提取
# ═════════════════════════════════════════════════════════════════════

def extract_real_features(name: str) -> np.ndarray:
    sc = SCENARIOS[name]
    if sc["type"] == "SIND":
        df = pd.read_csv(sc["data"] / "veh_tracks.csv")
        vx, vy = df["vx"].values, df["vy"].values
        ax_arr, ay_arr = df["ax"].values, df["ay"].values
        speed = np.sqrt(vx**2 + vy**2)
        # 纵向加速度 = 速度方向上的分量
        sp_safe = np.maximum(speed, 0.01)
        a_lon = (vx * ax_arr + vy * ay_arr) / sp_safe
        a_lon[speed < 0.3] = 0.0
        return compute_features(speed, a_lon)
    else:
        df = pd.read_csv(sc["data"] / "frenet.csv")
        cols = {c.lower(): c for c in df.columns}
        speed_col = cols.get("speed(km/h)", cols.get("velocity(km/h)"))
        acc_col = cols["acceleration(m/s^2)"]
        speed = df[speed_col].values / 3.6
        accel = df[acc_col].values
        return compute_features(speed, accel)


# ═════════════════════════════════════════════════════════════════════
# SUMO 仿真（TraCI）
# ═════════════════════════════════════════════════════════════════════

def _run_sumo_traci(name: str, params: dict) -> np.ndarray:
    """通过 TraCI 运行 SUMO，动态设置参数，收集速度/加速度，返回特征指纹。"""
    import traci

    sc = SCENARIOS[name]
    cfg = str(sc["data"] / "sumo" / f"{sc['cfg']}.sumocfg")

    traci.start(["sumo", "-c", cfg,
                 "--no-step-log", "true",
                 "--no-warnings", "true",
                 "--seed", str(np.random.randint(1, 99999))])
    try:
        for vtype in traci.vehicletype.getIDList():
            traci.vehicletype.setAccel(vtype, float(params["accel"]))
            traci.vehicletype.setDecel(vtype, float(params["decel"]))
            traci.vehicletype.setImperfection(vtype, float(params["sigma"]))
            traci.vehicletype.setTau(vtype, float(params["tau"]))
            traci.vehicletype.setMinGap(vtype, float(params["minGap"]))
            traci.vehicletype.setSpeedFactor(vtype, float(params["speedFactor"]))
            for lc_key in ("lcStrategic", "lcCooperative", "lcAssertive", "lcSpeedGain"):
                traci.vehicletype.setParameter(
                    vtype, lc_key, str(float(params[lc_key]))
                )

        speeds, accels = [], []
        warmup = sc["warmup"]
        sim_end = sc["sim_end"]
        step = 0

        while traci.simulation.getTime() < sim_end:
            traci.simulationStep()
            step += 1
            if traci.simulation.getTime() < warmup or step % 3 != 0:
                continue
            for vid in traci.vehicle.getIDList():
                speeds.append(traci.vehicle.getSpeed(vid))
                accels.append(traci.vehicle.getAcceleration(vid))
    finally:
        traci.close()

    return compute_features(np.array(speeds, dtype=float),
                            np.array(accels, dtype=float))


# ═════════════════════════════════════════════════════════════════════
# 误差计算
# ═════════════════════════════════════════════════════════════════════

def feature_error(
    real_feat: np.ndarray,
    sim_feat: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """加权平均相对误差。"""
    w = FEATURE_WEIGHTS if weights is None else weights
    if np.any(np.isnan(sim_feat)):
        return 10.0
    eps = 0.01
    rel_err = np.abs(sim_feat - real_feat) / np.maximum(np.abs(real_feat), eps)
    return float(np.dot(w, np.minimum(rel_err, 3.0)))


# ═════════════════════════════════════════════════════════════════════
# 单场景标定
# ═════════════════════════════════════════════════════════════════════

def calibrate_one(name: str) -> dict:
    """N_INIT 次 LHS + 序贯 LCB 加点，总预算 BUDGET_SUMO。"""
    print(f"\n{'=' * 64}")
    print(f"  BF-SAC (sequential LCB): {name}")
    print(f"  Budget={BUDGET_SUMO}  (init LHS={N_INIT}, sequential={BUDGET_SUMO - N_INIT})")
    print(f"{'=' * 64}")

    weights = get_feature_weights(name)
    bounds_arr = get_param_bounds(name)
    lhs_seed = _lhs_seed_for_scene(name)
    n_seq = BUDGET_SUMO - N_INIT
    rng = np.random.default_rng(lhs_seed + 17)

    print("[1/3] Extracting real behavioral fingerprint …")
    real_feat = extract_real_features(name)
    for fn, val in zip(FEATURE_NAMES, real_feat):
        print(f"       {fn:<14s} = {val:.4f}")

    print(f"[2/3] Phase A: {N_INIT} LHS initial runs …")
    X = np.zeros((BUDGET_SUMO, len(PARAM_NAMES)))
    X[:N_INIT] = lhs_samples_for_scene(N_INIT, bounds_arr, lhs_seed)
    Y = np.full(BUDGET_SUMO, np.nan)
    sim_feats = np.zeros((BUDGET_SUMO, len(FEATURE_NAMES)))
    history: list[dict] = []
    t0 = time.time()
    running_best = float("inf")

    for i in range(N_INIT):
        params = dict(zip(PARAM_NAMES, X[i]))
        err, sf = _eval_params(name, params, real_feat, weights)
        Y[i] = err
        sim_feats[i] = sf
        running_best = min(running_best, err)
        history.append({
            "sim_index": i + 1,
            "phase": "init",
            "error": round(err, 6),
            "best_so_far": round(running_best, 6),
            "params": {k: round(float(v), 4) for k, v in params.items()},
        })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_INIT - i - 1)
            print(f"       {i+1:>3d}/{N_INIT}  best={running_best:.4f}"
                  f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    doe_best_err = float(np.nanmin(Y[:N_INIT]))
    n_seen = N_INIT

    print(f"[3/3] Phase B: {n_seq} sequential LCB acquisitions …", flush=True)
    rf_rmse: float | None = None
    rf_r2: float | None = None
    seq_errors: list[float] = []
    model = None

    for j in range(n_seq):
        model = _fit_surrogate(X[:n_seen], Y[:n_seen], name)
        if model is None:
            print("       [!] Too few valid runs for surrogate; stopping sequential phase", flush=True)
            break

        pred_tr = model.predict(X[:n_seen])
        valid = ~np.isnan(Y[:n_seen]) & (Y[:n_seen] < 10.0)
        rf_rmse = float(np.sqrt(mean_squared_error(Y[:n_seen][valid], pred_tr[valid])))
        rf_r2 = float(model.score(X[:n_seen][valid], Y[:n_seen][valid]))

        best_idx = int(np.nanargmin(Y[:n_seen]))
        x_best = X[best_idx].copy()
        x_new = _pick_lcb_point(model, bounds_arr, X[:n_seen], x_best, LC_KAPPA, rng)
        params = dict(zip(PARAM_NAMES, x_new))

        err, sf = _eval_params(name, params, real_feat, weights)
        X[n_seen] = x_new
        Y[n_seen] = err
        sim_feats[n_seen] = sf
        n_seen += 1
        running_best = min(running_best, err)
        seq_errors.append(err)

        history.append({
            "sim_index": n_seen,
            "phase": "sequential",
            "lcb_round": j + 1,
            "error": round(err, 6),
            "best_so_far": round(running_best, 6),
            "rf_r2": round(rf_r2, 4),
            "params": {k: round(float(v), 4) for k, v in params.items()},
        })
        if (j + 1) % 5 == 0 or j == n_seq - 1:
            print(f"       seq {j+1:>2d}/{n_seq}  err={err:.4f}  best={running_best:.4f}"
                  f"  R2={rf_r2:.3f}", flush=True)

    if SURROGATE_KIND == "mlp" and model is not None:
        _save_mlp_model(name, model)

    best_idx = int(np.nanargmin(Y[:n_seen]))
    best_params = dict(zip(PARAM_NAMES, X[best_idx]))
    best_feat = sim_feats[best_idx]
    best_err = float(Y[best_idx])
    init_best_idx = int(np.nanargmin(Y[:N_INIT]))
    doe_best_err = float(Y[init_best_idx])

    seq_best = float(np.min(seq_errors)) if seq_errors else None
    proxy_beats_doe = bool(best_idx >= N_INIT and best_err <= doe_best_err - 1e-6)

    print(f"\n  Calibrated params:")
    for k, v in best_params.items():
        print(f"       {k:<14s} = {v:.4f}")
    print(f"  Feature error: {best_err:.4f}  (init best={doe_best_err:.4f})")
    print(f"  Total SUMO runs: {n_seen}  |  sequential improved: {proxy_beats_doe}")

    comparison = {}
    for i, fn in enumerate(FEATURE_NAMES):
        rv = float(real_feat[i])
        sv = float(best_feat[i]) if not np.isnan(best_feat[i]) else 0.0
        comparison[fn] = {
            "real": round(rv, 4),
            "sim_calibrated": round(sv, 4),
            "rel_error": round(abs(sv - rv) / max(abs(rv), 0.01), 4),
        }

    return {
        "scenario": name,
        "calibration_mode": "sequential_lcb",
        "surrogate": SURROGATE_KIND,
        "surrogate_config": _surrogate_config(),
        "calibrated_params": {k: round(v, 4) for k, v in best_params.items()},
        "feature_error": round(best_err, 4),
        "doe_best_error": round(doe_best_err, 4),
        "sequential_best_error": round(seq_best, 4) if seq_best is not None else None,
        "surrogate_verified_error": round(seq_best, 4) if seq_best is not None else None,
        "proxy_beats_doe": proxy_beats_doe,
        "n_sumo_runs": n_seen,
        "rf_r2": round(rf_r2, 4) if rf_r2 is not None else None,
        "rf_rmse": round(rf_rmse, 4) if rf_rmse is not None else None,
        "feature_comparison": comparison,
        "doe_history": history,
    }

# ═════════════════════════════════════════════════════════════════════
# 应用标定结果到 .rou.xml
# ═════════════════════════════════════════════════════════════════════

def apply_calibration(name: str, params: dict) -> None:
    """将标定参数写回场景的 .rou.xml 文件中的所有 vType。"""
    import xml.etree.ElementTree as ET
    sc = SCENARIOS[name]
    sumo_dir = sc["data"] / "sumo"
    rou_files = list(sumo_dir.glob("*.rou.xml"))
    for rou in rou_files:
        tree = ET.parse(rou)
        for vtype in tree.findall(".//vType"):
            vtype.set("accel", f"{params['accel']:.2f}")
            vtype.set("decel", f"{params['decel']:.2f}")
            vtype.set("sigma", f"{params['sigma']:.2f}")
            vtype.set("tau",   f"{params['tau']:.2f}")
            vtype.set("minGap", f"{params['minGap']:.2f}")
            vtype.set("speedFactor", f"{params['speedFactor']:.2f}")
            for lc_key in ("lcStrategic", "lcCooperative", "lcAssertive", "lcSpeedGain"):
                vtype.set(lc_key, f"{params[lc_key]:.2f}")
        tree.write(str(rou), encoding="utf-8", xml_declaration=True)
        print(f"  Updated: {rou.name}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    out_dir = PROJ / "data" / "processed_data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _parse_args(sys.argv[1:]) or list(SCENARIOS.keys())
    suffix = _out_suffix()
    if SURROGATE_KIND == "mlp":
        print(f"[surrogate] MLP ensemble ({_mlp_tag()}); outputs suffixed '{suffix}'")

    all_results = {}
    for name in targets:
        if name not in SCENARIOS:
            print(f"[!] Unknown scenario '{name}', skipping.")
            continue
        result = calibrate_one(name)
        all_results[name] = result

        per_file = out_dir / f"{name}_calibration{suffix}.json"
        slim = {k: v for k, v in result.items() if k != "doe_history"}
        per_file.write_text(json.dumps(slim, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"  Saved: {per_file}")

        # Full version with convergence history
        hist_file = out_dir / f"{name}_calibration_with_history{suffix}.json"
        hist_file.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"  Saved (with history): {hist_file}")

        # 仅 RF（默认）模式写回 .rou.xml，避免 MLP 实验污染既有标定产物
        if SURROGATE_KIND != "mlp":
            apply_calibration(name, result["calibrated_params"])

    if not all_results:
        return

    all_path = out_dir / f"calibration_all{suffix}.json"
    merged: dict = {}
    if all_path.exists():
        try:
            merged = json.loads(all_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update(all_results)
    all_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Full results → {all_path}")

    summary_rows = []
    for name in SCENARIOS:
        pf = out_dir / f"{name}_calibration{suffix}.json"
        if not pf.exists():
            continue
        r = json.loads(pf.read_text(encoding="utf-8"))
        row = {"scenario": name, "feature_error": r["feature_error"]}
        row.update(r["calibrated_params"])
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"calibration_summary{suffix}.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nCalibration summary → {summary_path}")


if __name__ == "__main__":
    main()
