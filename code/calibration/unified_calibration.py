"""unified_calibration.py  ──  FLAT 标定主程序（序贯代理 + LCB）

**FLAT** = Fingerprint-guided Learnable Acquisition for Traffic Calibration
（对外论文名）。代码与 cache 中历史缩写 **BF-SAC**（Behavioral Fingerprint
Surrogate-Assisted Calibration）与之等价，保留以免破坏已提交结果。

8 维行为指纹 + 固定 SUMO 预算下的序贯 RF/MLP 代理标定（LCB 采集）。

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

from gp_surrogate import GPSurrogate, gp_pick_lcb_point

PROJ = Path(__file__).resolve().parents[2]

# 代理模型选择："rf"（默认）、"mlp" 或 "gp"。
# 可用环境变量 BFSAC_SURROGATE 或命令行 --surrogate/--mlp/--rf/--gp 覆盖。
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


BUDGET_SUMO = 100   # 每场景总 SUMO 次数（默认 40 LHS + 60 序贯）
N_INIT = 40         # 默认阶段 A：LHS 初始设计点数
N_INIT_MAIN = 40    # 主对比实验固定 n_init
SWEEP_N_INIT = (20, 40, 60, 80, 100)  # 样本效率消融；100 = 纯 LHS（无序贯）
N_DOE = N_INIT


def lhs_samples_for_scene(n: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    sampler = LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit = sampler.random(n=n)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return unit * (hi - lo) + lo


def lhs_pool_for_scene(
    name: str,
    bounds: np.ndarray | None = None,
    *,
    run_seed: int = 42,
) -> np.ndarray:
    """场景级共用 LHS 池（BUDGET_SUMO 点）：BF-SAC 用前 N_INIT 行，n_init=100（纯 LHS）用全池。"""
    if bounds is None:
        bounds = get_param_bounds(name)
    lhs_seed = _lhs_seed_for_scene(name) + int(run_seed) * 9973
    return lhs_samples_for_scene(BUDGET_SUMO, bounds, lhs_seed)


def derive_run_seeds(name: str, run_seed: int) -> tuple[int, np.random.Generator]:
    """由场景基种子与 run_seed 派生 LHS 种子与序贯 RNG。"""
    lhs_seed = _lhs_seed_for_scene(name) + int(run_seed) * 9973
    rng = np.random.default_rng(lhs_seed + 17)
    return lhs_seed, rng


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v is not None else default


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None else default


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_hidden(key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """解析逗号分隔的隐藏层规格（如 "64,64" / "128"）；为空或非法时回退默认。"""
    v = os.environ.get(key)
    if not v:
        return default
    try:
        layers = tuple(int(x) for x in v.split(",") if x.strip())
    except ValueError:
        return default
    return layers or default


# 序贯 LCB 采集：κ 线性退火（探索→利用）；BFSAC_LC_KAPPA 设固定值可关闭退火
LC_KAPPA_START = _env_float("BFSAC_LC_KAPPA_START", 2.0)
LC_KAPPA_END = _env_float("BFSAC_LC_KAPPA_END", 0.5)
_LC_KAPPA_FIXED = os.environ.get("BFSAC_LC_KAPPA")
LC_KAPPA = float(_LC_KAPPA_FIXED) if _LC_KAPPA_FIXED is not None else LC_KAPPA_START


# κ 退火时间表形状（附录B鲁棒性消融用）："linear"（默认，现状）/ "exp"（几何插值）/
# "const"（不退火，恒为 LC_KAPPA_START）。BFSAC_LC_KAPPA 固定值优先级最高，不受此项影响。
KAPPA_SCHEDULE = os.environ.get("BFSAC_KAPPA_SCHEDULE", "linear").lower()


def _kappa_at_step(j: int, n_seq: int) -> float:
    """序贯第 j 轮（0-indexed）的 LCB κ：默认从 LC_KAPPA_START 线性降至 LC_KAPPA_END。"""
    if _LC_KAPPA_FIXED is not None:
        return float(_LC_KAPPA_FIXED)
    if KAPPA_SCHEDULE == "const":
        return LC_KAPPA_START
    if n_seq <= 1:
        return LC_KAPPA_END
    t = j / (n_seq - 1)
    if KAPPA_SCHEDULE == "exp":
        start = max(LC_KAPPA_START, 1e-6)
        end = max(LC_KAPPA_END, 1e-6)
        return float(start * (end / start) ** t)
    return LC_KAPPA_START + (LC_KAPPA_END - LC_KAPPA_START) * t


N_CAND_LOCAL = 2500
# 全局候选池大小；BFSAC_N_CAND_GLOBAL=0 即"纯信赖域采样"（附录B候选几何消融用）。
N_CAND_GLOBAL = _env_int("BFSAC_N_CAND_GLOBAL", 2500)
# 信赖域半宽（各维 ± 该比例 × 参数跨度）；可用 BFSAC_TRUST_FRAC 覆盖（附录B宽度敏感性用）。
TRUST_FRAC = _env_float("BFSAC_TRUST_FRAC", 0.18)
MIN_DIST_REL = 0.02 # 与已评估点最小相对距离（避免重复仿真）
MIN_VALID_RF = 10
MIN_VALID_FLOOR = 5
FAIL_ERROR = 10.0


def _surrogate_valid_mask(Y: np.ndarray) -> np.ndarray:
    y = np.asarray(Y, dtype=float)
    return np.isfinite(y) & (y < FAIL_ERROR - 1e-9)

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


def _fit_rf(
    X: np.ndarray, Y: np.ndarray, name: str, *, run_seed: int = 42,
) -> RandomForestRegressor | None:
    valid = _surrogate_valid_mask(Y)
    if int(valid.sum()) < MIN_VALID_FLOOR:
        return None
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=20,
        random_state=42 + int(run_seed), n_jobs=_rf_n_jobs(name),
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
# 小数据（每场景约 100 个评估点、10 维输入、标量输出）下的稳健默认配置。
# 取舍围绕一个核心：集成成员需保持足够分歧，才能给出可信的不确定度 σ 供 LCB 探索。
#   - 集成个数 10：兼顾 σ 估计的稳定性与训练成本（5 偏少、15 收益递减）。
#   - 2 隐藏层 × 64、tanh + lbfgs：容量足以拟合 10 维平滑目标，小样本下收敛快、曲面平滑，
#     alpha 提供 L2 正则。
#   - bootstrap + 子采样（subagging）：每个成员只在固定大小（= N_INIT）的随机子集上训练，
#     既抑制单网络过拟合，又让成员看到不同数据以维持分歧——这是 σ 不塌缩的关键（见 MLPEnsemble.fit）。
#   - σ 校准系数 1.2：成员间 σ 偏过自信，LCB 前按此放大以留足探索余量（见 _mlp_lcb）。
#   - warm-start 增量续训：Phase B 每轮在上一轮成员权重上继续优化；输出归一化在首训冻结，
#     以保证续训目标尺度一致（见 MLPEnsemble.update）。
# 下列默认值与 env / CLI 覆盖无关，仅供 _mlp_tag 判断是否偏离默认、用于追加文件名标记。
_MLP_TAG_DEFAULTS = {
    "activation": "tanh",
    "solver": "lbfgs",
    "alpha": 1e-3,
    "max_iter": 2000,
    "bootstrap": True,
    "subset": N_INIT,
    "warm_start": True,
    "sigma_calib": 1.2,
}

MLP_HIDDEN_LAYERS = _env_hidden("BFSAC_MLP_HIDDEN", (64, 64))  # 每个 MLP 的隐藏层 → 神经元数
MLP_ENSEMBLE_SIZE = _env_int("BFSAC_MLP_ENSEMBLE_SIZE", 10)
MLP_ACTIVATION = "tanh"
MLP_SOLVER = "lbfgs"
MLP_ALPHA = _env_float("BFSAC_MLP_ALPHA", 1e-3)
MLP_MAX_ITER = 2000
MLP_BOOTSTRAP = _env_bool("BFSAC_MLP_BOOTSTRAP", True)
MLP_SIGMA_CALIB = _env_float("BFSAC_MLP_SIGMA_CALIB", 1.2)
MLP_TRAIN_SUBSET = _env_int("BFSAC_MLP_TRAIN_SUBSET", N_INIT)
MLP_WARM_START = _env_bool("BFSAC_MLP_WARM_START", True)
MLP_SAVE_MODEL = _env_bool("BFSAC_MLP_SAVE_MODEL", True)  # 批量消融可关，避免并行写同名 joblib


def _mlp_tag() -> str:
    """编码网络结构与「非默认超参」用于模型文件名/输出后缀，确保不同消融配置互不覆盖。

    主标签 ``mlp_h<隐藏层>_m<集成数>`` 始终反映结构与平行量；其余超参仅在被改成非默认时
    追加紧凑标记（``_nob``/``_s40``/``_sfull``/``_nows``/``_a..``/``_it..``/``_sc..`` 等），
    从而保持默认配置文件名不变、向后兼容，又让参数扫描各自落盘、不相互覆盖。
    """
    h = "-".join(str(x) for x in MLP_HIDDEN_LAYERS)
    tag = f"mlp_h{h}_m{MLP_ENSEMBLE_SIZE}"
    d = _MLP_TAG_DEFAULTS
    if MLP_ACTIVATION != d["activation"]:
        tag += f"_ac{MLP_ACTIVATION}"
    if MLP_SOLVER != d["solver"]:
        tag += f"_sv{MLP_SOLVER}"
    if MLP_ALPHA != d["alpha"]:
        tag += f"_a{MLP_ALPHA:g}"
    if MLP_MAX_ITER != d["max_iter"]:
        tag += f"_it{MLP_MAX_ITER}"
    if not MLP_BOOTSTRAP:
        tag += "_nob"
    if MLP_TRAIN_SUBSET != d["subset"]:
        tag += "_sfull" if not MLP_TRAIN_SUBSET else f"_s{MLP_TRAIN_SUBSET}"
    if not MLP_WARM_START:
        tag += "_nows"
    if MLP_SIGMA_CALIB != d["sigma_calib"]:
        tag += f"_sc{MLP_SIGMA_CALIB:g}"
    return tag


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
                 solver, alpha, max_iter, bootstrap, subset_size=None,
                 warm_start=False):
        self.hidden_layers = hidden_layers
        self.ensemble_size = ensemble_size
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.bootstrap = bootstrap
        self.subset_size = subset_size   # 每个成员训练子集上限（None=用全部历史数据）
        self.warm_start = warm_start     # True → 各成员可在已有权重上续训（见 update）
        self.seeds: list[int] = []       # 各成员实际抽到的随机种子（运行时记录）
        self.members: list[_ScaledMLP] = []
        # 首训时冻结的归一化（供 warm-start 续训复用，保证同一输入/目标尺度）
        self.x_lo = None
        self.x_span = None
        self.y_mean = None
        self.y_std = None

    def fit(
        self, X: np.ndarray, Y: np.ndarray, bounds: np.ndarray,
        *, ensemble_seed: int | None = None,
    ) -> "MLPEnsemble":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        x_lo = bounds[:, 0]
        x_span = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-6)
        y_mean = float(Y.mean())
        y_std = float(Y.std()) or 1.0
        # 冻结首训归一化，供后续 warm-start 续训复用（保证同一输入/目标尺度）
        self.x_lo, self.x_span = x_lo, x_span
        self.y_mean, self.y_std = y_mean, y_std
        n = len(X)
        # 每个成员训练子集大小 k：抗过拟合子采样，固定为 subset_size（= N_INIT），
        # 但不超过当前可用样本数 n。Phase B 数据增至 100 时仍只取 60，使单网络不“吃满”全部数据。
        k = min(self.subset_size, n) if self.subset_size else n
        self.members = []
        self.seeds = []
        # 由操作系统熵源播种，给每个子 MLP 抽取互不相同的随机种子，
        # 使各成员的网络参数初始化（及子采样）真正随机、互相多样。
        seed_rng = np.random.default_rng(ensemble_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制 lbfgs 偶发未收敛告警
            for m in range(self.ensemble_size):
                seed = int(seed_rng.integers(0, 2**31 - 1))
                self.seeds.append(seed)
                # bootstrap=True → 有放回抽 k 个（保留 σ 分歧，CV 验证必需）；
                # bootstrap=False → 无放回抽 k 个（k<n 时为真子集）。k==n 且无放回时即全量。
                if self.bootstrap or k < n:
                    idx = np.random.RandomState(seed).choice(
                        n, size=k, replace=self.bootstrap)
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
                    warm_start=self.warm_start,  # True → 后续 update 可在此权重上续训
                )
                mlp.fit(xs, ys)
                self.members.append(
                    _ScaledMLP(mlp, x_lo, x_span, y_mean, y_std))
        return self

    def update(
        self, X: np.ndarray, Y: np.ndarray, bounds: np.ndarray,
        *, subset_seed: int | None = None,
    ) -> "MLPEnsemble":
        """Phase B 增量续训：在「已有成员权重」上继续训练，而非从零重建。

        - 复用首训冻结的归一化(x/y)，使续训在同一输入/目标尺度下进行；
        - 每个成员各自重抽一份子集（大小 = subset_size），从当前权重出发再用
          lbfgs 优化（warm_start=True，通常很快收敛到含新样本的新最优）；
        - 各成员起始权重不同、每轮所见子集不同，故续训中仍保持成员间分歧（σ）。
        若尚未首训（members 为空），退化为从零 fit。
        """
        if not self.members:
            return self.fit(X, Y, bounds)
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n = len(X)
        k = min(self.subset_size, n) if self.subset_size else n
        sub_rng = np.random.default_rng(subset_seed)  # 每轮重新随机抽子集
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制 lbfgs 偶发未收敛告警
            for member in self.members:
                if self.bootstrap or k < n:
                    idx = sub_rng.choice(n, size=k, replace=self.bootstrap)
                    xb, yb = X[idx], Y[idx]
                else:
                    xb, yb = X, Y
                xs = (xb - self.x_lo) / self.x_span
                ys = (yb - self.y_mean) / self.y_std
                member.mlp.fit(xs, ys)  # warm_start=True → 在已有权重上续训
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([e.predict(X) for e in self.members], axis=0)
        return preds.mean(axis=0)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        return float(r2_score(np.asarray(Y, dtype=float), self.predict(X)))


def _fit_mlp(
    X: np.ndarray, Y: np.ndarray, name: str, *, run_seed: int = 42,
) -> MLPEnsemble | None:
    valid = _surrogate_valid_mask(Y)
    if int(valid.sum()) < MIN_VALID_FLOOR:
        return None
    ens = MLPEnsemble(
        hidden_layers=MLP_HIDDEN_LAYERS,
        ensemble_size=MLP_ENSEMBLE_SIZE,
        activation=MLP_ACTIVATION,
        solver=MLP_SOLVER,
        alpha=MLP_ALPHA,
        max_iter=MLP_MAX_ITER,
        bootstrap=MLP_BOOTSTRAP,
        subset_size=MLP_TRAIN_SUBSET,
        warm_start=MLP_WARM_START,
    )
    return ens.fit(
        X[valid], Y[valid], get_param_bounds(name), ensemble_seed=1000 + int(run_seed),
    )


def _update_mlp(
    ens: MLPEnsemble, X: np.ndarray, Y: np.ndarray, name: str, *, round_idx: int, run_seed: int,
) -> MLPEnsemble:
    """Phase B 续训：在已有集成上增量更新（warm-start），保留各成员权重。"""
    valid = _surrogate_valid_mask(Y)
    if int(valid.sum()) < MIN_VALID_FLOOR:
        return ens
    return ens.update(
        X[valid], Y[valid], get_param_bounds(name),
        subset_seed=2000 + int(run_seed) * 100 + int(round_idx),
    )


def _mlp_lcb(ens: MLPEnsemble, X: np.ndarray, kappa: float) -> np.ndarray:
    """MLP 下置信界：成员预测均值 − κ×（校准后的）成员间标准差（越小越优）。

    成员间 σ 偏过自信，按 MLP_SIGMA_CALIB 放大后再代入 LCB，使探索更充分（见配置区注释）。
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
# 代理三：Gaussian Process（GP）
# ═════════════════════════════════════════════════════════════════════

def _fit_gp(
    X: np.ndarray, Y: np.ndarray, name: str, *, run_seed: int = 42,
) -> GPSurrogate | None:
    valid = _surrogate_valid_mask(Y)
    n_valid = int(valid.sum())
    if n_valid < MIN_VALID_FLOOR:
        return None
    gp = GPSurrogate(
        random_state=42 + int(run_seed),
        min_valid=min(MIN_VALID_RF, n_valid),
    )
    gp.fit(X[valid], Y[valid])
    return gp


def _gp_pick_lcb_point(
    gp: GPSurrogate,
    bounds: np.ndarray,
    X_seen: np.ndarray,
    x_best: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return gp_pick_lcb_point(
        gp, bounds, X_seen, x_best, kappa, rng,
        propose_candidates=_propose_candidates,
        min_rel_dist=_min_rel_dist,
        n_local=N_CAND_LOCAL,
        n_global=N_CAND_GLOBAL,
        min_dist_rel=MIN_DIST_REL,
    )

# ═════════════════════════════════════════════════════════════════════
# 统一接口：让 calibrate_one 与具体代理无关（按类型分派到对应块，两块互不共用）
# ═════════════════════════════════════════════════════════════════════

def _fit_surrogate(X: np.ndarray, Y: np.ndarray, name: str, *, run_seed: int = 42):
    """按 SURROGATE_KIND 选择代理：mlp / gp / rf。"""
    if SURROGATE_KIND == "mlp":
        return _fit_mlp(X, Y, name, run_seed=run_seed)
    if SURROGATE_KIND == "gp":
        return _fit_gp(X, Y, name, run_seed=run_seed)
    return _fit_rf(X, Y, name, run_seed=run_seed)


# 获取（打分）规则消融（附录B）："lcb"（默认，现状）/ "exploit"（argmin μ）/
# "explore"（argmax σ）/ "random"（候选池均匀随机）/ "explore_then_exploit"
# （序贯前半程纯探索、后半程纯利用——公平性对照，见 unified_calibration 顶部说明）。
ACQUISITION_MODE = os.environ.get("BFSAC_ACQUISITION", "lcb").lower()


def _predict_mu_sigma(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """统一 (μ, σ) 提取：跨代理复用同一套获取规则消融逻辑（仅供 ACQUISITION_MODE!=lcb 使用）。"""
    if isinstance(model, MLPEnsemble):
        preds = np.stack([m.predict(X) for m in model.members], axis=0)
        return preds.mean(axis=0), preds.std(axis=0) * MLP_SIGMA_CALIB
    if isinstance(model, GPSurrogate):
        return model.predict(X, return_std=True)
    preds = np.stack([t.predict(X) for t in model.estimators_], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


def _generic_pick_point(
    model,
    bounds: np.ndarray,
    X_seen: np.ndarray,
    x_best: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
    *,
    j: int | None,
    n_seq: int | None,
) -> np.ndarray:
    """ACQUISITION_MODE != "lcb" 时的统一选点：候选几何（信赖域+全局池）保持不变，只换打分规则。"""
    cand = _propose_candidates(bounds, x_best, N_CAND_LOCAL, N_CAND_GLOBAL, rng)
    md = _min_rel_dist(cand, X_seen, bounds)
    ok = md >= MIN_DIST_REL

    if ACQUISITION_MODE == "random":
        pool = np.where(ok)[0]
        if len(pool) == 0:
            pool = np.arange(len(cand))
        return cand[int(rng.choice(pool))]

    mu, sigma = _predict_mu_sigma(model, cand)
    if ACQUISITION_MODE == "exploit":
        score = mu.copy()
    elif ACQUISITION_MODE == "explore":
        score = -sigma
    elif ACQUISITION_MODE == "explore_then_exploit":
        half = (n_seq or 1) / 2.0
        score = -sigma if (j is not None and j < half) else mu.copy()
    else:
        score = mu - kappa * sigma
    score = np.asarray(score, dtype=float).copy()
    score[~ok] = np.inf
    if not np.any(np.isfinite(score)):
        return cand[int(rng.integers(0, len(cand)))]
    return cand[int(np.argmin(score))]


def _pick_lcb_point(
    model,
    bounds: np.ndarray,
    X_seen: np.ndarray,
    x_best: np.ndarray,
    kappa: float,
    rng: np.random.Generator,
    *,
    j: int | None = None,
    n_seq: int | None = None,
) -> np.ndarray:
    """统一选点接口：按代理类型分派到各自块内实现（RF/MLP 互不共用建模逻辑）。

    ACQUISITION_MODE != "lcb" 时改走 `_generic_pick_point`（附录B获取规则消融），
    候选几何（信赖域宽度/全局池大小）仍由模块级 TRUST_FRAC / N_CAND_GLOBAL 控制，
    与默认 LCB 路径共用，保证消融只改变"怎么打分"这一个变量。
    """
    if ACQUISITION_MODE != "lcb":
        return _generic_pick_point(model, bounds, X_seen, x_best, kappa, rng, j=j, n_seq=n_seq)
    if isinstance(model, MLPEnsemble):
        return _mlp_pick_lcb_point(model, bounds, X_seen, x_best, kappa, rng)
    if isinstance(model, GPSurrogate):
        return _gp_pick_lcb_point(model, bounds, X_seen, x_best, kappa, rng)
    return _rf_pick_lcb_point(model, bounds, X_seen, x_best, kappa, rng)


def _surrogate_config() -> dict | None:
    if SURROGATE_KIND == "gp":
        return {"kernel": "Matern(2.5)+WhiteKernel", "input_scaled": True}
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
        "train_subset": MLP_TRAIN_SUBSET,
        "warm_start": MLP_WARM_START,
    }


def _out_suffix() -> str:
    """RF 模式无后缀（保持既有文件名/结果不变）；MLP 模式追加网络标签以并存。"""
    return "" if SURROGATE_KIND != "mlp" else f"__{_mlp_tag()}"


def _parse_args(argv: list[str]) -> list[str]:
    """解析命令行；均通过模块级全局变量生效（不改变既有“位置参数=场景”的用法）。

    代理选择：--mlp / --rf / --surrogate=<kind>
    MLP 超参（仅 MLP 模式有意义，RF 不受影响；CLI 覆盖同名环境变量默认）：
      --ensemble=<M>                 集成成员数（平行量，如 5/10）       → MLP_ENSEMBLE_SIZE
      --hidden=<A,B,...>             隐藏层神经元（如 64,64 或 128）       → MLP_HIDDEN_LAYERS
      --activation=<name>            激活函数（tanh/relu/...）             → MLP_ACTIVATION
      --solver=<name>                求解器（lbfgs/adam/sgd）              → MLP_SOLVER
      --alpha=<x>                    L2 正则强度（如 1e-3）                → MLP_ALPHA
      --max-iter=<N>                 单次拟合最大迭代                      → MLP_MAX_ITER
      --sigma-calib=<x>              σ 标准放大系数                        → MLP_SIGMA_CALIB
      --bootstrap / --no-bootstrap   开/关成员自助重采样（有放回）         → MLP_BOOTSTRAP
      --subset=<N|none>              成员训练子集大小；none/all/full/0=全部 → MLP_TRAIN_SUBSET
      --warm-start / --no-warm-start Phase B 是否在上一轮权重上续训        → MLP_WARM_START
    其余参数一律视作场景名。
    """
    global SURROGATE_KIND, MLP_BOOTSTRAP, MLP_TRAIN_SUBSET, MLP_WARM_START
    global MLP_ENSEMBLE_SIZE, MLP_HIDDEN_LAYERS, MLP_ACTIVATION, MLP_SOLVER
    global MLP_ALPHA, MLP_MAX_ITER, MLP_SIGMA_CALIB

    def _num(flag, raw, cast, current):
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"       [!] 无法解析 {flag}={raw}，保留默认 {current}", flush=True)
            return current

    def _hidden(raw):
        layers = tuple(int(x) for x in raw.split(",") if x.strip())
        if not layers:
            raise ValueError(raw)
        return layers

    scenes: list[str] = []
    for a in argv:
        al = a.lower()
        if al == "--mlp":
            SURROGATE_KIND = "mlp"
        elif al == "--rf":
            SURROGATE_KIND = "rf"
        elif al == "--gp":
            SURROGATE_KIND = "gp"
        elif al.startswith("--surrogate="):
            SURROGATE_KIND = al.split("=", 1)[1]
        elif al == "--bootstrap":
            MLP_BOOTSTRAP = True
        elif al == "--no-bootstrap":
            MLP_BOOTSTRAP = False
        elif al == "--warm-start":
            MLP_WARM_START = True
        elif al == "--no-warm-start":
            MLP_WARM_START = False
        elif al.startswith("--subset="):
            v = al.split("=", 1)[1]
            MLP_TRAIN_SUBSET = None if v in ("none", "all", "full", "0") \
                else _num("--subset", v, int, MLP_TRAIN_SUBSET)
        elif al.startswith("--ensemble="):
            MLP_ENSEMBLE_SIZE = _num("--ensemble", al.split("=", 1)[1], int, MLP_ENSEMBLE_SIZE)
        elif al.startswith("--hidden="):
            MLP_HIDDEN_LAYERS = _num("--hidden", a.split("=", 1)[1], _hidden, MLP_HIDDEN_LAYERS)
        elif al.startswith("--activation="):
            MLP_ACTIVATION = al.split("=", 1)[1]
        elif al.startswith("--solver="):
            MLP_SOLVER = al.split("=", 1)[1]
        elif al.startswith("--alpha="):
            MLP_ALPHA = _num("--alpha", al.split("=", 1)[1], float, MLP_ALPHA)
        elif al.startswith("--max-iter="):
            MLP_MAX_ITER = _num("--max-iter", al.split("=", 1)[1], int, MLP_MAX_ITER)
        elif al.startswith("--sigma-calib="):
            MLP_SIGMA_CALIB = _num("--sigma-calib", al.split("=", 1)[1], float, MLP_SIGMA_CALIB)
        else:
            scenes.append(a)
    return scenes


def _eval_params(
    name: str,
    params: dict,
    real_feat: np.ndarray,
    weights: np.ndarray,
    *,
    real_speed=None,
    objective_mode: str = "jb",
) -> tuple[float, float, np.ndarray]:
    """Run one SUMO evaluation.

    Returns (optimization_error, jb, sim_features).
    optimization_error is what the surrogate/LCB loop minimizes (J_b or raw RMSE);
    jb is always the fingerprint error for downstream reporting.
    """
    try:
        sim_feat, per_sec_sim = _run_sumo_sim(name, params)
        jb = feature_error(real_feat, sim_feat, weights)
        if objective_mode == "raw_rmse":
            if real_speed is None:
                real_speed = build_real_speed_profile(name)
            opt_err = per_second_speed_rmse(per_sec_sim, real_speed)
        else:
            opt_err = jb
    except Exception as exc:
        print(f"       [!] SUMO failed: {exc}", flush=True)
        sim_feat = np.full(len(FEATURE_NAMES), np.nan)
        opt_err = jb = 10.0
    return float(opt_err), float(jb), sim_feat


def build_real_speed_profile(name: str):
    """Per-second mean speed profile from observed data (SIND or UTE)."""
    sc = SCENARIOS[name]
    warmup, sim_end = sc["warmup"], sc["sim_end"]
    if sc["type"] == "SIND":
        df = pd.read_csv(sc["data"] / "veh_tracks.csv")
        t_s = df["timestamp_ms"].values / 1000.0
        speed = np.sqrt(df["vx"].values ** 2 + df["vy"].values ** 2)
        mask = (t_s >= warmup) & (t_s < sim_end)
        per_sec = pd.Series(speed[mask], name="speed").groupby(t_s[mask].astype(int)).mean()
    else:
        df = pd.read_csv(sc["data"] / "frenet.csv")
        cols = {c.lower(): c for c in df.columns}
        spd_col = cols.get("speed(km/h)", cols.get("velocity(km/h)"))
        time_col = cols.get("time(s)", cols.get("timestamp(s)", cols.get("time")))
        speed = df[spd_col].values / 3.6
        if time_col:
            t_s = df[time_col].values
            mask = (t_s >= warmup) & (t_s < sim_end)
            per_sec = pd.Series(speed[mask], name="speed").groupby(t_s[mask].astype(int)).mean()
        else:
            t_s = np.arange(len(df)) * 0.1
            mask = (t_s >= warmup) & (t_s < sim_end)
            per_sec = pd.Series(speed[mask]).groupby(t_s[mask].astype(int)).mean()
    return per_sec.sort_index()


def per_second_speed_rmse(per_sec_sim: dict, real_speed) -> float:
    """RMSE of per-second mean speed (sim vs. observed), same as Fig.~2 left panel."""
    common_secs = sorted(set(per_sec_sim.keys()) & set(real_speed.index))
    if len(common_secs) < 10:
        return 10.0
    sim_means = np.array([np.mean(per_sec_sim[s]) for s in common_secs])
    real_means = np.array([real_speed[s] for s in common_secs])
    return float(np.sqrt(np.mean((sim_means - real_means) ** 2)))


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

def _run_sumo_sim(
    name: str, params: dict, *, seed: int | None = None,
    return_speed_samples: bool = False,
) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, np.ndarray]:
    """Replay; optionally expose half-open-window speed samples and a fixed seed.

    Default calls retain the original fingerprint, sampling and random-seed policy.
    """
    import traci

    sc = SCENARIOS[name]
    cfg = str(sc["data"] / "sumo" / f"{sc['cfg']}.sumocfg")

    traci.start(["sumo", "-c", cfg,
                 "--no-step-log", "true",
                 "--no-warnings", "true",
                 "--seed", str(np.random.randint(1, 99999) if seed is None else int(seed))])
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
        distribution_speeds = []
        per_sec_sim: dict[int, list[float]] = {}
        warmup = sc["warmup"]
        sim_end = sc["sim_end"]
        step = 0

        while traci.simulation.getTime() < sim_end:
            traci.simulationStep()
            step += 1
            t = traci.simulation.getTime()
            if t < warmup:
                continue
            vids = traci.vehicle.getIDList()
            if not vids:
                continue
            speeds_now = [traci.vehicle.getSpeed(vid) for vid in vids]
            accels_now = [traci.vehicle.getAcceleration(vid) for vid in vids]
            per_sec_sim.setdefault(int(t), []).extend(speeds_now)
            if step % 3 == 0:
                speeds.extend(speeds_now)
                accels.extend(accels_now)
                if return_speed_samples and t < sim_end:
                    distribution_speeds.extend(speeds_now)
    finally:
        traci.close()

    sim_feat = compute_features(np.array(speeds, dtype=float),
                            np.array(accels, dtype=float))
    if return_speed_samples:
        return sim_feat, per_sec_sim, np.asarray(distribution_speeds, dtype=float)
    return sim_feat, per_sec_sim


def _run_sumo_traci(name: str, params: dict) -> np.ndarray:
    """Legacy wrapper: fingerprint only."""
    sim_feat, _ = _run_sumo_sim(name, params)
    return sim_feat


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

def calibrate_one(
    name: str,
    *,
    surrogate_kind: str | None = None,
    n_init: int | None = None,
    run_seed: int = 42,
    budget_sumo: int | None = None,
    objective_mode: str = "jb",
) -> dict:
    """n_init 次 LHS + (BUDGET_SUMO - n_init) 次序贯 LCB，总预算 BUDGET_SUMO.

    n_init: 初始 LHS 点数；默认 N_INIT（40）。n_init=100 时退化为纯 LHS（无序贯加点）。
    surrogate_kind: 覆盖模块级 SURROGATE_KIND（如 run_comparison 指定 MLP 版）。
    run_seed: 实验重复编号；影响 LHS 池、序贯 RNG 与代理随机种子。
    objective_mode: ``jb`` (fingerprint, default) or ``raw_rmse`` (per-second speed RMSE).
    """
    eff_n_init = N_INIT if n_init is None else int(n_init)
    max_b = budget_sumo or BUDGET_SUMO
    if eff_n_init < 1 or eff_n_init > max_b:
        raise ValueError(f"n_init 须在 [1, {max_b}]，收到 {eff_n_init}")
    if objective_mode not in ("jb", "raw_rmse"):
        raise ValueError(f"objective_mode must be 'jb' or 'raw_rmse', got {objective_mode!r}")

    global SURROGATE_KIND
    prev_kind = SURROGATE_KIND
    if surrogate_kind is not None:
        SURROGATE_KIND = surrogate_kind.lower()
    try:
        return _calibrate_one_impl(
            name, n_init=eff_n_init, run_seed=int(run_seed),
            budget_sumo=max_b, objective_mode=objective_mode,
        )
    finally:
        SURROGATE_KIND = prev_kind


def _calibrate_one_impl(
    name: str, *, n_init: int, run_seed: int = 42, budget_sumo: int = BUDGET_SUMO,
    objective_mode: str = "jb",
) -> dict:
    n_seq = budget_sumo - n_init
    print(f"\n{'=' * 64}")
    print(f"  BF-SAC (sequential LCB): {name}  surrogate={SURROGATE_KIND}"
          f"  objective={objective_mode}")
    print(f"  Budget={budget_sumo}  (init LHS={n_init}, sequential={n_seq})  run_seed={run_seed}")
    if _LC_KAPPA_FIXED is not None:
        print(f"  LCB kappa={LC_KAPPA} (fixed)", end="")
    else:
        print(f"  LCB kappa={LC_KAPPA_START}→{LC_KAPPA_END} (anneal, schedule={KAPPA_SCHEDULE})", end="")
    print(f"  acquisition={ACQUISITION_MODE} trust_frac={TRUST_FRAC} n_cand_global={N_CAND_GLOBAL}", end="")
    if SURROGATE_KIND == "mlp":
        print(
            f"  MLP m={MLP_ENSEMBLE_SIZE} subset={MLP_TRAIN_SUBSET}"
            f" sigma_calib={MLP_SIGMA_CALIB} warm_start={MLP_WARM_START}",
            end="",
        )
    print()
    print(f"{'=' * 64}")

    weights = get_feature_weights(name)
    bounds_arr = get_param_bounds(name)
    lhs_seed, rng = derive_run_seeds(name, run_seed)

    print("[1/3] Extracting real behavioral fingerprint …")
    real_feat = extract_real_features(name)
    real_speed = build_real_speed_profile(name) if objective_mode == "raw_rmse" else None
    for fn, val in zip(FEATURE_NAMES, real_feat):
        print(f"       {fn:<14s} = {val:.4f}")

    print(f"[2/3] Phase A: {n_init} LHS initial runs …")
    X = np.zeros((budget_sumo, len(PARAM_NAMES)))
    _lhs_s = _lhs_seed_for_scene(name) + int(run_seed) * 9973
    X[:n_init] = lhs_samples_for_scene(budget_sumo, bounds_arr, _lhs_s)[:n_init]
    Y = np.full(budget_sumo, np.nan)
    Y_jb = np.full(budget_sumo, np.nan)
    sim_feats = np.zeros((budget_sumo, len(FEATURE_NAMES)))
    history: list[dict] = []
    t0 = time.time()
    running_best = float("inf")
    running_best_jb = float("inf")

    for i in range(n_init):
        params = dict(zip(PARAM_NAMES, X[i]))
        opt_err, jb, sf = _eval_params(
            name, params, real_feat, weights,
            real_speed=real_speed, objective_mode=objective_mode,
        )
        Y[i] = opt_err
        Y_jb[i] = jb
        sim_feats[i] = sf
        running_best = min(running_best, opt_err)
        running_best_jb = min(running_best_jb, jb)
        history.append({
            "sim_index": i + 1,
            "phase": "init",
            "error": round(opt_err, 6),
            "jb": round(jb, 6),
            "best_so_far": round(running_best, 6),
            "best_jb_so_far": round(running_best_jb, 6),
            "params": {k: round(float(v), 4) for k, v in params.items()},
        })
        if (i + 1) % 10 == 0 or (i + 1) == n_init:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_init - i - 1) if i + 1 < n_init else 0
            print(f"       {i+1:>3d}/{n_init}  best={running_best:.4f}"
                  f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    doe_best_err = float(np.nanmin(Y[:n_init]))
    n_seen = n_init

    if n_seq > 0:
        print(f"[3/3] Phase B: {n_seq} sequential LCB acquisitions …", flush=True)
    else:
        print("[3/3] Phase B: skipped (n_init = budget, pure LHS)", flush=True)
    rf_rmse: float | None = None
    rf_r2: float | None = None
    seq_errors: list[float] = []
    model = None

    for j in range(n_seq):
        _step_t0 = time.perf_counter()
        if (SURROGATE_KIND == "mlp" and MLP_WARM_START
                and isinstance(model, MLPEnsemble)):
            # MLP 续训：在上一轮成员权重上增量更新（首轮 model 仍为 None，走下面从零训练）
            model = _update_mlp(
                model, X[:n_seen], Y[:n_seen], name, round_idx=j, run_seed=run_seed,
            )
        else:
            model = _fit_surrogate(X[:n_seen], Y[:n_seen], name, run_seed=run_seed)
        if model is None:
            print("       [!] Too few valid runs for surrogate; stopping sequential phase", flush=True)
            break

        pred_tr = model.predict(X[:n_seen])
        valid = _surrogate_valid_mask(Y[:n_seen])
        rf_rmse = float(np.sqrt(mean_squared_error(Y[:n_seen][valid], pred_tr[valid])))
        rf_r2 = float(model.score(X[:n_seen][valid], Y[:n_seen][valid]))

        best_idx = int(np.nanargmin(Y[:n_seen]))
        x_best = X[best_idx].copy()
        kappa_j = _kappa_at_step(j, n_seq)
        x_new = _pick_lcb_point(model, bounds_arr, X[:n_seen], x_best, kappa_j, rng, j=j, n_seq=n_seq)
        # 代理重拟合 + 采集选点的墙钟耗时（毫秒）：附录B"在线决策成本可忽略"论证用埋点。
        step_wall_ms = round((time.perf_counter() - _step_t0) * 1000.0, 2)
        params = dict(zip(PARAM_NAMES, x_new))

        opt_err, jb, sf = _eval_params(
            name, params, real_feat, weights,
            real_speed=real_speed, objective_mode=objective_mode,
        )
        X[n_seen] = x_new
        Y[n_seen] = opt_err
        Y_jb[n_seen] = jb
        sim_feats[n_seen] = sf
        n_seen += 1
        running_best = float(np.nanmin(Y[:n_seen]))
        running_best_jb = float(np.nanmin(Y_jb[:n_seen]))
        seq_errors.append(opt_err)

        history.append({
            "sim_index": n_seen,
            "phase": "sequential",
            "lcb_round": j + 1,
            "step_wall_ms": step_wall_ms,
            "error": round(opt_err, 6),
            "jb": round(jb, 6),
            "best_so_far": round(running_best, 6),
            "best_jb_so_far": round(running_best_jb, 6),
            "rf_r2": round(rf_r2, 4),
            "params": {k: round(float(v), 4) for k, v in params.items()},
        })
        if (j + 1) % 5 == 0 or j == n_seq - 1:
            print(f"       seq {j+1:>2d}/{n_seq}  err={opt_err:.4f}  best={running_best:.4f}"
                  f"  jb={jb:.4f}  R2={rf_r2:.3f}", flush=True)

    if SURROGATE_KIND == "mlp" and model is not None and MLP_SAVE_MODEL:
        _save_mlp_model(name, model)

    best_idx = int(np.nanargmin(Y[:n_seen]))
    best_params = dict(zip(PARAM_NAMES, X[best_idx]))
    best_feat = sim_feats[best_idx]
    best_opt = float(Y[best_idx])
    best_jb = float(Y_jb[best_idx])
    init_best_idx = int(np.nanargmin(Y[:n_init]))
    doe_best_err = float(Y[init_best_idx])

    seq_best = float(np.min(seq_errors)) if seq_errors else None
    proxy_beats_doe = bool(best_idx >= n_init and best_opt <= doe_best_err - 1e-6)

    print(f"\n  Calibrated params:")
    for k, v in best_params.items():
        print(f"       {k:<14s} = {v:.4f}")
    print(f"  Objective ({objective_mode}): {best_opt:.4f}  J_b={best_jb:.4f}"
          f"  (init best obj={doe_best_err:.4f})")
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
        "optimization_objective": objective_mode,
        "surrogate": SURROGATE_KIND,
        "surrogate_config": _surrogate_config(),
        "calibrated_params": {k: round(v, 4) for k, v in best_params.items()},
        "feature_error": round(best_jb, 4),
        "optimization_best": round(best_opt, 4),
        "doe_best_error": round(doe_best_err, 4),
        "doe_best_jb": round(float(np.nanmin(Y_jb[:n_init])), 4),
        "sequential_best_error": round(seq_best, 4) if seq_best is not None else None,
        "surrogate_verified_error": round(seq_best, 4) if seq_best is not None else None,
        "proxy_beats_doe": proxy_beats_doe,
        "n_init": n_init,
        "run_seed": int(run_seed),
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
