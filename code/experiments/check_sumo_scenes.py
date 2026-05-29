"""检查六场景 SUMO 配置与单次仿真是否可跑（诊断用）。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "code" / "calibration"))

from unified_calibration import (  # noqa: E402
    PARAM_NAMES,
    SCENARIOS,
    extract_real_features,
    get_param_bounds,
    _eval_params,
)


def main() -> int:
    issues: list[str] = []
    print("=" * 60)
    print("SUMO 场景前置检查")
    print("=" * 60)

    for name, sc in SCENARIOS.items():
        data = sc["data"]
        cfg = data / "sumo" / f"{sc['cfg']}.sumocfg"
        sumo_dir = data / "sumo"
        rou = list(sumo_dir.glob("*.rou.xml")) if sumo_dir.exists() else []
        print(f"\n[{name}]")
        print(f"  data dir:     {'OK' if data.exists() else 'MISSING'} {data}")
        print(f"  sumocfg:      {'OK' if cfg.exists() else 'MISSING'} {cfg.name}")
        print(f"  rou.xml:      {len(rou)} file(s)")

        if not data.exists():
            issues.append(f"{name}: 缺少数据目录")
            continue
        if not cfg.exists():
            issues.append(f"{name}: 缺少 {cfg}")
            continue
        if not rou:
            issues.append(f"{name}: 缺少 rou.xml")

        try:
            real = extract_real_features(name)
            if not (real.size == 8 and np_all_finite(real)):
                issues.append(f"{name}: 真实指纹无效")
                print(f"  real features: INVALID {real}")
            else:
                print(f"  real features: OK  J_b-ref dims=8")
        except Exception as exc:
            issues.append(f"{name}: 真实指纹读取失败 — {exc}")
            print(f"  real features: FAIL — {exc}")
            continue

        bounds = get_param_bounds(name)
        mid = {k: float((bounds[i, 0] + bounds[i, 1]) / 2) for i, k in enumerate(PARAM_NAMES)}
        print("  trial SUMO (1 eval, mid params) …", flush=True)
        try:
            err, sim = _eval_params(name, mid, real, __import__("unified_calibration").get_feature_weights(name))
            if err >= 9.99:
                issues.append(f"{name}: SUMO 评估返回失败占位误差 10.0")
                print(f"  trial SUMO: FAIL err={err}")
            else:
                print(f"  trial SUMO: OK  err={err:.4f}")
        except Exception as exc:
            issues.append(f"{name}: SUMO 仿真异常 — {exc}")
            print(f"  trial SUMO: EXCEPTION — {exc}")

    print("\n" + "=" * 60)
    if issues:
        print("未通过项:")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("六场景均可跑通前置检查 + 单次 SUMO 评估")
    return 0


def np_all_finite(arr) -> bool:
    import numpy as np
    return bool(np.all(np.isfinite(arr)))


if __name__ == "__main__":
    raise SystemExit(main())
