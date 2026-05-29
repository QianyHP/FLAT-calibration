"""Remove nearly stationary vehicle tracks from SinD veh_tracks.csv.

Heuristic (parked / roadside static, not signal queues):
  - displacement (first->last) < disp_max_m  AND  max speed < vmax_max_m/s

Typical signal-queue vehicles still exceed vmax or move enough to stay.

Usage:
  python code/preprocessing/clean_static_vehicle_tracks.py Xian
  python code/preprocessing/clean_static_vehicle_tracks.py Xian --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
RAW = PROJ / "data" / "raw_data" / "SIND"

# 与诊断脚本一致：全程静止 / 路边停放
DISP_MAX_M = 5.0
VMAX_MAX_MPS = 0.5


def classify_tracks(df: pd.DataFrame) -> pd.DataFrame:
    speed = np.hypot(df["vx"].values, df["vy"].values)
    df = df.assign(_speed=speed)
    agg = df.groupby("track_id").agg(
        n=("frame_id", "count"),
        dur_s=("timestamp_ms", lambda s: (s.max() - s.min()) / 1000.0),
        vmax=("_speed", "max"),
        vmean=("_speed", "mean"),
        stop_frac=("_speed", lambda s: (s < 1.0).mean()),
        x0=("x", "first"),
        y0=("y", "first"),
        x1=("x", "last"),
        y1=("y", "last"),
        agent=("agent_type", "first"),
    )
    agg["disp_m"] = np.hypot(agg.x1 - agg.x0, agg.y1 - agg.y0)
    agg["is_static"] = (agg.disp_m < DISP_MAX_M) & (agg.vmax < VMAX_MAX_MPS)
    return agg


def pooled_fingerprint(df: pd.DataFrame) -> dict:
    speed = np.hypot(df["vx"].values, df["vy"].values)
    vx, vy = df["vx"].values, df["vy"].values
    ax, ay = df["ax"].values, df["ay"].values
    sp_safe = np.maximum(speed, 0.01)
    a_lon = (vx * ax + vy * ay) / sp_safe
    a_lon[speed < 0.3] = 0.0
    if len(speed) < 20:
        return {}
    return {
        "v_mean": round(float(speed.mean()), 4),
        "v_p15": round(float(np.percentile(speed, 15)), 4),
        "v_p85": round(float(np.percentile(speed, 85)), 4),
        "stop_frac": round(float((speed < 1.0).mean()), 4),
        "n_rows": int(len(df)),
        "n_tracks": int(df["track_id"].nunique()),
    }


def clean_city(
    city: str,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict:
    city_dir = RAW / city
    path = city_dir / "veh_tracks.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    agg = classify_tracks(df)
    static_ids = set(agg.index[agg["is_static"]])
    kept = df[~df["track_id"].isin(static_ids)].copy()

    report = {
        "city": city,
        "disp_max_m": DISP_MAX_M,
        "vmax_max_mps": VMAX_MAX_MPS,
        "tracks_before": int(df["track_id"].nunique()),
        "tracks_removed": len(static_ids),
        "tracks_after": int(kept["track_id"].nunique()),
        "rows_before": len(df),
        "rows_after": len(kept),
        "row_removed_pct": round(100 * (1 - len(kept) / len(df)), 2),
        "fingerprint_before": pooled_fingerprint(df),
        "fingerprint_after": pooled_fingerprint(kept),
        "removed_track_ids": sorted(int(x) for x in static_ids),
    }

    print(f"\n[{city}] static track filter")
    print(f"  tracks: {report['tracks_before']} -> {report['tracks_after']} "
          f"(removed {report['tracks_removed']})")
    print(f"  rows:   {report['rows_before']} -> {report['rows_after']} "
          f"({report['row_removed_pct']}% removed)")
    fb, fa = report["fingerprint_before"], report["fingerprint_after"]
    print(f"  fingerprint BEFORE: v_mean={fb.get('v_mean')}  stop_frac={fb.get('stop_frac')}  "
          f"v_p85={fb.get('v_p85')}")
    print(f"  fingerprint AFTER:  v_mean={fa.get('v_mean')}  stop_frac={fa.get('stop_frac')}  "
          f"v_p85={fa.get('v_p85')}")

    if dry_run:
        print("  [dry-run] no files written")
        return report

    if backup:
        bak = city_dir / "veh_tracks_before_static_filter.csv"
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"  backup: {bak.name}")

    kept.drop(columns=["_speed"], errors="ignore").to_csv(path, index=False)
    print(f"  wrote: {path}")

    meta_path = city_dir / "static_filter_report.json"
    meta_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  report: {meta_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("city", nargs="?", default="Xian")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    clean_city(args.city, dry_run=args.dry_run, backup=not args.no_backup)


if __name__ == "__main__":
    main()
