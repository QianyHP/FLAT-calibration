"""
将 data/external_data/SinD 仓库中的核心数据文件
按统一命名整理到 data/raw_data/SIND/{city}/ 下。

整理后结构:
  raw_data/SIND/
  ├── Tianjin/
  │   ├── veh_tracks.csv
  │   ├── ped_tracks.csv
  │   ├── traffic_lights.csv
  │   ├── veh_meta.csv          (如有)
  │   ├── ped_meta.csv          (如有)
  │   ├── recording_meta.csv    (如有)
  │   └── map.osm
  ├── Changchun/ ...
  └── Xian/ ...
"""
from __future__ import annotations

import shutil
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
SIND_EXTERNAL = PROJ_ROOT / "data" / "external_data" / "SinD"
SIND_DATA_LEGACY = SIND_EXTERNAL / "Data"
SIND_UPSTREAM = PROJ_ROOT / "_local_archive" / "data_upstream" / "SinD_Data"
DST_DIR = PROJ_ROOT / "data" / "raw_data" / "SIND"


def _src_root(cfg: dict, src_city: str) -> Path:
    """优先 _local_archive/data_upstream/SinD_Data/{city}；其次 SinD/Data 或 SinD 根目录。"""
    if (SIND_UPSTREAM / src_city).exists():
        return SIND_UPSTREAM
    if cfg.get("src_root") == "external":
        return SIND_EXTERNAL
    if SIND_DATA_LEGACY.exists():
        return SIND_DATA_LEGACY
    return SIND_EXTERNAL


CITY_CONFIG: dict[str, dict] = {
    "Tianjin": {
        "record": "8_2_1",
        "map": "map_relink_law_save.osm",
    },
    "Changchun": {
        "record": "changchun_pudong_507_009",
        "map": "Changchun_Pudong.osm",
    },
    "Xian": {
        "src_name": "Xi'an",
        "record": "Xi'an_412_m1",
        "map": "Xi'an_Shanglin.osm",
        "src_root": "external",
    },
}

TRACK_MAPPING = {
    "Veh_smoothed_tracks.csv": "veh_tracks.csv",
    "Ped_smoothed_tracks.csv": "ped_tracks.csv",
    "Veh_tracks_meta.csv": "veh_meta.csv",
    "Ped_tracks_meta.csv": "ped_meta.csv",
    "recording_metas.csv": "recording_meta.csv",
}


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  {src.name}  ->  {dst.name}")
        return True
    return False


def organize(cities: list[str] | None = None) -> None:
    targets = cities or list(CITY_CONFIG.keys())
    for city in targets:
        if city not in CITY_CONFIG:
            print(f"[!] Unknown city '{city}', skip.")
            continue
        cfg = CITY_CONFIG[city]
        src_city = cfg.get("src_name", city)
        record = cfg["record"]
        root = _src_root(cfg, src_city)
        record_dir = root / src_city / record
        if not record_dir.exists() and city == "Xian":
            # 兼容仅放在 external_data/SinD/Xi'an 的发布包
            record_dir = SIND_EXTERNAL / src_city / record
        map_file = root / src_city / cfg["map"]

        dst_city = DST_DIR / city
        dst_city.mkdir(parents=True, exist_ok=True)
        print(f"\n[{city}]  {record_dir}")

        for src_name, dst_name in TRACK_MAPPING.items():
            copy_if_exists(record_dir / src_name, dst_city / dst_name)

        # traffic lights: filename varies per city
        for f in record_dir.glob("Traffic*.*"):
            shutil.copy2(f, dst_city / "traffic_lights.csv")
            print(f"  {f.name}  ->  traffic_lights.csv")
            break

        copy_if_exists(map_file, dst_city / "map.osm")

    print("\n整理完成。")


if __name__ == "__main__":
    import sys
    organize(sys.argv[1:] if len(sys.argv) > 1 else None)
