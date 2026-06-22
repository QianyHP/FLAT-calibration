#!/usr/bin/env bash
# 本机(macOS)跑完整流水线：把本次所有产物重定向到仓库内 runs/<名字>/，
# 并在运行期间阻止电脑睡眠/休眠(caffeinate)。
#
# 用法:  bash scripts/run_all_to.sh [名字]      # 不填默认 local
# 结果:  <repo>/runs/<名字>/{outputs,calibration}
# 仓库自带的 outputs/、calibration/ 全程不动，结束(成功/报错/Ctrl-C)自动还原。
#
# 注意：先激活装了 SUMO + traci 的环境(venv/conda)再跑。
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

# 目标目录固定在仓库根目录下的 runs/<名字>
NAME="${1:-local}"
case "$NAME" in -*|/*|*/*) echo "只传一个简单名字(不带 /)" >&2; exit 1;; esac
DEST="$REPO/runs/$NAME"
mkdir -p "$DEST/outputs" "$DEST/calibration"

# 批量跑：不弹图窗、日志实时刷新
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

SHIM=""
OUT_LINK="$REPO/outputs";                          OUT_BAK="$REPO/outputs.__orig__"
CAL_LINK="$REPO/data/processed_data/calibration";  CAL_BAK="$REPO/data/processed_data/calibration.__orig__"

restore() {
  if [ -L "$OUT_LINK" ]; then rm -f "$OUT_LINK"; fi
  if [ -d "$OUT_BAK" ];  then mv "$OUT_BAK" "$OUT_LINK"; fi
  if [ -L "$CAL_LINK" ]; then rm -f "$CAL_LINK"; fi
  if [ -d "$CAL_BAK" ];  then mv "$CAL_BAK" "$CAL_LINK"; fi
  if [ -n "$SHIM" ]; then rm -rf "$SHIM"; fi
}

# 上次被强杀(如 kill -9)可能留下残留，先拦住、让你手动恢复
if [ -e "$OUT_BAK" ] || [ -e "$CAL_BAK" ]; then
  echo "检测到上次残留，请先恢复后重跑:" >&2
  echo "  rm -f '$OUT_LINK' && mv '$OUT_BAK' '$OUT_LINK'" >&2
  echo "  rm -f '$CAL_LINK' && mv '$CAL_BAK' '$CAL_LINK'" >&2
  exit 1
fi
trap restore EXIT INT TERM

# 需要 SUMO；不在 PATH 上就早报错(多半是忘了激活环境)
command -v sumo >/dev/null 2>&1 || { echo "✗ 没找到 sumo，请先激活装了 SUMO 的环境再跑" >&2; exit 1; }

# run_all.sh 里是裸 `python`，但 mac 常只有 python3 —— 建临时 shim 兜一下。
# (必须用 exec 包装脚本而非软链接，否则触发 macOS /usr/bin/python3 的按名分发报错)
SHIM="$(mktemp -d)"
printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$(command -v python || command -v python3)" > "$SHIM/python"
chmod +x "$SHIM/python"
export PATH="$SHIM:$PATH"

# 重定向输出到 runs/<名字>
echo "=== 输出 -> $DEST ==="
if [ -e "$OUT_LINK" ]; then mv "$OUT_LINK" "$OUT_BAK"; fi
ln -s "$DEST/outputs" "$OUT_LINK"
if [ -e "$CAL_LINK" ]; then mv "$CAL_LINK" "$CAL_BAK"; fi
ln -s "$DEST/calibration" "$CAL_LINK"

# 全程防睡眠：caffeinate 在 run_all.sh 运行期间阻止 系统/磁盘/显示器 睡眠
echo "=== 已启用 caffeinate 防睡眠，开始运行 ==="
caffeinate -dimsu bash "$REPO/scripts/run_all.sh"

echo "=== 完成。结果在: $DEST (outputs/ 与 calibration/) ==="
