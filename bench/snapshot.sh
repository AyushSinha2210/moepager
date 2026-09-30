#!/usr/bin/env bash
# Snapshot system-wide memory/IO metrics into a directory.
# Usage: bench/snapshot.sh OUT_DIR [BLOCK_DEV]
set -euo pipefail
out="$1"; dev="${2:-}"
mkdir -p "$out"
cat /proc/vmstat > "$out/vmstat"
cat /proc/diskstats > "$out/diskstats"
cat /proc/meminfo > "$out/meminfo"
cat /proc/pressure/memory > "$out/pressure_memory" 2>/dev/null || true
cat /proc/pressure/io > "$out/pressure_io" 2>/dev/null || true
date +%s.%N > "$out/time"
[ -n "$dev" ] && echo "$dev" > "$out/device"
true
