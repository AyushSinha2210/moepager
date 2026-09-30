#!/usr/bin/env bash
# Run a command inside a transient cgroup v2 scope with a hard memory wall,
# without root (systemd user delegation of the memory controller).
# The cgroup's memory.stat / memory.pressure / memory.peak are saved to OUT
# just before the scope exits.
#
# Usage: bench/cgroup_run.sh MEMORY_MAX MEMORY_SWAP_MAX OUT_DIR -- CMD [ARGS...]
#   e.g. bench/cgroup_run.sh 6G 0 out/run1 -- llama-cli -m model.gguf -nr ...
set -euo pipefail
[ $# -ge 5 ] && [ "$4" = "--" ] || { echo "usage: $0 MEM_MAX SWAP_MAX OUT -- CMD..." >&2; exit 2; }
mem="$1"; swap="$2"; out="$3"; shift 4
mkdir -p "$out"
unit="moepager-bench-$$-$RANDOM"
inner='
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
echo "$cg" > "$OUT/cgroup_path"
cat "$cg/memory.max" > "$OUT/memory.max"
set +e
"$@"
rc=$?
cat "$cg/memory.stat" > "$OUT/memory.stat"
cat "$cg/memory.pressure" > "$OUT/memory.pressure" 2>/dev/null
cat "$cg/memory.peak" > "$OUT/memory.peak" 2>/dev/null
cat "$cg/memory.events" > "$OUT/memory.events" 2>/dev/null
exit $rc
'
OUT="$out" exec systemd-run --user --scope --quiet --unit="$unit" \
  -p MemoryMax="$mem" -p MemorySwapMax="$swap" -E OUT="$out" \
  bash -c "$inner" bash "$@"
