#!/usr/bin/env bash
# Baseline matrix for llama.cpp under a cgroup memory wall (BENCHMARKS.md).
# UNTESTED-ON-HW: requires a llama.cpp build and a real MoE model.
#
# Env:
#   LLAMA_CLI   path to llama-cli (required)
#   MODEL       path to the GGUF model (required)
#   BUDGETS     space-separated MemoryMax values (default "6G 8G")
#   CONFIGS     subset of: B0 B1 B2 B4 B5 B6 D1 D2 D3 (default all but B4)
#   NTOK        tokens to generate (default 256)
#   THREADS     llama.cpp threads (default: physical cores)
#   PROMPT      prompt text (default: a fixed prose prompt)
#   DEV         block device for diskstats (default: device of $MODEL)
#   PIN_MB      daemon pin budget in MiB (default: 40% of the budget)
#   OUT         results dir (default bench/results/run-<timestamp>)
# B3 (MGLRU off) needs root: run with LRU_GEN_OFF=1 under sudo-capable shell.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$here/.."
: "${LLAMA_CLI:?set LLAMA_CLI}"; : "${MODEL:?set MODEL}"
BUDGETS="${BUDGETS:-6G 8G}"
CONFIGS="${CONFIGS:-B0 B1 B2 B5 B6 D1 D2 D3}"
NTOK="${NTOK:-256}"
THREADS="${THREADS:-$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l)}"
PROMPT="${PROMPT:-Write a detailed, multi-paragraph explanation of how operating systems manage virtual memory, page caches and swapping.}"
OUT="${OUT:-$root/bench/results/run-$(date +%Y%m%d-%H%M%S)}"
MOEPAGER="${MOEPAGER:-$root/target/release/moepager}"
MOEPAGERD="${MOEPAGERD:-$root/target/release/moepagerd}"
if [ -z "${DEV:-}" ]; then
  src=$(df --output=source "$MODEL" | tail -1)
  DEV=$(lsblk -no PKNAME "$src" 2>/dev/null | head -1); DEV="${DEV:-$(basename "$src")}"
fi
mkdir -p "$OUT"
"$MOEPAGER" gguf-map "$MODEL" -o "$OUT/map.json" > "$OUT/map.txt" 2>&1 || true
uname -a > "$OUT/uname"; lscpu > "$OUT/lscpu"; cat /sys/kernel/mm/lru_gen/enabled > "$OUT/lru_gen" 2>/dev/null || true

to_mb() { numfmt --from=iec "$1" | awk '{printf "%d", $1/1048576}'; }

for budget in $BUDGETS; do
  pin_mb="${PIN_MB:-$(( $(to_mb "$budget") * 40 / 100 ))}"
  for cfg in $CONFIGS; do
    run="$OUT/$budget/$cfg"; mkdir -p "$run"
    llama=("$LLAMA_CLI" -m "$MODEL" -p "$PROMPT" -n "$NTOK" -t "$THREADS" --temp 0 --seed 1 -no-cnv)
    swap=0; daemon=()
    case "$cfg" in
      B0) swap=max ;;                                 # default llama.cpp (mmap + repack)
      B1) llama+=(--no-mmap); swap=max ;;             # anonymous memory
      B2) llama+=(-nr) ;;                             # page cache, kernel policy
      B4) llama+=(-nr --mlock) ;;                     # only if the model fits
      B5) llama+=(-nr); daemon=(--policy willneed-all --no-completion) ;;
      B6) llama+=(-nr); daemon=(--policy lru-pin --budget-mb "$pin_mb" --no-completion) ;;
      D1) llama+=(-nr); daemon=(--policy v --budget-mb "$pin_mb" --no-completion) ;;
      D2) llama+=(-nr); daemon=(--policy v --budget-mb "$pin_mb") ;;
      D3) llama+=(-nr); daemon=(--policy v --budget-mb "$pin_mb" --predict) ;;
      *) echo "unknown config $cfg" >&2; exit 2 ;;
    esac
    "$MOEPAGER" fault-io --drop "$MODEL" > /dev/null
    sleep "${SETTLE_S:-30}"
    "$here/snapshot.sh" "$run/before" "$DEV"
    if [ ${#daemon[@]} -gt 0 ]; then
      # The daemon must share the cgroup so its page-cache charges count.
      inner=("$MOEPAGERD" "$MODEL" --ops linux --source sentinel --stats-json "$run/daemon.json" "${daemon[@]}")
      "$here/cgroup_run.sh" "$budget" "$swap" "$run" -- bash -c '
        "${@:2:$1}" 2> "'"$run"'/daemon.log" & d=$!
        sleep 1
        "${@:$(( $1 + 2 ))}"; rc=$?
        kill -INT $d 2>/dev/null; wait $d 2>/dev/null
        exit $rc' _ "${#inner[@]}" "${inner[@]}" "${llama[@]}" > "$run/llama.out" 2> "$run/llama.err" || echo "$cfg failed" >&2
    else
      "$here/cgroup_run.sh" "$budget" "$swap" "$run" -- "${llama[@]}" > "$run/llama.out" 2> "$run/llama.err" || echo "$cfg failed" >&2
    fi
    "$here/snapshot.sh" "$run/after" "$DEV"
    echo "$budget $cfg done"
  done
done
python3 "$root/python/moepager_tools/collect.py" "$OUT" | tee "$OUT/report.md"
