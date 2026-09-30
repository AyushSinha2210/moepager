#!/usr/bin/env bash
# End-to-end demo with no model, no root and no llama.cpp:
#   real Qwen3-30B-A3B expert map (from its header) + synthetic routing →
#   analysis → policy simulation incl. Belady oracle → daemon dry run →
#   black-box recording of a replayed trace on a real (fixture) file.
# All routing data here is SYNTHETIC: it exercises the tooling and says
# nothing about real models.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
B="$root/target/release/moepager"; D="$root/target/release/moepagerd"
out="${OUT:-$root/out/demo}"; mkdir -p "$out"
qmap="$root/data/maps/qwen3-30b-a3b-q4_k_m.map.json"
fx="$root/fixtures/generated/big_moe.gguf"
step() { printf '\n== %s\n' "$*"; }

step "1. Expert map of the real Qwen3-30B-A3B Q4_K_M file (header only)"
python3 -c "import json;m=json.load(open('$qmap'));u=m['units'][0];print(f\"{m['arch']}: {m['n_layers']} layers x {m['n_experts']} experts, top-{m['top_k']}; unit L0/E0 = {u['bytes']} B in {len(u['slices'])} slices:\");[print(f\"   {s['kind']:>5} @ {s['offset']:>12} + {s['len']}\") for s in u['slices']]"

step "2. Synthetic routing shaped like Qwen3 (512 tokens)"
"$B" synth --like "$qmap" --tokens 512 --skew 0.9 --reuse 0.3 --affinity 0.3 -o "$out/qwen3-synth.mpt"

step "3. Offline analysis (reuse distance, exact LRU curve, predictability)"
"$B" analyze "$out/qwen3-synth.mpt" --map "$qmap" --json "$out/analyze.json" > "$out/analyze.txt"
head -12 "$out/analyze.txt"

step "4. Policy simulation at 15/25/35% of expert bytes (cost model: dev-laptop fault-io smoke numbers)"
"$B" sim "$out/qwen3-synth.mpt" --map "$qmap" --caps 0.15,0.25,0.35 \
  --policies lru,lfu,static-freq,v-full,v-miss,v-miss+predict,belady \
  --md "$out/sim.md" --csv "$out/sim.csv" | tail -n +2

step "5. Daemon dry run on the trace (mock OS layer)"
"$B" gguf-map "$fx" -o "$out/fixture.map.json" > /dev/null 2>&1
"$B" synth --like "$out/fixture.map.json" --tokens 30 -o "$out/fixture-synth.mpt" > /dev/null
"$D" "$fx" --source "trace:$out/fixture-synth.mpt" --ops mock --budget-mb 6 2>&1 | tail -2

step "6. Black box: replay the trace on a real file while recording page-cache misses"
"$B" fault-io --drop "$fx" > /dev/null
"$B" record "$fx" --mode sentinel --duration-s 2 --interval-us 200 -o "$out/recorded.mpt" &
rec=$!; sleep 0.3
"$B" replay "$fx" "$out/fixture-synth.mpt" --mode pread --no-readahead --drop-each-token \
  --t-layer-us 3000 --token-gap-us 3000
wait $rec
"$B" analyze "$out/recorded.mpt" --map "$out/fixture.map.json" > "$out/recorded.txt"
head -2 "$out/recorded.txt"
printf '\nDemo outputs in %s (synthetic routing; not results about real models).\n' "$out"
