# Runbook — Phase 7 go/no-go (real hardware)

Exact steps to run experiments A–D from IDEA_REVIEW §6 and decide against
kill criteria K1–K4. Everything up to this point has been built and tested
without a model. This is the first phase that needs one.

## 0. Prerequisites & Environment Verification

### Environment Checklist
Before launching benchmark runs, verify host environment conditions:
- **Cgroup v2 Delegation**: Verify user memory delegation with `systemd-run --user --scope -p MemoryMax=1G true`.
- **PSI Support**: Confirm pressure stall information is active: `cat /proc/pressure/memory`.
- **NVMe Readahead**: Check device readahead size with `cat /sys/block/<nvme-dev>/queue/read_ahead_kb` (typically 128 KiB).
- **ZRAM/Swap State**: Inspect swap configuration with `swapon --show`. For page-cache runs, disable swap inside the scope via `-p MemorySwapMax=0`.

```bash
# llama.cpp (CPU build). Record the commit hash in the results.
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build ~/llama.cpp/build -j

# Models (≈ 4 + 18.6 + 12.1 GB; check free disk first)
huggingface-cli download allenai/OLMoE-1B-7B-0924-GGUF olmoe-1b-7b-0924-q4_k_m.gguf --local-dir ~/models
huggingface-cli download unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir ~/models
huggingface-cli download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --local-dir ~/models

make build
for m in ~/models/*.gguf; do target/release/moepager gguf-map "$m"; done   # check the repack warning
```

### Loop-Mounted ext4 Image (for Btrfs hosts)
Btrfs uses large readahead windows (4 MiB) and extent compression (zstd) which alter fault dynamics. To isolate standard block-layer behavior, optionally mount a loopback ext4 image:

```bash
# Create and mount a 30 GB ext4 loopback image
truncate -s 30G /tmp/ext4_scratch.img
mkfs.ext4 /tmp/ext4_scratch.img
mkdir -p ~/mnt/ext4_models
sudo mount -o loop /tmp/ext4_scratch.img ~/mnt/ext4_models
sudo chown $USER:$USER ~/mnt/ext4_models
cp ~/models/Qwen3-30B-A3B-Q4_K_M.gguf ~/mnt/ext4_models/
```

## 1. P7.1 — ground-truth expert traces (experiment-only instrumentation)

Write `tools/moe_trace/` (C++, ~60 lines, links libllama):
- set `cb_eval` in `llama_context_params`; return `true` for tensors whose
  name starts with `ffn_moe_topk` (llama.cpp tags the selected-experts
  tensor `ffn_moe_topk-<layer>`);
- when the callback is called with `ask == false`, copy the I32 ids
  (`[n_expert_used, n_tokens]`) with `ggml_backend_tensor_get`, and append
  `token,layer,expert,t_ns` rows to a CSV;
- generate N tokens greedily for a fixed prompt.

Then add `moepager import-csv` (CSV → `.mpt`, `observation = full`,
`source = llama-evalcb`). This is the only engine instrumentation in the
project, and it never ships in the product path.

Workload: 3 domains (chat, code, prose) × 512 decode tokens, greedy and
T = 0.7.

## 2. Experiment A — simulator sweeps (K1)

```bash
B=target/release/moepager
for t in traces/*.mpt; do
  $B analyze $t --map maps/$(basename $t .mpt).map.json --json $t.analyze.json
  $B sim $t --map maps/$(basename $t .mpt).map.json --caps 0.15,0.25,0.35,0.5,0.7 \
     --csv $t.sim.csv --md $t.sim.md
done
```

K1 compares Belady\* against the **kernel-measured** miss bytes from B
when available, otherwise against simulated LRU.

## 3. Experiment B — simulator validation on the real kernel

```bash
for cap in 6G 8G; do
  target/release/moepager fault-io --drop $MODEL
  bench/cgroup_run.sh $cap 0 out/B/$cap -- \
    target/release/moepager replay $MODEL traces/qwen3-chat.mpt --mode mmap --threads 6 --json out/B/$cap/replay.json
done
```

Compare `storage_read / token` and major faults per token with
`moepager sim --policies lru` at the same expert capacity. Expert
capacity = budget − dense bytes − replayer/process overhead; read it from
`memory.peak` and `memory.stat`. If they agree within 20 %, the simulator
is validated. If not, find out why (e.g. MGLRU vs LRU, read-around
amplification) before trusting A.

## 4. Experiment C — bandwidth (K2)

```bash
make microbench MODEL=~/models/Qwen3-30B-A3B-Q4_K_M.gguf
target/release/moepager fault-io --file $MODEL --unit-bytes 13253760 --units 16   # gpt-oss-sized units
```

Run on the model file itself (real data, real extents), with 3 seeds and
both `--willneed-chunk-kb 128` and `0`.

## 5. Experiment D — end to end (K3)

```bash
make bench LLAMA_CLI=~/llama.cpp/build/bin/llama-cli MODEL=~/models/Qwen3-30B-A3B-Q4_K_M.gguf
# MGLRU-off baseline (B3) needs root:
#   echo n | sudo tee /sys/kernel/mm/lru_gen/enabled ; CONFIGS=B2 make bench ... ; echo y | sudo tee ...
# Co-tenant probe alongside each config:
#   python3 python/moepager_tools/cotenant.py --duration-s 120 --out out/cotenant-<cfg>.json
```

`bench/run_llama.sh` is untested. Expect to debug it on the first run
(daemon start-up inside the scope, llama-cli flags of the installed
version).

## 6. Decide

Fill BENCHMARKS.md tables from the raw outputs, never by hand. Then apply
K1–K4 from IDEA_REVIEW §6 and add a dated row to the decision log in
PHASES.md: **go** (phase 8), **narrow** (completion readahead helper
only, or residency only), or **kill** (publish the measurement study).
