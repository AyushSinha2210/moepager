# PRD — moepager

## Problem

People run Mixture-of-Experts LLMs (Qwen3-30B-A3B, gpt-oss-20b, OLMoE, …)
with llama.cpp on laptops where the GGUF file is larger than free RAM. With
`--no-repack` (or with non-repacked quant types), the weights are served from
the mmap'd file through the page cache. The kernel's generic reclaim (MGLRU)
and readahead know nothing about:

- the file's structure: a (layer, expert) unit is three slices in three
  tensors hundreds of MB apart;
- MoE access patterns: per-layer popularity skew, token-to-token expert reuse,
  cross-layer affinity.

The result is repeated re-reads of the same experts, small synchronous
read-around I/O instead of large requests, and memory pressure spilling onto
other applications. See IDEA_REVIEW.md for verified details and numbers.

## Users

- **Primary:** an individual running a local MoE model with llama.cpp (or a
  llama.cpp-based app) on a 8–32 GB Linux laptop or desktop with an NVMe SSD,
  who doesn't want to patch or rebuild the engine.
- **Secondary:** systems researchers studying page-cache behaviour of LLM
  inference. They need the trace tooling, simulator and oracle on their own.

## Goals

1. **G1 — Measure first.** Provide the tooling to decide whether
   structure-aware page-cache management can beat the kernel:
   - GGUF expert map
   - black-box traces
   - offline analysis
   - simulator with a Belady oracle
   - real-kernel replay
   - harness
2. **G2 — Transparent speedup.** A daemon that improves decode tokens/s and
   reduces SSD bytes per token for an unmodified llama.cpp under a memory
   budget. Only if the go/no-go passes.
3. **G3 — Be a good neighbour.** Bound the model's page-cache footprint and
   shrink it under system memory pressure (PSI), so co-running apps stay
   responsive.
4. **G4 — Degrade gracefully by privilege.** It must be useful with no root
   (prefetch only), and better with `CAP_IPC_LOCK` (pinning), `CAP_SYS_NICE`
   (demotion) or root/BPF (precise tracing).

## Non-goals

- Modifying llama.cpp or any engine, LD_PRELOAD shims, or uprobes in the
  product path. An instrumented engine is allowed **only** for ground-truth
  experiments.
- GPU/VRAM tiering. This is CPU decode with host RAM and SSD.
- Changing model quality (no pruning, no low-precision substitution).
- Custom kernels or kernel modules. cache_ext is a research option only.
- Repacked/anonymous-memory weights. Swap management is out of scope; we
  detect the case and warn.
- Windows/macOS.

## Success metrics

All are measured under a cgroup v2 `memory.max` budget below model size, on
the same machine, and against the baselines in BENCHMARKS.md.

| metric | definition | target (if go) |
|---|---|---|
| decode tokens/s | llama.cpp eval rate (`llama-bench` tg or cli timings) | ≥ 1.25× the best kernel-only baseline at the same budget |
| SSD bytes read / token | Δ block-device read sectors × 512 / tokens | ≤ 0.8× the kernel baseline |
| major faults / token | Δ `pgmajfault` in the cgroup's memory.stat / tokens | report; expected ≪ baseline with completion readahead |
| co-running app slowdown | p95 latency of a fixed working-set probe app versus solo | ≤ 1.1× with PSI tuning on (phase 9) |
| oracle gap closed | (kernel − daemon) / (kernel − Belady) miss bytes | ≥ 40 % |
| overhead | daemon CPU time / wall time | ≤ 2 % of one core |

Go/no-go thresholds (K1–K4) are defined in IDEA_REVIEW.md §6 and take
precedence over these targets.

## Constraints

- Linux ≥ 5.10 (`process_madvise`). `cachestat` needs ≥ 6.5. Developed on
  6.19.
- Must run unprivileged. Privileged features are optional and detected at
  runtime.
- `RLIMIT_MEMLOCK` is small by default (8 MB here), so pinning needs
  `CAP_IPC_LOCK` or a raised limit.
- The daemon must live in the engine's cgroup for correct memory accounting.
- Everything must be testable without a GPU, root, eBPF or a real model:
  synthetic GGUF fixtures, synthetic traces, mock OS layer.
- No performance claims without measurements (BENCHMARKS.md marks TBD).
