# PHASES — living plan and project memory

This file is the hand-off document. A fresh session should read it first,
then IDEA_REVIEW.md and ARCHITECTURE.md. It is updated in the same commit as
the work it describes. Task IDs (`P3.2`) are referenced in commit messages
where useful.

## Current status

Phase 0 (idea review) done: **verdict KEEP, MODIFIED** (see IDEA_REVIEW.md).
Building the offline toolchain (phases 1–6), which is roughly the first half
of the project.

## Next up

- P1: repository scaffold.

## Phases

### Phase 0 — Idea review ✅
- [x] P0.1 Verify GGUF expert layout on real headers (Qwen3-30B-A3B, gpt-oss-20b, OLMoE)
- [x] P0.2 Verify llama.cpp decode access path (mul_mat_id, repack) from source
- [x] P0.3 Verify tracepoint fields, fadvise/mlock/mincore/cachestat semantics (source + experiments here)
- [x] P0.4 Related-work search (llama.cpp PRs/discussions, papers, kernel work)
- [x] P0.5 Back-of-envelope, counter-arguments, go/no-go design with kill criteria
- [x] P0.6 Write IDEA_REVIEW.md, RELATED_WORK.md

### Phase 1 — Scaffold and docs
- [ ] P1.1 PRD, ARCHITECTURE, PHASES, BENCHMARKS, README, CONTRIBUTING, trace spec, ADRs, PRIVILEGES
- [ ] P1.2 Cargo workspace, crate skeletons, rustfmt/clippy config
- [ ] P1.3 Python package skeleton (pyproject, ruff, pytest)
- [ ] P1.4 Makefile (`test`, `lint`, `demo`, `bench`)
- [ ] P1.5 CI workflow (fmt, clippy, tests, ruff, pytest, demo smoke)

### Phase 2 — GGUF map
- [x] P2.1 ggml type table (block size, type size) incl. K-quants, IQ, MXFP4
- [x] P2.2 GGUF v2/v3 header parser (metadata KV, tensor infos, alignment, data offset)
- [x] P2.3 Python independent fixture writer + generated fixtures
- [x] P2.4 ExpertMap: units, slices (3-D weights, 2-D bias rows), page sets, shared boundary pages
- [x] P2.5 Reverse lookup page → (unit, slice) / dense region; sentinel pages
- [x] P2.6 `moepager gguf-map` CLI → map.json + summary
- [x] P2.7 Repack-risk detection (types × host CPU flags) with warning

### Phase 3 — Traces, synthetic generator, analyzer
- [x] P3.1 Trace format spec + reader/writer (expert and page records), CSV export
- [x] P3.2 Token-boundary inference for black-box traces
- [x] P3.3 Deterministic RNG (SplitMix64/xoshiro256**) + Zipf sampler
- [ ] P3.4 Synthetic generator: layers, experts, top-k, skew, cross-layer affinity, reuse, drift, timing
- [ ] P3.5 Analyzer: reuse distance (units, bytes), exact LRU miss-ratio curve
- [ ] P3.6 Analyzer: transitions + top-m recall, token reuse, popularity, per-layer timing
- [ ] P3.7 `moepager synth` / `moepager analyze` CLIs

### Phase 4 — Simulator and policies
- [ ] P4.1 mp-core OnlineStats (full and miss-only observation regimes)
- [ ] P4.2 mp-core ResidencyEngine (V(e), budget, hysteresis, exploration)
- [ ] P4.3 mp-core PrefetchPlanner (transition scores, deadline budget) + completion readahead flag
- [ ] P4.4 Simulator engine: byte-capacity cache, FIFO I/O channel, stall accounting
- [ ] P4.5 Policies: LRU, LFU, prefix-pin, static-freq oracle, V-residency, V+prefetch, Belady*
- [ ] P4.6 Cross-check: simulated LRU == analyzer MRC
- [ ] P4.7 `moepager sim` CLI: policy × budget sweep → CSV/markdown table

### Phase 5 — Recorder and daemon skeleton
- [ ] P5.1 mp-os traits + MockOps
- [ ] P5.2 Linux probes: mincore, cachestat; Linux ops: fadvise, mlock, process_madvise, /proc/pid/maps lookup
- [ ] P5.3 Full-scan mincore diff recorder (page trace)
- [ ] P5.4 Sentinel recorder (expert trace, miss-only)
- [ ] P5.5 page→expert conversion; bpftrace script + ingest
- [ ] P5.6 moepagerd: config, event loop, dry-run on traces with MockOps
- [ ] P5.7 moepagerd: live mode wiring (sentinel source + LinuxOps) — untested on real engine

### Phase 6 — Benchmark harness
- [ ] P6.1 `moepager replay` (mmap replay of an expert trace on a real file) + `moepager fault-io` microbench
- [ ] P6.2 cgroup v2 runner (`systemd-run --user`, memory.max) and metric snapshots (vmstat, diskstats, PSI, memory.stat)
- [ ] P6.3 llama.cpp baseline matrix script (default, -nr, -nr+mlock where fits, -nr+naive WILLNEED, -nr+daemon policies)
- [ ] P6.4 Co-tenant probe app
- [ ] P6.5 Python metric parsers + report table, with tests
- [ ] P6.6 `make demo`, `make bench` entry points

### Phase 7 — Go/no-go on real hardware (NEXT after this session)
- [ ] P7.1 Ground-truth expert trace capture tool (libllama eval callback on `ffn_moe_topk-*`), experiment-only
- [ ] P7.2 Download models (OLMoE, Qwen3-30B-A3B Q4_K_M, gpt-oss-20b MXFP4); build llama.cpp
- [ ] P7.3 Experiment A: simulator sweeps on real traces
- [ ] P7.4 Experiment B: `moepager replay` under memory.max versus simulator LRU (validation)
- [ ] P7.5 Experiment C: `moepager fault-io` full run (fault vs bulk bandwidth)
- [ ] P7.6 Experiment D: end-to-end llama.cpp baselines (MGLRU on/off needs root)
- [ ] P7.7 Decide go/no-go against K1–K4, record in decision log

### Phase 8 — Live daemon (only if go)
- [ ] P8.1 libbpf-rs CO-RE tracer for filemap tracepoints (inode filter in-kernel)
- [ ] P8.2 Live actuators hardened (mlock budget accounting, process_madvise demotion, fallback chain)
- [ ] P8.3 Same-cgroup enforcement and accounting checks
- [ ] P8.4 Optional DAMON-assisted hit observation (root)

### Phase 9 — Co-tenant protection
- [ ] P9.1 PSI-driven budget controller (`/proc/pressure/memory`, cgroup `memory.pressure`)
- [ ] P9.2 Co-tenant benchmark

### Phase 10 — Full benchmark campaign and reporting
- [ ] P10.1 Full matrix on ≥ 2 machines, ≥ 3 models
- [ ] P10.2 Dashboard / report generation

### Future ideas (not scheduled)
- cache_ext backend implementing V(e) in-kernel (needs a patched kernel).
- Support other mmap engines (ollama is llama.cpp; check mistral.rs and candle mmap paths).
- Auto-tuning `read_ahead_kb` / `MADV_RANDOM` for the dense part.

## Known issues / untested on real hardware

- Repack default: llama.cpp on AVX2 repacks Q4_0/Q4_K/IQ4_NL/MXFP4 into
  anonymous memory. moepager needs `--no-repack`. Verified from source only,
  not yet measured.

## Decision log

| date | decision | why |
|---|---|---|
| 2026-09-24 | Keep the idea, modified (IDEA_REVIEW §7) | OS-level black-box niche is unoccupied. Assumptions corrected (repack, DONTNEED, hits invisible) |
| 2026-09-24 | Gate the daemon behind a go/no-go (K1–K4) | Prior evidence (arXiv 2608.12103) says prediction prefetch and fancy residency win less than intuition suggests |
| 2026-09-24 | Rust core + Python tooling, dual MIT/Apache-2.0 | Shared policy code between simulator and daemon. Python fixture writer is an independent check |
