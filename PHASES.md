# PHASES — living plan and project memory

This file is the hand-off document. A fresh session should read it first,
then IDEA_REVIEW.md and ARCHITECTURE.md. It is updated in the same commit as
the work it describes. Task IDs (`P3.2`) are referenced in commit messages
where useful.

## Current status

Phase 0 (idea review) done: **verdict KEEP, MODIFIED** (see IDEA_REVIEW.md).
Phases 2–4 done:
- GGUF expert map, verified on real Qwen3/gpt-oss/OLMoE headers;
- trace format, synthetic generator and analyzer;
- mp-core policy code and the simulator with Belady*.

Simulated LRU agrees exactly with the analyzer's LRU curve. Working on
phase 5 (OS layer, recorder, daemon skeleton).

## Next up

- P5.1–P5.2: mp-os traits, MockOps, Linux probes and actuators.

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
- [x] P1.2 Cargo workspace, crate skeletons, rustfmt/clippy config
- [x] P1.3 Python package skeleton (pyproject, ruff, pytest)
- [x] P1.4 Makefile (`test`, `lint`, `demo`, `bench`)
- [x] P1.5 CI workflow (fmt, clippy, tests, ruff, pytest, demo smoke)

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
- [x] P3.4 Synthetic generator: layers, experts, top-k, skew, cross-layer affinity, reuse, drift, timing
- [x] P3.5 Analyzer: reuse distance (units, bytes), exact LRU miss-ratio curve
- [x] P3.6 Analyzer: transitions + top-m recall, token reuse, popularity, per-layer timing
- [x] P3.7 `moepager synth` / `moepager analyze` CLIs

### Phase 4 — Simulator and policies
- [x] P4.1 mp-core OnlineStats (full and miss-only observation regimes)
- [x] P4.2 mp-core ResidencyEngine (V(e), budget, hysteresis, exploration)
- [x] P4.3 mp-core PrefetchPlanner (transition scores, deadline budget) + completion readahead flag
- [x] P4.4 Simulator engine: byte-capacity cache, FIFO I/O channel, stall accounting
- [x] P4.5 Policies: LRU, LFU, prefix-pin, static-freq oracle, V-residency, V+prefetch, Belady*
- [x] P4.6 Cross-check: simulated LRU == analyzer MRC
- [x] P4.7 `moepager sim` CLI: policy × budget sweep → CSV/markdown table

### Phase 5 — Recorder and daemon skeleton
- [x] P5.1 mp-os traits + MockOps
- [x] P5.2 Linux probes: mincore, cachestat; Linux ops: fadvise, mlock, process_madvise, /proc/pid/maps lookup
- [x] P5.3 Full-scan mincore diff recorder (page trace)
- [x] P5.4 Sentinel recorder (expert trace, miss-only)
- [x] P5.5 page→expert conversion; bpftrace script + ingest
- [x] P5.6 moepagerd: config, event loop, dry-run on traces with MockOps
- [x] P5.7 moepagerd: live mode wiring (sentinel source + LinuxOps) — untested on real engine

### Phase 6 — Benchmark harness
- [x] P6.1 `moepager replay` (mmap replay of an expert trace on a real file) + `moepager fault-io` microbench
- [x] P6.2 cgroup v2 runner (`systemd-run --user`, memory.max) and metric snapshots (vmstat, diskstats, PSI, memory.stat)
- [x] P6.3 llama.cpp baseline matrix script (default, -nr, -nr+mlock where fits, -nr+naive WILLNEED, -nr+daemon policies)
- [x] P6.4 Co-tenant probe app
- [x] P6.5 Python metric parsers + report table, with tests
- [x] P6.6 `make demo`, `make bench` entry points

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

- Prediction prefetch derives its deadline budget from the *observed* layer
  time. Under I/O stalls that estimate includes stall time, so the budget
  inflates and prefetch competes with demand misses. On synthetic traces this
  wastes ≈0.4 GB/token. The daemon only sees wall time, so a fix needs a
  compute-time estimate (e.g. the minimum gap over recent tokens). Open.

- btrfs device quirk: `stat()` reports the subvolume's anonymous device
  (0:45 here), while /proc/pid/maps and the filemap tracepoints use the
  superblock device (0:23). Mapping lookup therefore matches on inode plus
  device-or-path, and the eBPF filter must learn `s_dev` from our own
  /proc/self/maps (`kernel_dev_of_mapping`), not from `stat()`.
- `process_madvise(MADV_COLD)` demotion is implemented but untested: it
  needs `CAP_SYS_NICE` and a running engine.

- **btrfs readahead is 4 MiB** (`/sys/class/bdi/btrfs-1/read_ahead_kb =
  4096` on the dev laptop, versus 128 KiB for the NVMe block device).
  Sequential reads of one expert pull neighbouring experts' interiors into
  the cache. This causes false positives for miss-based expert inference.
  It may also mean mmap read-around on btrfs amplifies llama.cpp's expert
  reads far beyond the 128 KiB assumed in IDEA_REVIEW §1.2. Measure with
  `moepager fault-io` (bytes read vs touched) in Experiment C. The e2e
  recorder tests disable readahead (`POSIX_FADV_RANDOM`) to test recorder
  logic in isolation.
- btrfs `compress=zstd` reads whole 128 KiB compressed extents. Measured:
  reading a 544 KiB slice inserted 17 pages of the neighbouring expert.
  Page→expert conversion ignores pages within 128 KiB of slice edges.
- Sentinel and scan recorders are tested only against the replayer
  (pread, readahead off, cache dropped per token), not against llama.cpp.

- `posix_fadvise(WILLNEED)` had **no effect at all** on the GitHub Actions
  runner (0 pages populated in 10 s), although it works on the dev laptop.
  Likely cause: readahead disabled on that device (the kernel skips WILLNEED
  when `ra_pages == 0`). Unconfirmed. The mp-os tests skip in that case.
  For the product, completion readahead needs a fallback that doesn't depend
  on readahead settings: `MADV_POPULATE_READ` (5.14+) on the daemon's own
  mapping from a worker thread, which is synchronous. Not implemented yet.

## Decision log

| date | decision | why |
|---|---|---|
| 2026-09-24 | Keep the idea, modified (IDEA_REVIEW §7) | OS-level black-box niche is unoccupied. Assumptions corrected (repack, DONTNEED, hits invisible) |
| 2026-09-24 | Gate the daemon behind a go/no-go (K1–K4) | Prior evidence (arXiv 2608.12103) says prediction prefetch and fancy residency win less than intuition suggests |
| 2026-09-24 | Rust core + Python tooling, dual MIT/Apache-2.0 | Shared policy code between simulator and daemon. Python fixture writer is an independent check |
