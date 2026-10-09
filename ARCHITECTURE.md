# Architecture

## 1. Overview

```
            ┌──────────── offline / experiment path ─────────────┐
 GGUF file ─┤ mp-gguf: header parse → ExpertMap (units, slices,  │
            │           pages, sentinels, non-expert regions)    │
            └──────┬─────────────────────────────────────────────┘
                   │ map.json
  ┌────────────────┼──────────────────────────────┐
  │ sources of expert-level events (mp-trace fmt) │
  │  • mp-synth   synthetic MoE generator (full)  │
  │  • mp-recorder mincore full-scan / sentinel   │──► page or expert trace (.mpt)
  │  • bpf/moepager.bt (bpftrace) → ingest        │
  │  • llama.cpp eval-callback dump (phase 7)     │
  └────────────────┬──────────────────────────────┘
                   ▼
   mp-analyze (reuse distance, LRU MRC, transitions, timing) → JSON
   mp-sim (cache + I/O channel model, policies, Belady)      → CSV/JSON
                   ▲
                   │ same code
   mp-core: OnlineStats, ResidencyEngine (V(e)), PrefetchPlanner
                   │
                   ▼
   moepagerd: EventSource → mp-core → Actions → PageCacheOps (mp-os)
                                                ├ LinuxOps: fadvise/readahead,
                                                │  mmap+mlock, process_madvise
                                                └ MockOps (tests, dry-run)
```

### 1.1 Event Lifecycle Sequence

```text
llama.cpp            Linux Kernel           moepagerd (mp-core)          Disk I/O
   │                      │                          │                      │
   │─── faults on page ──►│                          │                      │
   │    of slice (up)     │── pulls in page ─────────┼─────────────────────►│
   │                      │                          │                      │
   │                      │◄── mincore sentinel poll ┤                      │
   │                      │    detects resident page │                      │
   │                      │                          │                      │
   │                      │                          │── Expert (l, e) seen │
   │                      │                          │   • Update stats     │
   │                      │                          │   • Rank V(e) value  │
   │                      │                          │                      │
   │                      │◄── posix_fadvise(WILLNEED)                      │
   │                      │    issued for gate + down slices in 128 KiB ───►│
   │                      │                          │    (Bulk read ahead) │
   │                      │                          │                      │
   │─── minor fault on ──►│ (already resident        │                      │
   │    gate/down slices  │  in page cache)          │                      │
   ▼                      ▼                          ▼                      ▼
```

## 2. Components

| crate | role | depends on |
|---|---|---|
| `mp-gguf` | GGUF v2/v3 header parser (no tensor data read), ggml type sizes, `ExpertMap` | — |
| `mp-trace` | `.mpt` trace format (spec: docs/TRACE_FORMAT.md), event types, reader/writer, CSV export, token inference | serde |
| `mp-core` | shared deterministic RNG, `OnlineStats`, `ResidencyEngine`, `PrefetchPlanner`, `CostModel` | mp-trace |
| `mp-synth` | synthetic MoE access-trace generator | mp-core, mp-trace |
| `mp-analyze` | reuse distance (Fenwick, O(N log N)), exact LRU miss-ratio curve, transitions, token reuse, popularity, timing | mp-trace |
| `mp-sim` | trace-driven simulator: byte-capacity cache, FIFO I/O channel, policies (LRU, LFU, prefix-pin, static-freq oracle, V-residency, V+prefetch, Belady) | mp-core, mp-trace |
| `mp-os` | `PageCacheOps` and `ResidencyProbe` traits. Linux impls (mincore, cachestat, fadvise, mlock, process_madvise, /proc/pid/maps lookup). Mock impl | libc |
| `mp-recorder` | mincore full-scan diff recorder, sentinel recorder, page→expert conversion, bpftrace output ingest | mp-gguf, mp-os, mp-trace |
| `moepager` (bin) | CLI: `gguf-map`, `synth`, `analyze`, `sim`, `trace-csv`, `record`, `ingest-bpftrace`, `page2expert`, `replay`, `fault-io` | all |
| `moepagerd` (lib + bin) | daemon: policies `none`, `willneed-all` (B5), `lru-pin` (B6), `v` (D1–D3); trace or sentinel sources; mock or Linux ops; graceful SIGINT; live mode untested against llama.cpp | mp-core, mp-os, mp-recorder |
| `python/moepager_tools` | independent GGUF fixture writer (cross-checks the Rust parser), bench metric parsers, plots | stdlib (+matplotlib optional) |
| `bench/` | cgroup v2 runner, llama.cpp baseline matrix, co-tenant probe, metric collection | bash, python |

## 3. Data model

- **Unit** `(layer, expert)`: the residency/prefetch granule. It is the union
  of the `ffn_{up,gate,down}_exps` slices for that expert, plus the
  `_exps.bias` rows when present. Units are indexed densely as
  `layer * n_experts + expert`.
- **Slice**: a byte range `[off + e·nb2, off + (e+1)·nb2)` of one 3-D expert
  tensor, or a row `[off + e·nb1, …)` of a 2-D expert bias tensor.
- **Pages**: slices are 32-B aligned, not page aligned. A unit's page set is
  the union of `floor(start/P) ..= floor((end-1)/P)`, and boundary pages can
  be shared by two units. Reverse lookup uses a sorted interval list with
  binary search.
- **Sentinel page** per slice: the page at the middle of the slice. It is
  chosen ≥ 128 KB away from both slice edges when the slice is large enough,
  so a neighbour's read-around doesn't trip it.
- **Non-expert regions** (attention, norms, router, embeddings, output) are
  "dense". They count against the budget and are assumed hot.

## 4. Policy core (mp-core) — shared by simulator and daemon

The daemon's decision logic is exactly the code the simulator evaluates. The
simulator just feeds it simulated observations.

**OnlineStats** keeps the following, updated from *observations*:
- per-unit EWMA usage per token;
- per-unit miss-rate-while-unpinned (miss count / tokens of exposure);
- cross-layer transition counts `T[l][e][e']` (sparse, decayed);
- token-to-token reuse per layer;
- an estimate of layer duration.

**Observation regimes.**
- `full`: every access is observed. This is the ground truth, and it is also
  what a root/DAMON-assisted mode could approximate.
- `miss-only`: only accesses that were page-cache misses are observed. This
  is the black-box reality (IDEA_REVIEW §1.3).

**Key design point.** Under miss-only, the quantity the pinning decision
needs is the number of *misses the unit would cause if not pinned*. That is
exactly what is observed for unpinned units. So the estimator is the miss
rate per token of exposure, conditioned on the unit being unpinned. For a
pinned unit, the estimate is frozen at its last unpinned value (no decay
while unobservable). An ε-exploration step periodically unpins a small random
fraction to refresh stale estimates. The simulator measures the cost of this
against `full`.

**ResidencyEngine.** Value per byte:

```
V(e) = rate(e) · (t_fault + s_e / B_demand) / s_e
```

`rate(e)` is the usage probability per token (full) or the unpinned miss
rate (miss-only). The engine greedily picks the highest-V units until the pin
budget `C_pin` is reached. It re-plans every `replan_tokens` tokens and
applies hysteresis (a newcomer must beat the weakest pinned unit by a factor
`h`) to limit churn. Output: the desired pin set → diff → `Pin`/`Unpin`
actions.

**PrefetchPlanner.** When units of layer `l` are observed in the current
token, it scores candidates for layers `l+1 … l+H` from the decayed
transition counts and keeps the top-m that are not known to be resident,
subject to the deadline budget:

```
Σ s_e ≤ B_bulk · t_layer_est · H
```

**Expert-completion readahead** is the prediction-free mode. As soon as one
slice of a unit is observed missing, it issues `WILLNEED` for the unit's
other slices. It is modelled in the simulator as demand misses served at
`B_bulk` instead of `B_fault`, plus a detection latency.

## 5. Observation backends

| backend | privilege | sees | latency | status |
|---|---|---|---|---|
| eBPF `filemap:mm_filemap_add_to_page_cache` (+`delete_from_page_cache`), filtered by `s_dev`/`i_ino` | root / CAP_BPF+CAP_PERFMON | every insertion/eviction (misses) incl. folio order | µs | bpftrace script + ingest written; untested (no root here) |
| sentinel `mincore` poll | none | first insertion of each slice's sentinel page | poll period (~1 ms) | implemented, tested on a real file here |
| full-scan `mincore` diff | none | all insertions/evictions at page level | 50–500 ms per scan (file-size dependent) | implemented, for recording, not control |
| `cachestat` per slice | owner/writer of file | resident count + `recently_evicted` (refaults) per slice | per call | probe implemented and tested |
| DAMON / page_idle (hits) | root | accessed bits → **hits** | sampling | future (phase 8), would enable `full` observation |

Page→expert conversion only counts pages at least 128 KiB inside a slice.
Neighbouring experts get pulled in by read-around, by readahead (4 MiB
windows on btrfs here) and by whole compressed extents (btrfs zstd); see
PHASES.md known issues.

Token boundaries are inferred in black-box traces: a new token starts when
the observed layer index drops below the previous one by more than a
threshold. Experts at layer 0 are evidence of a new token.

## 6. Actuators (`PageCacheOps`)

| op | Linux mechanism | privilege | notes |
|---|---|---|---|
| `prefetch(range)` | `posix_fadvise(WILLNEED)` on the file fd, **in 128 KiB chunks** | none | populates the shared page cache. The engine takes a minor fault. The kernel truncates each call to `max(io_pages, ra_pages)`, hence the chunking |
| `pin(range)` / `unpin` | `mlock`/`munlock` on the daemon's own `MAP_SHARED` read-only mapping | `CAP_IPC_LOCK` or `RLIMIT_MEMLOCK` | the shared folio becomes unevictable for every mapper |
| `demote(range)` | `process_madvise(pidfd, MADV_COLD)` on the engine's mapping (address found through `/proc/pid/maps`, matched by inode + device-or-path; btrfs reports different devices in `stat()` and maps) | `CAP_SYS_NICE` + ptrace-read | `fadvise(DONTNEED)` doesn't work on mapped pages (verified) |
| `residency(range)` | `mincore` / `cachestat` | none / owner | probes |

All ops go through the trait. `MockOps` records calls and simulates
residency, so the daemon loop is unit-tested deterministically.

## 7. Simulator model (mp-sim)

- **Cache:** a byte capacity `C_exp` = budget − dense bytes, at unit
  granularity with per-unit sizes from `map.json` (or uniform).
- **Clock:** for each token and each layer, the layer's units are accessed;
  then the clock advances by `t_layer` of compute. A hit costs 0. A miss
  enqueues a read on a single FIFO I/O channel:
  `start = max(now, channel_free)`, `done = start + t_fault + s/B`. The
  engine stalls until `done`. Prefetches use the same channel (so bad
  prefetch delays demand reads), are inserted on issue, and a demand access
  to an in-flight unit stalls until its completion.
- **Belady:** furthest next use among non-pinned residents. With variable
  sizes this is the classic heuristic, not the exact optimum (exact
  variable-size MIN is NP-hard). It is labelled `belady*` in outputs.
- **Outputs per policy:** hits, misses, SSD bytes read (demand + prefetch),
  wasted prefetch bytes, stall seconds, estimated tok/s, pin churn.
- **Approximations:** MGLRU is modelled as LRU, and the kernel's demand
  read-around is folded into `B_fault`. Phase 7 validates the simulator
  against the real kernel with `moepager replay`.

## 8. Language and library choices

- **Rust** for everything on the hot path and for the simulator:
  - The daemon needs predictable latency, low overhead and direct syscalls
    (`libc`).
  - The simulator runs millions of events × policies × budgets.
  - One language for core logic means the simulator and daemon share
    `mp-core` verbatim.
  - Strong testing story (`cargo test`, clippy).
- **Dependencies are kept small:** `serde`/`serde_json` (map/trace headers,
  reports), `clap` (CLI), `anyhow`/`thiserror`, `libc`.
- **No `rand` crate.** A tiny SplitMix64/xoshiro256** generator in `mp-core`
  keeps synthetic traces bit-identical across dependency upgrades.
- **eBPF:** a bpftrace script for recording now (simple, auditable, no build
  step). Phase 8 moves to `libbpf-rs` with a CO-RE C program for the live
  daemon. Not `aya`: it needs a nightly + bpf-linker toolchain, while libbpf
  is the kernel-blessed loader and works with `clang` (available).
- **Python (stdlib + pytest):**
  - an **independent** GGUF fixture writer, so parser tests aren't
    self-confirming;
  - parsers for llama.cpp/vmstat/diskstats/PSI output;
  - plots (matplotlib, optional).
- **Bash** for cgroup and baseline orchestration (`systemd-run --user`, no
  root).

## 9. Testing without special hardware

| component | how it's tested here |
|---|---|
| GGUF parser/map | Python-written fixtures (dense + MoE + bias + unaligned offsets + v2), golden checks of slice offsets. Optional real-header test if `MOEPAGER_REAL_HEADERS` is set |
| trace format | round-trip, corruption/version errors, CSV export |
| synth | determinism by seed, statistical properties (top-k distinct, reuse ≈ target, skew monotone) |
| analyze | hand-computed reuse distances. **LRU MRC from reuse distances must equal simulated LRU misses** (cross-check of two independent implementations) |
| sim | tiny traces with known answers, Belady ≤ every policy, monotone in capacity, conservation of bytes |
| mp-core | engine respects budget, converges to the hot set, hysteresis bounds churn, miss-only estimator matches the analytic rate |
| mp-os | Linux ops on a real temp file in `$HOME` (not tmpfs): mincore/cachestat/fadvise behaviour. Mock ops |
| recorder | end-to-end on a real file: a replayer touches synthetic expert units, the sentinel recorder recovers them (cache dropped with fadvise between tokens) |
| daemon | dry-run on synthetic traces with MockOps, asserting action sequences |
| bench parsers | sample outputs checked into `python/tests/data` |

Anything that needs root, eBPF, `CAP_*`, llama.cpp or a real model is listed
under "Known issues / untested on real hardware" in PHASES.md.


---

### Memory Budget Partitioning

```
Total Memory Budget (C)
|-- Dense Regions (~0.5 - 1.2 GB) [Always Hot: Attention, Norms, Routers, Embeddings]
|-- Pinned Expert Units (C_pin)   [Locked in RAM via mlock]
`-- Dynamic Readahead Headroom    [Temporary folios populated by WILLNEED readahead]
```

