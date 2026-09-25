# Benchmarks

> **No results below are measured unless a row says so explicitly. Every
> unmeasured cell is TBD.** Numbers are only added from runs whose raw
> outputs are committed under `bench/results/<date>-<host>/` (git-ignored by
> default; force-add the summary).

## Methodology

### Budget enforcement (cgroup v2, no root)
The engine **and** the daemon run in one transient scope with a hard memory
wall:

```
systemd-run --user --scope -p MemoryMax=<C> -p MemorySwapMax=<S> -- <cmd>
```

(`bench/cgroup_run.sh` wraps this.)
- Page-cache pages are charged to the cgroup of the task that first reads
  them. Running the daemon outside the scope would let it smuggle memory
  past the budget (IDEA_REVIEW §1.4).
- A cgroup wall is used instead of a memory balloon. The balloon
  methodology inflates MGLRU traffic about 2× (arXiv 2608.12103), so
  results with a balloon are not comparable.
- `MemorySwapMax=0` for the page-cache configurations. For the default
  (repack) baseline, swap/zram is allowed and recorded, because that is how
  the default behaves on a real laptop.

### Cold start
Before each run, evict the model file from the page cache with
`moepager fault-io --drop <file>`. That calls `posix_fadvise(DONTNEED)` on
an unmapped fd, which works without root for any file (verified on btrfs,
kernel 6.19). There is a 30 s idle period before the run.

### Workload
- `llama-bench -p 0 -n 128 -r 3` for decode throughput.
- `llama-cli` with fixed prompts from 3 domains (chat, code, prose), 256
  generated tokens, `--temp 0`, a fixed seed, and `-t <physical cores>`.

### Metrics

| metric | source |
|---|---|
| tokens/s (decode) | llama.cpp timings line (`eval time … tokens per second`) or llama-bench `tg` |
| SSD bytes read / token | Δ field 3 (sectors read) of `/sys/block/<dev>/stat` × 512 / tokens |
| major faults / token | Δ `pgmajfault` in the scope's `memory.stat` (fallback: `/proc/vmstat`) / tokens |
| refaults | Δ `workingset_refault_file` in `memory.stat` |
| memory PSI | `/sys/fs/cgroup/<scope>/memory.pressure` and `/proc/pressure/memory` avg10, plus Δ total |
| co-tenant slowdown | `bench/cotenant_probe.py` p50/p95 latency of touching its own 512 MB working set every 250 ms, relative to the solo run |
| daemon overhead | `/proc/<pid>/stat` utime+stime / wall |

### Machines

| id | CPU | RAM | SSD | kernel |
|---|---|---|---|---|
| L1 (dev laptop) | i5-12450H (8C/12T, AVX2) | 15 GiB + 8 GiB zram | Solidigm P41 Plus 512 GB (PCIe4 QLC), btrfs | 6.19.14 Fedora 42 |

## Baselines

| id | configuration |
|---|---|
| B0 | default llama.cpp (mmap + repack). Swap/zram allowed |
| B1 | `--no-mmap` (anonymous memory). Swap/zram allowed |
| B2 | `-nr` (mmap, no repack). Kernel default (MGLRU on) |
| B3 | `-nr`, MGLRU off (`/sys/kernel/mm/lru_gen/enabled = n`, root) |
| B4 | `-nr --mlock`. Only where the model fits the budget |
| B5 | `-nr` + naive readahead: `moepagerd --policy willneed-all` (WILLNEED the next layer's whole expert tensors) |
| B6 | `-nr` + `moepagerd --policy lru-pin` (LRU over experts, pinned) |
| D1 | `-nr` + `moepagerd --policy v` (V(e)-residency) |
| D2 | `-nr` + `moepagerd --policy v --prefetch completion` |
| D3 | `-nr` + `moepagerd --policy v --prefetch completion,predict` |
| O | Belady* oracle (simulator only, on the ground-truth trace) |

## Results

### Microbenchmark: fault-driven vs bulk read bandwidth (Experiment C)

| machine | unit size | threads | mmap-fault GB/s | WILLNEED GB/s | pread 1 MiB GB/s | notes |
|---|---|---|---|---|---|---|
| L1 | TBD | TBD | TBD | TBD | TBD | TBD |

### Simulator vs kernel validation (Experiment B)

| model | budget | sim LRU miss GB/token | kernel miss GB/token | ratio |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

### End-to-end (Experiment D)

| model | budget | config | tok/s | SSD GB/token | majflt/token | memory PSI avg10 | co-tenant p95× |
|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M | TBD | B0…D3 | TBD | TBD | TBD | TBD | TBD |
| gpt-oss-20b MXFP4 | TBD | B0…D3 | TBD | TBD | TBD | TBD | TBD |
| OLMoE-1B-7B Q4_K_M | TBD | B0…D3 | TBD | TBD | TBD | TBD | TBD |

### Simulator results on synthetic traces
Synthetic traces only exercise the tooling. They are **not** evidence about
real models. `make demo` prints such a table. It is deliberately not copied
here.
