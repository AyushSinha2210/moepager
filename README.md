# moepager

**Expert-aware page-cache management for running Mixture-of-Experts LLMs on
low-RAM Linux machines, without modifying the inference engine.**

> **Status: research prototype, before the go/no-go experiment.** The
> offline toolchain is built and tested on synthetic data and on real-model
> *headers*. Nothing has been benchmarked end to end against llama.cpp or a
> real model yet, and **no speedup is claimed**. See
> [PHASES.md](PHASES.md) for exact status and
> [IDEA_REVIEW.md](IDEA_REVIEW.md) for why the project has this shape.

## The idea

llama.cpp running a model larger than RAM via mmap leaves expert weights to
the kernel's generic page-cache policy. moepager:

1. parses the GGUF header to learn where every (layer, expert) unit lives.
   A unit is three slices in three tensors, hundreds of MB apart (verified
   on real Qwen3-30B-A3B, gpt-oss-20b and OLMoE headers);
2. watches which units fault into the page cache, as a black box: eBPF, or
   sentinel-page `mincore` polling without root;
3. learns routing statistics from those misses;
4. pins high-value experts, **completes partially-faulted experts with one
   bulk readahead**, and demotes cold ones. All of this goes through the
   shared page cache, with no copy of the weights and no engine changes.

Two findings shape it (details in IDEA_REVIEW.md):
- On x86 AVX2, default llama.cpp **repacks** Q4_K/MXFP4 experts into
  anonymous memory (77 % of Qwen3-30B-A3B Q4_K_M's expert bytes), so
  moepager needs `--no-repack`.
- On the dev laptop, mmap-faulting cold experts measured **≈0.45 GB/s with
  3.3–3.7× read amplification**, versus **≈2.4 GB/s with none** for
  chunked WILLNEED of each slice (smoke run, BENCHMARKS.md).
- The kernel silently truncates each `WILLNEED` to the device's readahead
  size, so moepager issues 128 KiB chunks.

## Quickstart

Requirements: Linux, Rust stable, Python ≥ 3.10 (+ pytest for tests). No
root, GPU or model needed.

```bash
make test      # Rust + Python tests (some use real page-cache syscalls on a temp file)
make demo      # real Qwen3 map + synthetic routing → analysis → simulation → daemon dry run → black-box recording
```

The demo prints, among other things, a policy table (LRU, LFU, static
frequency, V(e) with full or miss-only observation, predictive prefetch,
Belady\* oracle) at 15/25/35 % of expert bytes. **The routing is
synthetic**, so these tables exercise the tooling and are not results.

### Tools

```bash
B=target/release/moepager
$B gguf-map model.gguf -o map.json        # expert map + repack warning (header only; a truncated prefix works)
$B synth --like map.json --tokens 512 -o t.mpt
$B analyze t.mpt --map map.json            # reuse distance, exact LRU curve, routing predictability, timing
$B sim t.mpt --map map.json                # policy × capacity sweep incl. Belady*
$B record model.gguf -o rec.mpt            # black-box miss recording while llama.cpp runs (no root)
$B replay model.gguf t.mpt --cold          # touch units in trace order on the real file (simulator validation)
$B fault-io --file model.gguf              # fault vs bulk read bandwidth on your disk
target/release/moepagerd model.gguf --source trace:t.mpt --ops mock   # daemon dry run
```

### With a real model (untested on real hardware)

```bash
# Llama.cpp must run with -nr (no repack) so experts stay in the page cache.
make microbench MODEL=~/models/Qwen3-30B-A3B-Q4_K_M.gguf
make bench LLAMA_CLI=~/llama.cpp/build/bin/llama-cli MODEL=~/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

`make bench` runs the baseline matrix in BENCHMARKS.md. Each run goes
inside a rootless cgroup v2 memory wall (`systemd-run --user`), and the
daemon runs in the same cgroup. Live daemon operation, eBPF recording,
pinning beyond `RLIMIT_MEMLOCK` and `process_madvise` demotion are
**untested on real hardware**. See PHASES.md "Known issues" and
docs/PRIVILEGES.md.

## Repository map

| path | what |
|---|---|
| `crates/mp-gguf` | GGUF parser, expert map, repack detection |
| `crates/mp-trace` | `.mpt` trace format ([spec](docs/TRACE_FORMAT.md)) |
| `crates/mp-core` | policy core shared by simulator and daemon (stats, V(e) residency, prefetch, cost model) |
| `crates/mp-synth`, `mp-analyze`, `mp-sim` | synthetic traces, analysis, simulator |
| `crates/mp-os`, `mp-recorder` | OS layer (mincore, cachestat, fadvise, mlock, process_madvise), recorders, replayer |
| `crates/moepager`, `moepagerd` | CLI tools, daemon |
| `python/moepager_tools` | independent GGUF fixture writer, benchmark parsers, co-tenant probe |
| `bench/` | cgroup runner, llama.cpp baseline matrix, results |
| `bpf/moepager.bt` | bpftrace recorder (root) |
| `data/maps` | expert maps of real models (from headers only) |

## Documents

[PRD](PRD.md) · [ARCHITECTURE](ARCHITECTURE.md) · [PHASES](PHASES.md) ·
[IDEA_REVIEW](IDEA_REVIEW.md) · [RELATED_WORK](RELATED_WORK.md) ·
[BENCHMARKS](BENCHMARKS.md) · [ADRs](docs/adr) ·
[trace format](docs/TRACE_FORMAT.md) · [privileges](docs/PRIVILEGES.md) ·
[Tuning Guide](docs/TUNING_GUIDE.md) · [FAQ](docs/FAQ.md) ·
[Glossary](docs/GLOSSARY.md) · [Deep Dive](docs/ARCHITECTURE_DEEP_DIVE.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Compatibility](docs/MODEL_COMPATIBILITY.md) · [CONTRIBUTING](CONTRIBUTING.md)

## License

Dual-licensed under MIT or Apache-2.0, at your option.

