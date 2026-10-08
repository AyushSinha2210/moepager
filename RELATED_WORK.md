# Related work, and exactly how moepager differs

moepager's position has three parts:
- **black-box**: no engine changes, no LD_PRELOAD, no uprobes in the product
  path;
- **OS-level**: it works only through the shared page cache with stock
  syscalls;
- **model-structure-aware**: it reads the GGUF tensor map.

Each entry below says which of these three the work lacks. Entries marked
*(not reviewed in depth)* are listed for completeness, and no claims are made
about their internals.

## A. Inside the inference engine (MoE offloading / caching / prefetch)

| work | where | mechanism | how moepager differs |
|---|---|---|---|
| mixtral-offloading (Eliseev & Mazur, 2023) | PyTorch, GPU | LRU expert cache on GPU, speculative expert prefetch from the next layer's gate | in-app, GPU memory tier |
| MoE-Infinity (2024) | PyTorch | request-level activation tracing, sparsity-aware expert cache, prefetch | in-app, needs the engine's routing |
| HOBBIT (2024) | llama.cpp fork (~8k LoC) | mixed-precision experts, cache misses served by low-precision copies | modifies the engine and the model quality |
| ProMoE | llama.cpp / transformers integration | learned predictor for proactive expert caching | in-app predictor with router access |
| Inter-layer expert affinity (Yao et al., IPDPS 2024) | distributed training/inference | exploits P(e' at l+1 \| e at l) | we *learn* the same statistic, but from page-cache misses |
| PreScope, DuoServe-MoE, fMoE, OD-MoE, SwapMoE, AdapMoE, EdgeMoE, Pre-gated MoE, SiDA-MoE, ExpertFlow *(not reviewed in depth)* | various | prediction / caching / bit-width tricks | all in-app |
| llama.cpp PR #25294 (2026, open) | llama.cpp | O_DIRECT expert streaming into a per-layer slot cache. **Disables mmap.** Async I/O workers | in-app. Bypasses the page cache entirely |
| llama.cpp PR #27861, #26824, #29887, issue #20757, discussion #27149 | llama.cpp | LRU expert cache, heatmap + mmap pinning, GPU cache of host experts, design proposals | in-app. #26824 is the closest *mechanism* (pin hot experts in mmap) but needs the engine |
| llama-cpp-expert-sniper (2026) | llama.cpp patch (~430 LoC) | eval callback reads top-k ids at `mul_mat_id`, `MADV_WILLNEED` on the selected experts | uses router ids from inside the engine. Its result that a user-space LRU copy *lost* to plain madvise motivates our "no duplicate copy, manage the shared page cache" design |
| routhjim/llama.cpp-lab PR #9 | llama.cpp fork | bounded mlock'd cache of expert slabs. Reports WILLNEED "within 1.5 % of nothing" | in-app. Their negative WILLNEED result is one of the reasons prediction prefetch is demoted here |
| jiro-prog/llamacpp-moe-expert-cache | llama.cpp, Windows | expert cache + next-layer read-ahead | in-app, Windows |
| ik_llama.cpp draft #2449 | fork | notes exact cross-layer prefetch is impossible in its pipeline. `--defer-experts` | in-app |

**Summary:** every MoE-specific system knows the router decision because it
*is* the engine. moepager never sees router outputs. It infers expert use
from page-cache misses (§C) and acts only on the shared page cache.

## B. On-device weight management (dense or neuron-level)

| work | mechanism | difference |
|---|---|---|
| LLM in a flash (Alizadeh et al., Apple, 2023) | predicted-sparsity neuron loading from flash, row-column bundling, sliding window | in-app, dense FFN sparsity. *Bundling* (co-locating what's read together) is the analogue of our expert-completion readahead, which reaches the same effect without rewriting the file |
| PowerInfer / PowerInfer-2 (2023–24) | hot/cold neuron split, neuron-cluster I/O on phones | in-app, ReLU-sparse dense models |
| mzCache *(not reviewed in depth)* | — | — |

## C. Kernel and OS mechanisms (generic, structure-blind)

| work | what it does | difference |
|---|---|---|
| Si et al., *Who Should Own the Expert Cache?* (arXiv 2608.12103, 2026) | Evaluates the **page cache as the MoE expert tier** on GH200: LRU/LFU/static/Belady on router traces (incl. llama.cpp-captured Qwen3-30B-A3B), `fadvise(WILLNEED)` lookahead, MGLRU vs classic LRU, cgroup wall vs balloon | **The closest work.** It is a datacenter measurement study using a pread-based engine with router access. It has no GGUF map, no black-box operation, no mmap-fault bandwidth study, and no consumer laptop. We reuse its methodology (equal-memory comparisons, Belady bound, cgroup wall rather than balloon) and treat its results as priors |
| cache_ext / cachebpf (Zussman, Zarkadas, Cidon et al., 2025) | eBPF hooks that replace page-cache eviction per cgroup | needs a patched kernel. Generic. A possible *backend* for a future in-kernel V(e) policy |
| LearnedCache (2026) | perceptron eviction on cache_ext | generic |
| MGLRU (Linux 6.1+) | multi-generation LRU with page-table A-bit scanning | the default we compete with. Structure-blind |
| DAMON / DAMOS (+ PSI-driven quota auto-tuning) | access sampling + schemes (pageout, lru_prio) | root-only, region-granular, structure-blind. A possible *hit* oracle if root is available (ARCHITECTURE §5) |
| UBM / loop-detection buffer management (2000s) | detect sequential / looping references, MRU for loops | the classic insight that LRU fails on loops applies to dense layer sweeps, not to MoE expert reuse |
| vmtouch | lock or evict file ranges from the page cache (user-specified) | static, structure-blind. No policy. moepager's pin/prefetch actuators are vmtouch-like primitives driven by a model-aware policy |
| preload, ureadahead, systemd-readahead | boot/app-launch readahead from recorded traces | file-granular, launch-time only |
| `process_madvise` (5.10+), `cachestat` (6.5+), `MADV_COLD`/`MADV_PAGEOUT` (5.4+) | the actuators and probes we use | — |

## D. llama.cpp's own options

| option | effect for MoE larger than RAM |
|---|---|
| default (mmap + repack on AVX2/NEON) | repacked types (Q4_0/Q4_K/IQ4_NL/MXFP4 on AVX2) are **copied to anonymous memory**. Pressure means swap/zram, not page cache. A mandatory baseline |
| `--no-repack` (`-nr`) | all weights served from the mmap'd file. Precondition for moepager. Slower kernels |
| `--mlock` | locks the whole model. Impossible when model > RAM |
| `--no-mmap` | reads everything into anonymous memory. Swap-backed under pressure |
| `--cpu-moe`, `--n-cpu-moe`, `-ot` | place expert tensors on CPU buffers (GPU setups). A reported side effect: experts in anonymous memory |
| `--no-mmap-prefetch` (PR #29250) | skips the load-time whole-file WILLNEED |

## A caveat when comparing with prior WILLNEED results

Several negative results above concern WILLNEED-style prefetch:
- routhjim's "within 1.5 % of nothing";
- the 0.3–5 % gains in arXiv 2608.12103.

The kernel silently truncates each `posix_fadvise(WILLNEED)` /
`madvise(MADV_WILLNEED)` request to `max(bdi->io_pages, ra->ra_pages)`
(often 128 KiB, see IDEA_REVIEW §1.4). A single call covering a whole
multi-MB expert therefore prefetches only its head unless it is split. We
don't know whether those works split their requests. Phase 7 should
reproduce their setup both ways before treating their results as priors
for moepager.

## What exactly is new here

1. **A GGUF-structure map used as an OS-level policy input.** It covers
   (layer, expert) units spanning three tensors plus bias rows, with shared
   boundary pages, mapped to page ranges.
2. **Black-box expert-usage inference from page-cache misses**: an eBPF
   tracepoint, or sentinel-page `mincore`/`cachestat` polling without root.
   This comes with an explicit treatment of *miss-only* (censored)
   observation, which in-app systems never face.
3. **Expert-completion readahead**: a structure-derived, prediction-free
   bandwidth optimisation that the kernel cannot express, because the
   ranges are in different tensors.
4. **Residency through pin + demote on the shared page cache**, with no
   duplicate copy and graceful degradation by privilege level.
5. **A consumer-laptop go/no-go methodology.** It uses a cgroup-wall budget,
   compares default llama.cpp (repack + zram) against `-nr` + kernel against
   `-nr` + oracle, and includes a simulator validated against a real-kernel
   replay.
