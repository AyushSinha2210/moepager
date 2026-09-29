# Idea review (Step 0)

**Question.** Should we build a user-space daemon that makes MoE inference in
llama.cpp (mmap mode, model larger than RAM) faster by managing the page cache
with knowledge of the GGUF expert layout, without changing llama.cpp?

**Verdict: KEEP, MODIFIED.** The OS-level, black-box angle is still unoccupied
as far as I can find. Several of the original assumptions are wrong, though,
and three mechanisms must change. The project is also gated: the first real
deliverable is a go/no-go experiment with explicit kill criteria (§6). The
simulator, trace tooling and harness built in this phase are what make that
experiment possible.

Legend: ✅ verified (how is stated), ⚠️ partly true / needs care,
❌ wrong, 🔬 unverified, to measure.

---

## 1. Technical assumptions, checked

### 1.1 GGUF expert layout — ✅ contiguous per expert, ❌ "one contiguous slice per expert"

*How verified:* I wrote a throwaway parser and ran it on the real headers of
three published models. Only the first 8–16 MB of each file was downloaded,
using HTTP range requests.

| model (file) | file size | layers × experts, top-k | expert bytes | per-expert slice (up / gate / down) |
|---|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M (unsloth) | 18.56 GB | 48 × 128, k=8 | 17.55 GB (94.6 %) | 884,736 / 884,736 / 1,290,240 (Q6_K) or 884,736 (Q4_K) B |
| gpt-oss-20b MXFP4 (ggml-org) | 12.11 GB | 24 × 32, k=4 | 10.18 GB (84.1 %) | 4,406,400 × 3 B, plus 11,520 B bias rows |
| OLMoE-1B-7B Q4_K_M (allenai) | 4.21 GB | 16 × 64, k=8 | 3.90 GB (92.6 %) | 1,179,648 / 1,179,648 / 1,720,320 B |

- Expert tensors are 3-D with `ne = [ne0, ne1, n_expert]`. In ggml, `ne[0]` is
  the fastest-varying dimension, so the expert index is outermost. Expert `e`
  of a tensor is the byte range `[off + e·nb2, off + (e+1)·nb2)`, which is
  contiguous.
- llama.cpp's CPU kernel (`ggml_compute_forward_mul_mat_id`) addresses
  `src0->data + cur_a*nb02` and reads only full rows of the selected experts
  (source read).
- One "expert" at one layer is **three** slices in three tensors
  (`ffn_up_exps`, `ffn_gate_exps`, `ffn_down_exps`) about 110–410 MB apart. In
  gpt-oss there are also three 2-D bias tensors with the expert index in
  `ne[1]`. **The mapper must group slices into an expert unit; it cannot
  assume one byte range per expert.**
- Data offsets are aligned to 32 bytes (`general.alignment`), **not to
  pages**. Neighbouring experts share one boundary page, so the map has to
  handle pages that belong to two units.
- Down-projection quantisation varies by layer in the `_M` mixes (Q6_K in
  half the layers, Q4_K in the rest), so expert sizes differ across layers.
- The claim in llama.cpp discussion #27149 that GGUF "interleaves" the 128
  experts, so that one expert touches all 6,912 pages, is **false** for these
  files.

### 1.2 How llama.cpp mmap touches weights during decode — ⚠️ **the biggest surprise**

- ✅ Non-expert weights (attention, norms, router, `output.weight`) are read
  in full on every token. `token_embd` is read one row per token
  (`get_rows`).
- ❌ **On x86 with AVX2 (this laptop: i5-12450H) llama.cpp repacks Q4_0,
  Q4_K, IQ4_NL and MXFP4 weights by default, including 3-D MoE expert
  tensors.** Source read: `ggml-cpu/repack.cpp`, where
  `ggml_repack_get_optimal_repack_type` selects `q4_K_8x8_q8_K` and
  `mxfp4_8x8_q8_0` under AVX2, and `supports_op` accepts `MUL_MAT_ID` with a
  3-D `src0`. Repacked tensors are copied into a separate CPU_REPACK buffer of
  **anonymous memory**. The mmap'd file pages are touched once at load time
  and never again. Consequences:
  - Qwen3-30B-A3B Q4_K_M: every Q4_K expert tensor is repacked, which is
    **77 % of expert bytes** (`moepager gguf-map` on the real header, AVX2
    host). Only the Q6_K down tensors stay file-backed.
  - gpt-oss-20b MXFP4: all expert weights become anonymous memory.
  - Under memory pressure these pages go to **swap or zram, not the page
    cache**. A page-cache daemon has nothing to manage there.
  - **Precondition for this project:** run llama.cpp with `--no-repack`
    (`-nr`), or use a quantisation type that isn't repacked on the host CPU
    (e.g. Q5_K/Q6_K/Q8_0 on x86). The daemon detects the situation from the
    GGUF types and host CPU flags and warns. Repacked kernels are faster, so
    `--no-repack` costs some compute. Default llama.cpp (repack + swap) is
    therefore a **mandatory baseline**: we must beat it, not only beat mmap
    without repack. 🔬 The magnitude is unmeasured. A third-party report
    (llama-cpp-expert-sniper) independently notes that "CPU_REPACK doubles
    memory use".
- ⚠️ A fault on a missing page runs synchronous mmap read-around sized by
  the backing device's readahead. **On this laptop's btrfs that is 4 MiB**
  (`/sys/class/bdi/btrfs-1/read_ahead_kb`), not the NVMe block device's
  128 KiB. Six to eight compute threads fault different rows of the same
  expert concurrently.
  ✅ **Measured (smoke run, `moepager fault-io`, BENCHMARKS.md
  Experiment C):** cold 2.86 MB expert units faulted through mmap arrive
  at **0.41–0.52 GB/s with 3.3–3.6× read amplification**. Issuing one
  `WILLNEED` per slice first gives **1.75–2.07 GB/s with 1.00×
  amplification**. For 13.25 MB (gpt-oss-sized) units: 0.89 vs ≈2.0 GB/s.
  The bandwidth lever (§4) is therefore real on this machine, roughly 4×
  for Qwen3-sized experts. Still to show: whether it survives llama.cpp's
  actual access pattern (P7.5/P7.6).
- ✅ Up, gate and down of the same expert sit in different tensors, so
  kernel readahead can never fetch "the rest of the expert". The kernel has
  no way to know the three ranges belong together. Only a structure-aware
  agent can issue that readahead.

### 1.3 Observation channels — ⚠️ **black-box tracing sees misses, not hits**

- ✅ `filemap:mm_filemap_add_to_page_cache` exists. Its fields are `i_ino`,
  `pfn`, `index`, `s_dev` and `order` (source:
  `include/trace/events/filemap.h`). Note that `order` matters with large
  folios, where one event can cover up to 2^order pages. Related events:
  `mm_filemap_delete_from_page_cache`, `mm_filemap_fault`,
  `mm_filemap_map_pages`, `mm_filemap_get_pages`.
- ❌ **Hits are invisible.** A weight page already in the page cache and
  mapped into llama.cpp is read through the page table with no kernel entry.
  The tracepoints fire on insertion (miss) and fault, not on access.
  `mincore` and `cachestat` also only show residency. The only hit signals
  are page-table accessed bits: page_idle bitmap, DAMON, or MGLRU's
  page-table walks. All of these need root. A residency policy driven by
  black-box traces therefore **learns its statistics from misses only**, and
  it is blind to how often pinned (always-hit) experts are used. The fix is a
  modelled observation regime plus estimators that tolerate censoring (see
  ARCHITECTURE §5). The simulator measures how much accuracy this costs
  (`--observe miss-only`).
- ✅ Measured here on kernel 6.19, with no root:
  - `mincore()` on my own mapping of a file reports page-cache residency even
    for root-owned files I can only read (libc, bash). This is the
    no-root fallback.
  - `cachestat()` (syscall 451, 6.5+) returns `EPERM` on files the caller
    doesn't own or can't write. It works on my own files, where it reports
    `nr_cache` and `nr_recently_evicted` per range. Usable when the user owns
    the model file, which is the usual case for downloaded models.
  - Polling `mincore` over the whole 18.6 GB file (4.5 M pages) is far too
    slow to run once per layer (≈1 ms per layer). So the poller watches
    **sentinel pages**: one page in the middle of each expert slice, at least
    128 KB from the slice edges so neighbouring read-around doesn't trigger
    false positives. That is about 6,144 single-page probes per sweep.

### 1.4 fadvise / madvise / mlock semantics — ⚠️

- ❌ `posix_fadvise(POSIX_FADV_DONTNEED)` **does not evict pages that another
  process has mapped.** Measured here on btrfs: pages mapped by a process
  stayed 100 % resident after DONTNEED, and were fully dropped once
  unmapped. The kernel's `mapping_evict_folio` skips mapped folios. Since
  llama.cpp maps every weight page it touches, "release cold experts with
  DONTNEED" does not work.
- ❌ `madvise(MADV_DONTNEED)` and `MADV_PAGEOUT` on the daemon's own mapping
  only affect the daemon's page tables. `MADV_PAGEOUT` also skips shared
  folios, and for file pages it requires file ownership or write access.
- ✅ The working demotion primitive is `process_madvise(pidfd, …, MADV_COLD
  or MADV_PAGEOUT)` on **llama.cpp's** address range. It has been available
  since 5.10 and requires `CAP_SYS_NICE` plus a ptrace-read check (man
  page). Without that capability the daemon can only pin and prefetch, and
  leaves demotion to the kernel.
- ✅ `posix_fadvise(WILLNEED)` and `readahead(2)` on the file populate the
  **shared** page cache and need no privilege. llama.cpp then takes a minor
  fault instead of a major one.
- ✅ `mlock` of a page through the daemon's own mapping makes the shared
  folio unevictable for everyone. ⚠️ `RLIMIT_MEMLOCK` here is **8 MB**, so a
  multi-GB pin budget needs `CAP_IPC_LOCK` or a raised limit.
- ⚠️ cgroup v2 charges a page-cache page to the cgroup of the task that
  **first** brings it in. Pages the daemon prefetches are charged to the
  daemon's cgroup, not llama.cpp's `memory.max`. For fair benchmarks the
  daemon must run **inside the same cgroup** as llama.cpp. This also offers
  an alternative pin mechanism without `CAP_IPC_LOCK`: `memory.low` on a
  delegated child cgroup holding daemon-charged pages. 🔬 Untested.
- ✅ cgroup v2 memory delegation works for this user without root:
  `user@1000.service` delegates `cpu io memory pids`, so
  `systemd-run --user -p MemoryMax=…` is available for experiments.

## 2. Is the "no app changes, OS-level" angle unoccupied?

Mostly yes, but the in-app side is crowded and moving fast. Details are in
RELATED_WORK.md. Summary:

- **In llama.cpp (all in-app, none merged as of 2026-10):**
  - PR #25294: O_DIRECT expert streaming with a slot cache.
  - PR #27861: LRU expert cache.
  - PR #26824: expert heatmap plus mmap pinning.
  - PR #29887: GPU cache for host-resident experts.
  - Issue #20757 and discussion #27149.
  - Forks: llama-cpp-expert-sniper (eval-callback `MADV_WILLNEED`),
    routhjim/llama.cpp-lab (mlock'd expert cache),
    jiro-prog/llamacpp-moe-expert-cache (Windows).
- **Closest to this project:** Si et al., *"Who Should Own the Expert Cache?
  Kernel-Managed Tiering for Trillion-Parameter MoE Inference"*
  (arXiv 2608.12103, Aug 2026). It studies the **page cache as the expert
  tier**, compares it against LRU, LFU, static frequency and Belady on real
  router traces (including Qwen3-30B-A3B traces captured from llama.cpp), and
  tests `fadvise(WILLNEED)` router-lookahead prefetch. Its findings bound our
  expectations:
  - Kernel recency ≈ an oracle static-frequency pin at equal memory.
  - Belady's bound says about ⅓ of LRU's remaining misses are addressable.
  - fadvise lookahead with 64.7 % recall changed median iteration time by
    only 0.3 %. Even perfect one-layer advice gained only 5 %.
  - MGLRU under balloon-style (global) pressure inflated device traffic about
    2× compared with a cgroup wall.

  Its setting is GH200, 27 GB/s NVMe, pread-based, no GGUF awareness, no
  black-box operation, and no consumer laptops.
- **Generic tools:** vmtouch (lock or evict file ranges), preload,
  ureadahead, cache_ext/cachebpf (custom eviction via eBPF, needs a patched
  kernel), DAMON/DAMOS. None of them know the model structure.

So nobody I found does **black-box, GGUF-structure-aware page-cache
management for an unmodified engine on a consumer Linux machine**. The paper
above is strong evidence, though, that *prediction-based prefetch* and
*fancy residency* each win less than intuition suggests.

## 3. Arguing against the idea

1. **"MGLRU plus existing flags already get most of it."** Plausible for
   residency. LRU over experts is not pathological: per-layer popularity skew
   and token-to-token reuse give it real hits. The loop pathology (LRU at 0 %
   on a cyclic scan) applies to the non-expert weights, about 1 GB, which fit
   in RAM anyway. Counterpoints:
   - The paper's data shows MGLRU misbehaving under *global* pressure, which
     is exactly the laptop co-tenant situation.
   - At very small budgets LRU collapses (0.5 % hits versus 12.7 % for LFU at
     8 slots with k=16).

   Both are testable, and "turn MGLRU off" (`lru_gen/enabled=n`) is a
   baseline.
2. **"Prefetch won't help."** The paper's evidence agrees for *prediction*
   prefetch. The window is also tiny here: about 0.8 ms of compute per layer
   × ~3 GB/s ≈ 2.4 MB, less than one Qwen3 expert. **Reframe:** the useful
   prefetch is not prediction. It is **expert completion**: once any page of
   (layer l, expert e) faults, immediately issue one large `WILLNEED` for all
   three slices of that expert. That turns about 23 small synchronous
   read-arounds into large requests at higher queue depth. It changes
   *bandwidth*, not *hit rate*, which is a lever the paper didn't test (it
   used pread of whole experts, which is already bulk).
3. **"The repack default makes the whole page-cache story moot."** Partly
   true (§1.2). We have to show that `--no-repack` plus the daemon beats
   default repack plus swap/zram. If it doesn't, the project dies (K3).
4. **"Upstream will ship an in-app expert cache and obsolete this."** Likely
   within months for llama.cpp. The OS-level tool remains relevant for
   unmodified engines and forks (ollama, LM Studio, other GGUF/mmap
   runtimes), and as a reference and measurement tool. Accepted risk, tracked
   as K4.
5. **"Miss-only observation makes the online policy too blind."** Real
   concern (§1.3). Measurable in simulation before any real-hardware work.

## 4. Back-of-envelope (Qwen3-30B-A3B Q4_K_M on this laptop)

Inputs:
- From the real header: file 18.56 GB, of which expert units are 17.55 GB.
  An expert unit (one layer, one expert, up+gate+down) is 2.65–3.06 MB,
  ~2.86 MB on average. One token touches 48 × 8 = 384 units, about
  **1.10 GB if all cold**.
- Compute: CPU decode is RAM-bandwidth bound. About 1.6–1.8 GB of weights
  are read per token (experts plus ~0.5 GB of attention and output). At
  ~45 GB/s of DDR bandwidth that is ≈ 40 ms per token (≈ 25 tok/s ceiling),
  or ≈ 0.8 ms per layer. 🔬 To measure.
- SSD: Solidigm P41 Plus (PCIe 4, QLC), rated ~4 GB/s sequential.
  🔬 Effective rate under fault-driven read-around unmeasured. I assume
  1 GB/s (fault-driven) versus 3 GB/s (bulk).
- Budget: 15 GB RAM with ~8–11 GB used by the desktop gives **C ≈ 4–7 GB**
  for the model. That is ~1 GB of non-expert weights plus 3–6 GB of experts,
  i.e. **16–34 % of expert units resident**.

Stall per token = 384 · m · 2.86 MB / B_eff, where m is the miss rate per
unit access:

| miss rate m | B_eff 1 GB/s | B_eff 3 GB/s |
|---|---|---|
| 0.50 | 0.55 s → 1.7 tok/s | 0.18 s → 4.5 tok/s |
| 0.35 | 0.38 s → 2.4 tok/s | 0.13 s → 6.0 tok/s |
| 0.25 | 0.27 s → 3.2 tok/s | 0.09 s → 7.7 tok/s |

(tok/s includes the 40 ms of compute.) Takeaways:

- At these budgets **I/O dominates by 3–14×**, so any byte not read is
  almost pure speedup.
- Two independent levers of roughly similar size:
  - **Miss rate** (residency policy): the Belady-bound evidence suggests
    ≤ ⅓ fewer misses at best, and realistically 10–20 %.
  - **Effective bandwidth** (expert-completion readahead): possibly 2–3×,
    🔬 to measure.
- One extra percentage point of miss rate costs ≈ 11 MB per token ≈ 4–11 ms
  per token. That is a large effect relative to 40 ms of compute.

So the idea matters **if** either the policy gap (oracle versus kernel) is
≥ 15 % or the bandwidth gap (fault-driven versus bulk) is ≥ 1.3×. Neither
number is known yet, and §6 is designed to measure both.

## 5. Alternatives considered (and why not pivot)

| direction | novelty | why not (now) |
|---|---|---|
| cache_ext/cachebpf MoE eviction policy in-kernel | medium | Needs a patched kernel that isn't upstream. Unusable for the target users. Kept as a "phase 6+ research" option if the userspace policy proves the gap. |
| GGUF re-layout (expert-major file) | low | ggml requires each 3-D tensor to be contiguous, so up/gate/down cannot be interleaved without engine changes. The "interleaving fix" in #27149 rests on a wrong premise. |
| Generic structure-aware prefetch daemon for any mmap file | low | vmtouch, preload and ureadahead already cover the generic case. The value is precisely the model structure. |
| In-app llama.cpp expert cache | none | 5+ competing PRs already exist. |
| Pure measurement study (MoE page-cache behaviour on consumer Linux) | medium | Becomes the fallback deliverable if the go/no-go fails. Phases 1–3 produce it anyway. |

The modified idea dominates because its first half (map, traces,
simulator, oracle, harness) is also exactly what the measurement study
needs. Nothing built now is wasted if the kill criteria fire.

## 6. Go/no-go experiment

**Setup.**
- Ground truth: record exact router decisions with a ~40-line eval-callback
  tool against libllama. This instruments the *experiment only*; the
  product stays black-box.
- Models: OLMoE-1B-7B (fits easily), Qwen3-30B-A3B Q4_K_M, gpt-oss-20b MXFP4.
- Workload: 3 prompt domains (chat, code, prose) × 512 decode tokens, greedy
  and T=0.7.

**Measurements.**
- **A, simulator.** Replay each trace at budgets C ∈ {15, 25, 35, 50, 70} % of
  expert bytes under LRU, LFU, static-frequency, V-residency (full and
  miss-only observation) and Belady. Report miss bytes per token.
- **B, real kernel, simulator validation.** Use `moepager replay` to touch a
  real mmap'd copy of the model in trace order under
  `systemd-run --user -p MemoryMax=C`. Record major faults and block-device
  read bytes, and compare with the simulator's LRU. Agreement within 20 %
  validates the simulator; otherwise investigate before trusting A.
- **C, bandwidth microbench (`moepager fault-io`).** Cold 2.9 MB expert
  units read by 6 threads faulting rows (llama.cpp's pattern), versus a
  single `WILLNEED` of all 3 slices, versus `pread` with 1 MB blocks.
  Report effective GB/s.
- **D, end-to-end.** llama.cpp `-nr` under `MemoryMax`, in two
  configurations: default (MGLRU on) and MGLRU off. Also run default
  llama.cpp (repack + zram/swap) and `--no-mmap` as baselines. Report
  tok/s, SSD bytes per token, and major faults per token.

**Kill criteria.**
- **K1, no policy gap:** Belady miss bytes ≥ 0.85 × kernel-measured miss
  bytes at C ∈ [15 %, 50 %] on ≥ 2 of 3 models. Prediction and residency
  then can't buy ≥ 15 %.
- **K2, no bandwidth gap:** fault-driven effective bandwidth ≥ 0.75 × bulk
  `WILLNEED` bandwidth (C). *Smoke result on L1: 0.41–0.52 vs 1.75–2.07
  GB/s, i.e. ≈ 0.25×, so K2 does not fire on this machine. The full run
  is pending.*
- **K1 and K2 both hold → kill the daemon.** Publish the measurement study
  instead.
- **Only K1** → shrink to an expert-completion readahead helper.
- **Only K2** → residency only.
- **K3, the baseline beats us:** default llama.cpp (repack + swap/zram)
  outperforms the best `-nr` + oracle simulated stall at equal memory →
  kill.
- **K4:** an upstream in-app expert cache is merged and beats the oracle
  bound for the page-cache approach. The project narrows to non-llama.cpp
  engines.

## 7. Changes from the original brief

1. The expert map groups 3 slices (+ bias rows) per (layer, expert). Shared
   boundary pages are handled explicitly.
2. Precondition `--no-repack` (or non-repacked types), with detection.
   Default llama.cpp is a baseline.
3. Observation is miss-only by design. Hits are unobservable without root.
   Estimators and the simulator model this explicitly.
4. Release via fadvise(DONTNEED) is replaced by pin (mlock) plus demote
   (process_madvise MADV_COLD), with graceful degradation without
   capabilities.
5. Prefetch is split into expert-completion readahead (bandwidth, no
   prediction) and cross-layer prediction (expected weak, kept behind a flag
   and evaluated in the simulator).
6. Baselines added: MGLRU off, default repack + swap, `--no-mmap`.
7. The daemon must run in the same cgroup as the engine for fair
   accounting.

## Sources

- llama.cpp PR #25294 — https://github.com/ggml-org/llama.cpp/pull/25294
- llama.cpp discussion #27149 — https://github.com/ggml-org/llama.cpp/discussions/27149
- llama.cpp issue #20757 — https://github.com/ggml-org/llama.cpp/issues/20757
- llama.cpp PR #26824 — https://github.com/ggml-org/llama.cpp/pull/26824
- llama.cpp PR #29887 — https://github.com/ggml-org/llama.cpp/pull/29887
- llama.cpp PR #29250 — https://github.com/ggml-org/llama.cpp/pull/29250
- llama.cpp PR #21067 — https://github.com/ggml-org/llama.cpp/pull/21067
- llama-cpp-expert-sniper — https://huggingface.co/waltgrace/llama-cpp-expert-sniper
- routhjim/llama.cpp-lab PR #9 — https://github.com/routhjim/llama.cpp-lab/pull/9
- jiro-prog/llamacpp-moe-expert-cache — https://github.com/jiro-prog/llamacpp-moe-expert-cache
- Si et al., arXiv 2608.12103 — https://arxiv.org/abs/2608.12103
- cachebpf / cache_ext, arXiv 2502.02750 — https://arxiv.org/abs/2502.02750
- ggml-cpu repack.cpp, ggml-cpu.c — https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cpu
- Linux include/trace/events/filemap.h — https://github.com/torvalds/linux/blob/master/include/trace/events/filemap.h
- process_madvise(2) — https://man7.org/linux/man-pages/man2/process_madvise.2.html
- GGUF headers: unsloth/Qwen3-30B-A3B-GGUF, ggml-org/gpt-oss-20b-GGUF, allenai/OLMoE-1B-7B-0924-GGUF on Hugging Face
