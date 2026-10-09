# Glossary of Terms and Concepts

A reference guide for technical terminology used throughout the `moepager` project, covering Mixture-of-Experts architecture, Linux memory management, and caching concepts.

---

### Mixture-of-Experts (MoE) Concepts

- **MoE (Mixture of Experts)**: A neural network architecture where sparse sub-networks ("experts") are conditionally activated per token rather than activating the entire parameter set.
- **Unit `(layer, expert)`**: The atomic management and pinning granule in `moepager`. Represents everything one expert at one layer needs during token decode: its slice of each 3-D expert tensor (`ffn_up_exps`, `ffn_gate_exps`, `ffn_down_exps`) plus rows of any 2-D per-expert tensors (`*_exps.bias`).
- **Slice**: A contiguous byte range inside a specific tensor belonging to an expert. In GGUF files, up, gate, and down slices for an expert are separated by hundreds of megabytes.
- **Top-K Routing**: The routing mechanism where the router selects the top $k$ highest-scoring experts for each token (e.g., $k=8$ out of 128 experts for Qwen3-30B-A3B).
- **Repacking**: A process in `llama.cpp` on x86 AVX2 where quantized weights (Q4_K, MXFP4) are rearranged into anonymous memory at load time for SIMD speed. Because repacked weights live in anonymous memory, they bypass the OS page cache and go to swap/zRAM under pressure.

---

### Linux Virtual Memory & Page Cache

- **Page Cache**: The Linux kernel cache storing disk blocks in physical memory to speed up file access. Unmodified `llama.cpp` accesses weights through a shared `mmap` backing this cache.
- **Folio**: A contiguous set of one or more physical pages managed as a single unit in the Linux memory subsystem (kernel 5.16+).
- **Read-around / Readahead**: When a process faults on a missing page, the kernel reads surrounding pages into the page cache based on the device's read-ahead window (often 128 KiB on NVMe block devices, or up to 4 MiB on filesystems like Btrfs).
- **Read Amplification**: The ratio of bytes physically read from storage to bytes actually needed. High read amplification occurs when the kernel reads unneeded neighboring tensor data during a fault.
- **MGLRU (Multi-Gen LRU)**: The modern Linux page reclamation policy that organizes folios into generations based on access recency and frequency.
- **`mincore(2)`**: A Linux system call that tests whether pages of a memory mapping are currently resident in RAM.
- **`cachestat(2)`**: A Linux system call (kernel 6.5+) returning cache statistics (resident pages, recently evicted pages) for a specified file range.
- **`process_madvise(2)`**: A syscall (kernel 5.10+) allowing an external process (with `CAP_SYS_NICE`) to issue memory advice (such as `MADV_COLD` or `MADV_PAGEOUT`) to another process's address space.

---

### `moepager` Specific Mechanisms

- **Sentinel Page**: The page chosen at the center of an expert slice (guarded by $\ge 128\text{ KiB}$ margins). Polled via `mincore()` to detect expert activation without observing individual page touches or requiring root privileges.
- **Expert-Completion Readahead**: The mechanism where observing a miss on any slice of an expert unit triggers an immediate bulk `WILLNEED` for the remaining slices of that expert, replacing slow read-around faults with high-speed sequential I/O.
- **WILLNEED Chunking**: Splitting `posix_fadvise(WILLNEED)` calls into 128 KiB chunks to avoid silent kernel truncation to the backing device readahead limit.
- **Miss-Only Observation**: The operational reality where page hits through `mmap` are handled in CPU page tables and are invisible to userspace. Statistics are inferred purely from observed page-cache insertions/misses.
- **$V(e)$ Value Metric**: The estimated value per byte of keeping expert unit $e$ pinned in memory, balancing miss probability against unit size and I/O fault latency:
  $$V(e) = \frac{\text{rate}(e) \cdot (t_{\text{fault}} + s_e / B_{\text{demand}})}{s_e}$$
- **Bélady's MIN\* Oracle**: The theoretical upper bound cache replacement policy that replaces the item whose next reference occurs furthest in the future, modified for variable-sized expert units.

---

### Mathematical Formulations

- **EWMA Access Rate ($\text{rate}_t(e)$)**:
  \text{rate}_t(e) = \alpha \cdot \mathbf{1}_{\{e \in U_t\}} + (1 - \alpha) \cdot \text{rate}_{t-1}(e)
  where $\alpha = 1 - 2^{-1 / H}$ with half-life $ tokens.
- **Value Metric (e)$**:
  V(e) = \frac{\text{rate}(e) \cdot (t_{\text{fault}} + s_e / B_{\text{demand}})}{s_e}
