# Frequently Asked Questions (FAQ)

### General

#### What is the core goal of `moepager`?
`moepager` provides expert-aware page-cache management for Mixture-of-Experts (MoE) LLMs in `llama.cpp`. By analyzing the GGUF header and monitoring page-cache misses from user space, it pins hot experts and issues bulk readahead for missing ones without modifying the inference engine or copying model weights.

#### Does `moepager` alter model weights or inference accuracy?
No. `moepager` is purely an I/O and memory caching layer. It never mutates tensor weights, never quantizes data, and does not alter the mathematical output of `llama.cpp`.

---

### Technical & Operating System

#### Why is `--no-repack` (`-nr`) required when running `llama.cpp`?
On x86 processors with AVX2 support, `llama.cpp` repacks quantized weights (such as Q4_K and MXFP4) at startup into separate anonymous memory buffers (`CPU_REPACK`) to accelerate matrix multiplication. This moves ~77% of expert bytes out of the file mapping into anonymous memory, where they are governed by swap/zRAM rather than the file page cache. Using `--no-repack` ensures all expert weights remain backed by the mmap file.

#### Why doesn't `moepagerd` need to run as root?
`moepager` is built on the principle of least privilege:
- `posix_fadvise(WILLNEED)` and `mincore()` require standard read permissions on the model file.
- Pinning beyond user memory limits requires only `CAP_IPC_LOCK`.
- Cross-process demotion (`process_madvise(MADV_COLD)`) requires only `CAP_SYS_NICE`.
These capabilities can be granted with `sudo setcap cap_ipc_lock,cap_sys_nice+ep target/release/moepagerd` or via systemd `AmbientCapabilities`.

#### Why not use `vmtouch` instead?
`vmtouch` is a generic tool that pins contiguous ranges of a file. It has no understanding of:
- GGUF tensor layouts: an expert at layer $l$ consists of 3 distinct slices separated by hundreds of megabytes.
- MoE routing patterns: token-to-token reuse distance, cross-layer affinity, or popularity skew.
`moepager` groups these separated slices into atomic units and dynamically manages residency according to runtime routing statistics.

#### Can `moepager` run on Windows or macOS?
No. `moepager` relies on Linux-specific virtual memory interfaces:
- `mincore` residency bitmasks
- `posix_fadvise(POSIX_FADV_WILLNEED)` 128 KiB chunked readahead
- `process_madvise(MADV_COLD)`
- `cgroup v2` memory controllers and Pressure Stall Information (PSI)

#### Can `moepager` be run inside Docker containers?
Yes, provided the container is granted the required capabilities and has access to the host model file:
```bash
docker run --rm \
  --cap-add=IPC_LOCK \
  --cap-add=SYS_NICE \
  -v /path/to/models:/models:ro \
  moepager:latest /models/model.gguf --policy v
```

#### How does context window length impact moepager?
During initial prompt evaluation (prefill), all prompt tokens are processed in batch mode, generating high burst cache misses across all layers. During token generation (decode), expert activation follows steady per-token top-$ sparsity, where (e)$ residency stabilization achieves maximal speedup.
