# ADR 0005 — 128 KiB guard distance for sentinel pages

**Status:** accepted (2026-10-02)

**Context.** In black-box, unprivileged operation, `moepager` relies on polling sentinel pages with `mincore` to detect when an expert unit is accessed. In GGUF files, 3-D expert tensors lay out expert slices sequentially, aligned to 32 bytes rather than page boundaries. When the CPU faults on a page in expert slice $e$, the Linux kernel performs synchronous read-around (typically 128 KiB on NVMe block devices). If a sentinel page for slice $e+1$ or $e-1$ sits within the read-around radius of slice $e$, it will be brought into the page cache as collateral I/O, generating a false positive.

**Decision.**
- Position the sentinel page at the geometric center of each expert slice.
- Enforce a minimum guard distance (`SENTINEL_GUARD = 128 * 1024` bytes) from both the starting and ending offsets of the slice whenever the slice length permits (`len >= 2 * SENTINEL_GUARD`).
- For smaller slices, fallback to the exact slice midpoint.

**Consequences.**
- Sentinel page hits reliably indicate intentional execution access of that specific expert, eliminating false-positive activations caused by neighboring read-around.
- Minor overhead during header mapping to compute guarded offsets.
- Verified on real Qwen3-30B-A3B and OLMoE tensor layouts.
