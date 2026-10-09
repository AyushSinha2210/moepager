# ADR 0007 — Mandatory --no-repack Flag for llama.cpp

**Status:** accepted (2026-09-07)

**Context.** On x86 hosts with AVX2, default `llama.cpp` repacks quantized weights (Q4_K, MXFP4) into anonymous memory (`CPU_REPACK`) to optimize SIMD operations. This copies ~77% of expert bytes out of the file mapping into anonymous RAM, completely bypassing the OS page cache.

**Decision.**
- Require `llama.cpp` to run with `--no-repack` (`-nr`).
- Inspect GGUF tensor quantization types and warn if repackable types are present.

**Consequences.**
- Keeps expert weights file-backed in page cache where `moepager` can manage them.
