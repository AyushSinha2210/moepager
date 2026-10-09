# Security Model & Threat Analysis

Security boundaries and guarantees upheld by the `moepager` architecture.

## Read-Only Model Access
`moepager` opens model files strictly with `O_RDONLY`. It never issues write system calls and cannot alter model weights or underlying storage.

## Memory Isolation
The daemon allocates its own read-only `MAP_SHARED` mapping. `process_madvise(MADV_COLD)` calls only advise eviction priority and can never overwrite or corrupt the target process memory space.
