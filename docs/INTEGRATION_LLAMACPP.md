# llama.cpp Integration Guide

Optimal command-line arguments and configuration settings when running `llama.cpp` alongside `moepagerd`.

## Mandatory Engine Arguments
- `--no-repack` (or `-nr`): Disables anonymous CPU weight repacking so quantized MoE weights remain in page cache.
- `--mmap`: Enables memory mapping of the model file (default behavior).
- `-t <N>`: Set thread count to the physical core count to prevent thread oversubscription during fault stalls.

## Launch Order
1. Start `moepagerd` specifying the GGUF model path.
2. In the same cgroup scope, launch `llama-cli` or `llama-server`.
