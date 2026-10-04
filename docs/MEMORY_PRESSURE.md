# Memory Pressure and Kernel Reclamation

Understanding how Linux reclaims memory and how `moepager` interacts with kernel resource management.

## Pressure Stall Information (PSI)
Linux Pressure Stall Information monitors system-level resource starvation in `/proc/pressure/memory`:
- **`some`**: Percentage of time at least some tasks were stalled on memory allocation.
- **`full`**: Percentage of time all non-idle tasks were frozen waiting for memory reclaim.

## Cgroup v2 Boundaries
`moepagerd` is designed to run within the same cgroup scope as `llama.cpp` using `systemd-run --user -p MemoryMax=<Budget>`. This ensures pages prefetched by the daemon are attributed correctly to the model's memory footprint.
