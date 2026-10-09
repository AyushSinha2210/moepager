# Benchmarking Reproduction Guide

Step-by-step instructions to reproduce `moepager` benchmarks and microbenchmarks.

## Prerequisites
- Linux with kernel >= 5.10 (tested on Fedora 42 with kernel 6.19).
- Cgroup v2 delegation enabled for user session (`systemd-run --user`).
- NVMe drive with known readahead window (`cat /sys/block/<dev>/queue/read_ahead_kb`).

## Clean Environment Reset
Before each microbenchmark or replay run, drop cached model pages:
```bash
moepager fault-io --drop <path-to-model.gguf>
```
Wait 30 seconds for filesystem background threads and kernel writeback to stabilize.
