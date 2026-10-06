# NVMe Storage Characteristics & MoE Flash Wear

Analysis of solid-state storage mechanics under page-cache fault loads.

## Read Amplification & Write Endurance
- **Read Amplification**: Random page faults trigger 4 KiB to 128 KiB read-around, wasting NVMe controller bus bandwidth on unreferenced weights.
- **Bulk WILLNEED Efficiency**: 128 KiB chunked sequential reads achieve drive-rated bus throughput (>2.5 GB/s) with near-zero read amplification.
- **Zero Drive Wear**: `moepager` only issues read requests (`fadvise` and `madvise`), generating zero NAND erase/write cycles.
