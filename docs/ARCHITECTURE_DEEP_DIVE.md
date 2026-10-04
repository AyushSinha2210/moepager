# Architecture Deep Dive: I/O Channel & Latency Models

This document describes the low-level queueing model implemented in `mp-sim` and monitored in `moepagerd`.

## FIFO Storage Channel Model
NVMe solid-state storage operates with asynchronous command submission across submission and completion queues. Under sequential access or large block sizes, drive controllers sustain high bandwidth. Under random read-around faults with low queue depths, throughput degrades drastically.

`mp-sim` models the NVMe channel as a single FIFO queue with:
- **Base Fault Latency ($t_{\text{fault}}$)**: Kernel trap, context switch, and flash address lookup overhead (~15–30 µs).
- **Transfer Latency**: Calculated by slice size divided by bandwidth: $s / B$.
- **Arrival & Stall**: A demand miss blocks inference until completion, while in-flight prefetches occupy queue depth.
