# Daemon Configuration and Tuning Guide

This document describes how to configure and tune `moepagerd` for various host memory sizes, storage characteristics, and workload domains.

---

## Core Configuration Parameters

### 1. Pin Budget (`--pin-budget-bytes`)
- **Default**: `0` (prefetch-only mode).
- **Recommended**: Set to `Available_RAM - Non_Expert_Weights - Co_Tenant_Reserve`.
- **Guidelines**:
  - Non-expert model weights (attention, norms, routers, output embeddings) take approximately 0.5–1.2 GB and are accessed on every single token.
  - Leave at least 3–4 GB of headroom for desktop environments, browser tabs, and terminal tasks.
  - For an 18.5 GB model on a 16 GB laptop with 10 GB free RAM, a pin budget of `4G` to `6G` (25–35% of expert bytes) is optimal.

### 2. Re-plan Frequency (`--replan-every`)
- **Default**: `4` tokens.
- **Tuning**:
  - Lower values (1–2): Rapidly adapts to domain shifts (e.g. switching from conversational text to Python code generation). Slightly increases userspace CPU overhead.
  - Higher values (8–16): Minimizes pin/unpin churn and system call overhead for stable single-topic generations.

### 3. Hysteresis (`--hysteresis`)
- **Default**: `1.25` (a candidate must have 25% higher $V(e)$ value than the lowest pinned expert to replace it).
- **Tuning**:
  - Prevents ping-pong eviction churn where experts oscillate in and out of the pinned set between adjacent tokens.
  - If observing high `churn` counters in `--stats`, increase to `1.4`–`1.5`.

### 4. Exploration Fraction (`--explore-frac`)
- **Default**: `0.02` (2% of pinned slots per re-plan).
- **Tuning**:
  - Essential in the **miss-only** observation regime: once an expert is pinned, hits become invisible to `mincore` or eBPF miss tracepoints.
  - Unpinning a random small fraction periodically allows the statistics engine to verify whether the expert is still actively hot.
  - Increase to `0.05` for highly dynamic multi-turn chat sessions.

### 5. Prefetch Modes (`--prefetch`)
- `completion` (Default recommended): Once any slice of an expert unit faults, immediately issues 128 KiB chunked `WILLNEED` calls for all 3 slices. Increases effective I/O bandwidth from ~0.45 GB/s to ~2.4 GB/s.
- `predict`: Uses cross-layer transition matrices to speculate on upcoming layers. Recommended only when NVMe bandwidth is high (>3 GB/s) and compute times provide adequate lookahead slack.

---

## Example Deployment Profiles

### Profile A: Low-RAM Laptop (16 GB Total RAM, 18.5 GB Model)
```bash
moepagerd \
  ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --policy v \
  --pin-budget-bytes 5368709120 \
  --replan-every 4 \
  --hysteresis 1.30 \
  --prefetch completion
```

### Profile B: Workstation (32 GB Total RAM, 60% Residency)
```bash
moepagerd \
  ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --policy v \
  --pin-budget-bytes 11811160064 \
  --replan-every 8 \
  --hysteresis 1.20 \
  --prefetch completion,predict
```

---

### Profile C: Cloud VM with High-IOPS NVMe (e.g. AWS i3en / GCP c3-highmem)
- When backing storage provides $>4\text{ GB/s}$ sustained sequential I/O:
  - Set --prefetch completion,predict
  - Reduce --replan-every to 2 tokens to capture bursty routing patterns
  - Set --pin-budget-bytes to 50% of expert bytes
