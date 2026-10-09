# Policy Analysis: Cache Replacement on MoE Workloads

A mathematical and empirical comparison of eviction and pinning policies evaluated by `moepager`.

## Evaluated Policies
- **LRU (Least Recently Used)**: Tracks temporal recency. Suffers under cyclic token loops when working set exceeds budget.
- **LFU (Least Frequently Used)**: Tracks access counts. Susceptible to cache pollution when topic domain shifts.
- **V(e) Residency**: Balances access frequency with unit byte size and fault penalty:
  $$V(e) = \frac{\text{rate}(e) \cdot (t_{\text{fault}} + s_e / B_{\text{demand}})}{s_e}$$
- **Bélady's MIN* Oracle**: Furthest-in-future next-use eviction. Serves as the theoretical upper bound for addressable misses.
