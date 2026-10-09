# ADR 0008 — Cgroup Colocation for Fair Accounting

**Status:** accepted (2026-09-08)

**Context.** Linux cgroup v2 charges page-cache folios to the memory cgroup of whichever process first faults or prefetches the page. If `moepagerd` runs outside the model's cgroup, prefetched pages are charged to the daemon's cgroup instead of the engine's memory budget.

**Decision.**
- Colocate `moepagerd` inside the exact same transient cgroup scope as `llama.cpp` using `systemd-run --user --scope`.

**Consequences.**
- Strict memory budget enforcement without leakage across cgroup boundaries.
