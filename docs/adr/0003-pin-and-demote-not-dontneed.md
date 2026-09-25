# ADR 0003 — Residency via pin + demote, not fadvise(DONTNEED)

**Status:** accepted (2026-09-24)

**Context.** `posix_fadvise(DONTNEED)` skips folios mapped by any process.
Verified here: 100 % of the pages stayed resident while mapped. llama.cpp
maps every weight page it touches.

**Decision.**
- Keep high-value units with `mlock` through the daemon's own shared
  mapping.
- Demote low-value units with `process_madvise(MADV_COLD)` on the engine's
  mapping when `CAP_SYS_NICE` is available. Otherwise leave eviction to the
  kernel.

**Consequences.**
- Pinning needs `CAP_IPC_LOCK` or a raised `RLIMIT_MEMLOCK`.
- Demotion needs the engine's pid and mapping address, found from
  /proc/pid/maps by inode.
