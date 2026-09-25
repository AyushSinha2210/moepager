# ADR 0002 — Treat black-box observation as miss-only

**Status:** accepted (2026-09-24)

**Context.** Hits on mmap'd weight pages don't enter the kernel, so neither
tracepoints nor mincore/cachestat see them. Seeing them needs root
(page_idle, DAMON).

**Decision.**
- Model two observation regimes, `full` and `miss-only`.
- Estimate a unit's value from its **miss rate while unpinned**. This is the
  counterfactual the pin decision needs.
- Freeze estimates of pinned units, and refresh them with ε-exploration.
- The simulator reports both regimes so the cost of blindness is measured,
  not guessed.

**Consequences.**
- Transition statistics learned from misses are biased toward cold
  experts. Prediction prefetch may suffer. It is behind a flag and must earn
  its place in the simulator.
