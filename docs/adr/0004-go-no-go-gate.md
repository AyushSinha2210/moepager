# ADR 0004 — Measure before building the live daemon

**Status:** accepted (2026-09-24)

**Context.** Prior work (arXiv 2608.12103) found that kernel recency is
close to an oracle static-frequency pin, and that WILLNEED lookahead gains
≤ 5 %. Our expected gains may be small.

**Decision.**
- Phases 1–6 build only tooling that is useful whether or not the idea
  survives: map, traces, analyzer, simulator, oracle, harness.
- The live daemon (phase 8+) starts only if experiments A–D clear kill
  criteria K1–K4 (IDEA_REVIEW §6).

**Consequences.** If the gate fails, the deliverable is a measurement study
of MoE page-cache behaviour on consumer Linux.
