# ADR 0001 — Rust core with policy code shared by simulator and daemon

**Status:** accepted (2026-09-24)

**Context.** We must judge policies offline (simulator, oracle) and then run
the same policy live. Re-implementing a policy in two languages invites
"the simulator said X but the daemon does Y" bugs.

**Decision.**
- Every decision procedure (statistics, residency, prefetch planning) lives
  in the pure, deterministic `mp-core` crate.
- `mp-sim` and `moepagerd` both call it.
- Rust is used for all of it; Python only for fixtures, parsing and plots.

**Consequences.**
- `mp-core` may not do I/O or read clocks. Time is passed in.
- The simulator is fast enough for sweeps.
- Python cannot be used to prototype new policies without a Rust port.
  Accepted.
