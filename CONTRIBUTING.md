# Contributing

## Setup
- Rust stable (see `rust-toolchain.toml`), Python ≥ 3.10 with `pytest`.
- `make test` runs everything that can run without privileges.
- `make lint` runs `cargo fmt --check`, `cargo clippy -D warnings` and `ruff`.

## Commits
- Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`,
  `refactor:`, `build:`, `ci:`), one logical change each.
- The tree must build and `make test` must pass at each commit wherever
  practical.
- Update `PHASES.md` in the same commit as the work it describes (tick
  tasks, update "Current status", "Next up" and "Known issues").

## Rules of the road
- **Never report a performance number you didn't measure.** BENCHMARKS.md
  cells stay TBD until raw outputs exist.
- Anything that touches the OS goes behind the `mp-os` traits. The policy
  code in `mp-core` must stay pure and deterministic, because the simulator
  depends on it.
- New observation or actuation mechanisms need an entry in
  docs/PRIVILEGES.md.
- Code that can't be tested here (root, eBPF, real engine) is marked
  `// UNTESTED-ON-HW:` and listed in PHASES.md "Known issues".
