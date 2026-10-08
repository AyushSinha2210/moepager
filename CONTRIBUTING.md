# Contributing

## Setup
- Rust stable (see `rust-toolchain.toml`), Python ≥ 3.10 with `pytest`.
- `make test` runs everything that can run without privileges.
- `make lint` runs `cargo fmt --check`, `cargo clippy -D warnings` and `ruff`.

## Commits & Development Workflow
- Follow Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`,
  `refactor:`, `build:`, `ci:`), one logical change per commit.
- The tree must build and `make test` must pass at each commit.
- Update `PHASES.md` in the same commit as the work it describes (tick
  tasks, update "Current status", "Next up" and "Known issues").

## Pull Request Checklist
Before submitting a pull request, ensure the following criteria are satisfied:
- [ ] Code formatting: `cargo fmt --all -- --check` produces no diffs.
- [ ] Linter checks: `cargo clippy --all-targets -- -D warnings` passes without warnings.
- [ ] Python linter: `ruff check python/` passes.
- [ ] Test coverage: `cargo test --all` and `pytest python/tests` pass.
- [ ] Determinism: Any changes to `mp-core` remain pure functions with zero clock reads or unseeded random state.
- [ ] Documentation: Any new syscalls or privilege requirements are documented in `docs/PRIVILEGES.md`.
- [ ] Architectural Decision Records: Architectural choices or contract changes are documented in `docs/adr/`.

## Rules of the Road
- **Never report a performance number you didn't measure.** BENCHMARKS.md
  cells stay TBD until raw outputs exist.
- Anything that touches the OS goes behind the `mp-os` traits. The policy
  code in `mp-core` must stay pure and deterministic, because the simulator
  depends on it.
- New observation or actuation mechanisms need an entry in
  `docs/PRIVILEGES.md`.
- Code that can't be tested here (root, eBPF, real engine) is marked
  `// UNTESTED-ON-HW:` and listed in PHASES.md "Known issues".


---

## Pull Request Review Standards
- Every PR must verify that synthetic trace simulator runs produce deterministic results.
- Any new actuator operation added to mp-os must provide a corresponding MockOps test fixture.

