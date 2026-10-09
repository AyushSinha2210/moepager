# ADR 0009 — Two-Tier Observation Architecture

**Status:** accepted (2026-09-09)

**Context.** User environments vary from root-restricted unprivileged desktops to dedicated benchmark setups with root capabilities.

**Decision.**
- Support a two-tier observation backend:
  1. **Tier 1 (Unprivileged)**: Sentinel-page polling via `mincore()`.
  2. **Tier 2 (Privileged)**: eBPF tracepoints (`filemap:mm_filemap_add_to_page_cache`).

**Consequences.**
- Zero mandatory root requirements for regular end users.
