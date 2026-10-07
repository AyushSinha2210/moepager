# Metrics Dictionary & Counter Reference

A catalog of metrics reported by `moepager sim`, `moepager analyze`, and `moepagerd`.

## Core Counters
- `events`: Total observed miss or access records processed.
- `tokens`: Count of inferred or ground-truth decode token cycles.
- `prefetch_ops`: Bulk `WILLNEED` operations dispatched.
- `pin_ops`: Expert units locked into RAM via `mlock`.
- `unpin_ops`: Expert units released via `munlock`.
- `budget_clamps`: Incidents where `RLIMIT_MEMLOCK` forced budget capping.
