# ADR 0010 — Token Boundary Detection via Layer Regression

**Status:** accepted (2026-09-10)

**Context.** In black-box inference, the daemon does not receive token stepping notifications from `llama.cpp`.

**Decision.**
- Infer token boundaries when the observed layer index drops significantly below the running maximum layer (`cur_layer < max_layer - slack`).
- Presence of layer 0 expert activations acts as definitive evidence of a new token start.

**Consequences.**
- Eliminates any need to patch the inference engine or insert callbacks in production.
