# moepager

**Expert-aware page-cache management for running Mixture-of-Experts LLMs on
low-RAM Linux machines, without modifying the inference engine.**

> Status: **research prototype, pre-go/no-go.** The offline toolchain (GGUF
> expert map, trace recorder, synthetic generator, analyzer, simulator with
> a Belady oracle, daemon skeleton, benchmark harness) is being built and
> tested. Nothing has been benchmarked against a real model yet, and no
> speedup is claimed. See [PHASES.md](PHASES.md) for exact status and
> [IDEA_REVIEW.md](IDEA_REVIEW.md) for why the project is shaped the way it
> is.

## The idea in one paragraph

llama.cpp running a model larger than RAM via mmap leaves expert weights to
the kernel's generic page-cache policy. moepager parses the GGUF header to
learn where every (layer, expert) unit lives in the file: three slices in
three tensors. It watches which units fault into the page cache (eBPF, or
`mincore` without root), learns routing statistics from those misses, and
then:
- pins high-value experts;
- completes partially-faulted experts with one large readahead;
- demotes cold ones.

Throughout, llama.cpp is a black box.
