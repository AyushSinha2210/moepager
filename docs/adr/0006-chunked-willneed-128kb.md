# ADR 0006 — Chunked 128 KiB WILLNEED Readahead

**Status:** accepted (2026-09-06)

**Context.** Linux `posix_fadvise(POSIX_FADV_WILLNEED)` internally invokes `force_page_cache_ra()`, which silently caps the readahead length to `max(bdi->io_pages, ra->ra_pages)`. On standard NVMe drives with 128 KiB readahead, issuing `WILLNEED` for an entire 1 MB or 4 MB slice only populates the first 128 KiB.

**Decision.**
- Slice all `WILLNEED` requests into contiguous 128 KiB chunks.
- Loop over the entire byte extent of each slice in 128 KiB increments.

**Consequences.**
- Avoids silent kernel truncation and guarantees the full expert slice is loaded into page cache.
