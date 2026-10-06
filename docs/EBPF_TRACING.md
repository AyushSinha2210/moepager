# eBPF Tracing Guide

Instructions for root/privileged tracing using `bpf/moepager.bt` and kernel tracepoints.

## Monitored Tracepoints
- `tracepoint:filemap:mm_filemap_add_to_page_cache`: Captures every insertion into the page cache, recording folio orders, page frame numbers, and device/inode identifiers.
- `tracepoint:filemap:mm_filemap_delete_from_page_cache`: Monitors folio evictions.

## Inode and Device Filtering
To avoid tracing unrelated system processes (e.g. desktop services, browsers), filter by device and inode number:
```bash
bpftrace bpf/moepager.bt <device_id> <inode_number> > out.bt
```
The output can be imported directly into `.mpt` format using `moepager ingest-bpftrace`.
