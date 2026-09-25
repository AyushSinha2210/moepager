# moepager trace format (`.mpt`), version 1

A trace is one binary file: a fixed preamble, a JSON header, then
fixed-size little-endian records.

```
offset  size  field
0       8     magic        "MPTRACE\0"
8       2     version      u16 = 1
10      2     record_kind  u16: 1 = expert, 2 = page
12      4     header_len   u32, length of the JSON header in bytes
16      N     header       UTF-8 JSON object (below)
16+N    …     records      record_size × count; a trailing partial record is an error
```

## Expert record (`record_kind = 1`, 16 bytes)

| offset | type | field | meaning |
|---|---|---|---|
| 0 | u64 | `t_ns` | nanoseconds since trace start |
| 8 | u32 | `token` | decode step index (exact in synthetic or ground-truth traces, inferred in black-box ones) |
| 12 | u16 | `layer` | block index |
| 14 | u16 | `expert` | expert index within the layer |

One record means "unit (layer, expert) was used (observation=full) or
missed (observation=miss-only) at t_ns". Records are ordered by
`(t_ns, token, layer)` non-decreasing. Duplicate (token, layer, expert)
records are allowed in black-box traces, and readers deduplicate per token.

## Page record (`record_kind = 2`, 24 bytes)

| offset | type | field | meaning |
|---|---|---|---|
| 0 | u64 | `t_ns` | nanoseconds since trace start |
| 8 | u64 | `page` | file page index (offset / page_size) |
| 16 | u32 | `n_pages` | pages covered (2^order for folio events, run length for scans) |
| 20 | u8 | `kind` | 1 = insert (entered page cache), 2 = evict (left page cache), 3 = fault |
| 21 | u8×3 | reserved | must be 0 |

## JSON header

```json
{
  "source": "synth | mincore-scan | sentinel | bpftrace | llama-evalcb | replay",
  "observation": "full | miss-only",
  "n_layers": 48,
  "n_experts": 128,
  "top_k": 8,
  "page_size": 4096,
  "model_file": "Qwen3-30B-A3B-Q4_K_M.gguf",
  "model_size": 18556686912,
  "params": { "...": "source-specific, e.g. generator parameters" }
}
```

Unknown fields must be ignored by readers. `top_k`, `model_file` and
`model_size` may be `null`. A breaking change bumps `version`. Readers
reject versions they don't know.

## CSV export

`moepager trace-csv <file>` writes `t_ns,token,layer,expert` or
`t_ns,page,n_pages,kind`, for Python/pandas.
