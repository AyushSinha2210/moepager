# moepager trace format (`.mpt`), version 1

A trace is one binary file: a fixed preamble, a JSON header, then
fixed-size little-endian records.

```
+--------------------------------------------------------------+
| Magic (8B): "MPTRACE\0"                                      |
+------------------------------+-------------------------------+
| Version (u16): 1             | Record Kind (u16): 1 or 2     |
+------------------------------+-------------------------------+
| Header Length (u32, little-endian)                           |
+--------------------------------------------------------------+
| JSON Header (UTF-8 bytes of length `header_len`)            |
+--------------------------------------------------------------+
| Contiguous Stream of Fixed-Size Records                      |
| (16B each if record_kind = 1, 24B each if record_kind = 2)   |
+--------------------------------------------------------------+
```

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

```
0                   4                   8                   12                  16
+-------------------+-------------------+-------------------+---------+---------+
|                  t_ns (u64)                   |    token (u32)    |layer(u16)|exp(u16) |
+-----------------------------------------------+-------------------+---------+---------+
```

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

```
0                   8                   16        20   21      24
+-------------------+-------------------+---------+----+--------+
|     t_ns (u64)    |     page (u64)    |n_pg(u32)|kind|reserved|
+-------------------+-------------------+---------+----+--------+
```

| offset | type | field | meaning |
|---|---|---|---|
| 0 | u64 | `t_ns` | nanoseconds since trace start |
| 8 | u64 | `page` | file page index (offset / page_size) |
| 16 | u32 | `n_pages` | pages covered (2^order for folio events, run length for scans) |
| 20 | u8 | `kind` | 1 = insert (entered page cache), 2 = evict (left page cache), 3 = fault |
| 21 | u8×3 | reserved | must be 0 |

### Page Event Kinds
- `1` (`PAGE_INSERT`): Page added to the page cache (cache miss resolution or readahead).
- `2` (`PAGE_EVICT`): Page removed from the page cache by kernel reclaim or DONTNEED.
- `3` (`PAGE_FAULT`): Page fault event triggered by user-space access.

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

## Token Boundary Inference

For passive traces (`sentinel`, `bpftrace`, `mincore-scan`) where token boundaries are not tagged by the inference engine:
- A new token boundary is detected when `layer` drops significantly below the previously observed maximum layer (e.g. `prev_layer - cur_layer > token_slack`).
- Accesses at layer 0 strongly indicate the start of a subsequent token step.

## Conversion and Tools

- Exporting to CSV:
  ```bash
  moepager trace-csv trace.mpt -o trace.csv
  ```
- Converting page trace to expert trace:
  ```bash
  moepager page2expert page_trace.mpt --map map.json -o expert_trace.mpt
  ```


---

## Python Trace Parser Example
Read binary .mpt traces directly in Python without dependencies:
```python
import struct, json

def read_mpt(path):
    with open(path, "rb") as f:
        magic, version, kind, hlen = struct.unpack("<8sHHI", f.read(16))
        header = json.loads(f.read(hlen).decode("utf-8"))
        while chunk := f.read(16 if kind == 1 else 24):
            if kind == 1:
                t_ns, token, layer, expert = struct.unpack("<QIH H", chunk)
                yield {"t_ns": t_ns, "token": token, "layer": layer, "expert": expert}
```

