# Real-model expert maps

Produced by `moepager gguf-map` from the first 8–16 MB (header only, fetched
with HTTP range requests) of these published files. They contain tensor
offsets and sizes only, no weights.

| file | source | file size |
|---|---|---|
| qwen3-30b-a3b-q4_k_m.map.json | huggingface.co/unsloth/Qwen3-30B-A3B-GGUF `Qwen3-30B-A3B-Q4_K_M.gguf` | 18,556,686,912 |
| gpt-oss-20b-mxfp4.map.json | huggingface.co/ggml-org/gpt-oss-20b-GGUF `gpt-oss-20b-MXFP4.gguf` | 12,109,566,624 |
| olmoe-1b-7b-0924-q4_k_m.map.json | huggingface.co/allenai/OLMoE-1B-7B-0924-GGUF `olmoe-1b-7b-0924-q4_k_m.gguf` | 4,213,511,776 |

Regenerate:

```
curl -sL -r 0-16777215 -o qwen3.head <url>
moepager gguf-map qwen3.head --file-size 18556686912 -o qwen3-30b-a3b-q4_k_m.map.json
```

They are used for realistic unit sizes in simulator demos and tests. Routing
behaviour in those demos is still synthetic.
