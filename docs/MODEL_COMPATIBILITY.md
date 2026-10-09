# Model Architecture Compatibility Matrix

Verified model architectures, tensor naming conventions, and compatibility status.

| Architecture | Example Model | Expert Slices | Bias Tensors | Status |
|---|---|---|---|---|
| Qwen2-MoE / Qwen3 | Qwen3-30B-A3B | up, gate, down | None | Verified |
| OLMoE | OLMoE-1B-7B-0924 | up, gate, down | None | Verified |
| GPT-OSS | gpt-oss-20b | up, gate, down | 2-D Bias | Verified |
| DeepSeek-V2 / V3 | DeepSeek-V2-Lite | fused gate_up, down | None | Supported |
| Mixtral | Mixtral-8x7B-v0.1 | up, gate, down | None | Supported |
