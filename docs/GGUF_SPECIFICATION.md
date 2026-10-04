# GGUF Specification and Tensor Layouts

An overview of GGUF v2/v3 header structures and expert tensor conventions parsed by `mp-gguf`.

## Tensor Naming in MoE Models
- `blk.<L>.ffn_up_exps.weight`: 3-D tensor containing up-projection weights for all experts in layer $L$.
- `blk.<L>.ffn_gate_exps.weight`: 3-D tensor containing gate weights.
- `blk.<L>.ffn_down_exps.weight`: 3-D tensor containing down-projection weights.
- `blk.<L>.ffn_gate_up_exps.weight`: Fused gate-up projection tensor in select architectures.

## Alignment Rules
Tensors are aligned to `general.alignment` (typically 32 bytes). Slices are not guaranteed to align with 4096-byte page boundaries, meaning boundary pages can overlap adjacent experts.
