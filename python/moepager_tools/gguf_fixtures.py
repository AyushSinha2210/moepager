#!/usr/bin/env python3
"""Independent GGUF fixture writer.

Writes small synthetic GGUF files (v2/v3) that mimic real MoE layouts
(merged 3-D expert tensors, 2-D per-expert bias tensors, mixed quant types,
32-byte-aligned but not page-aligned offsets) together with an
``*.expected.json`` file that records, computed independently of the Rust
parser, where every expert slice lives.

This is deliberately written from the GGUF spec, not from the Rust code, so
the Rust parser tests are not self-confirming.

Usage: gguf_fixtures.py OUT_DIR [--big]
"""

from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ggml type id -> (name, block size, bytes per block)
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    14: ("Q6_K", 256, 210),
    39: ("MXFP4", 32, 17),
}
TYPE_ID = {v[0]: k for k, v in GGML_TYPES.items()}

# GGUF metadata value types
U32, I32, F32, BOOL, STRING, ARRAY, U64 = 4, 5, 6, 7, 8, 9, 10


def tensor_nbytes(dims: list[int], ty: str) -> int:
    _, bs, ts = GGML_TYPES[TYPE_ID[ty]]
    n = 1
    for d in dims:
        n *= d
    assert dims[0] % bs == 0, "row length must be a multiple of the block size"
    return n // bs * ts


@dataclass
class Tensor:
    name: str
    dims: list[int]  # ne[0] fastest
    ty: str
    offset: int = 0  # relative to the data section


@dataclass
class Fixture:
    version: int = 3
    alignment: int = 32
    arch: str = "testmoe"
    n_layers: int = 2
    n_experts: int = 4
    top_k: int = 2
    kv: list[tuple[str, int, object]] = field(default_factory=list)
    tensors: list[Tensor] = field(default_factory=list)


def _s(b: bytearray, s: str) -> None:
    raw = s.encode()
    b += struct.pack("<Q", len(raw)) + raw


def _value(b: bytearray, vtype: int, v: object) -> None:
    if vtype == U32:
        b += struct.pack("<I", v)
    elif vtype == I32:
        b += struct.pack("<i", v)
    elif vtype == F32:
        b += struct.pack("<f", v)
    elif vtype == BOOL:
        b += struct.pack("<?", v)
    elif vtype == U64:
        b += struct.pack("<Q", v)
    elif vtype == STRING:
        _s(b, v)  # type: ignore[arg-type]
    elif vtype == ARRAY:
        etype, items = v  # type: ignore[misc]
        b += struct.pack("<IQ", etype, len(items))
        for it in items:
            _value(b, etype, it)
    else:
        raise ValueError(vtype)


def write(fx: Fixture, path: Path, fill: bool = True) -> dict:
    """Write the fixture and return the expected-layout dictionary."""
    # Lay out tensors in declaration order with alignment padding.
    off = 0
    for t in fx.tensors:
        off = (off + fx.alignment - 1) // fx.alignment * fx.alignment
        t.offset = off
        off += tensor_nbytes(t.dims, t.ty)
    data_len = off

    h = bytearray(b"GGUF")
    kv = list(fx.kv)
    if fx.alignment != 32:
        kv.append(("general.alignment", U32, fx.alignment))
    h += struct.pack("<I", fx.version)
    h += struct.pack("<QQ", len(fx.tensors), len(kv))
    for key, vtype, v in kv:
        _s(h, key)
        h += struct.pack("<I", vtype)
        _value(h, vtype, v)
    for t in fx.tensors:
        _s(h, t.name)
        h += struct.pack("<I", len(t.dims))
        for d in t.dims:
            h += struct.pack("<Q", d)
        h += struct.pack("<IQ", TYPE_ID[t.ty], t.offset)
    data_off = (len(h) + fx.alignment - 1) // fx.alignment * fx.alignment
    h += b"\0" * (data_off - len(h))

    with open(path, "wb") as f:
        f.write(h)
        if fill:
            # Deterministic non-zero pattern so pages are real, not holes.
            chunk = bytes((i * 131 + 7) & 0xFF or 1 for i in range(1 << 16))
            left = data_len
            while left > 0:
                n = min(left, len(chunk))
                f.write(chunk[:n])
                left -= n
        else:
            f.truncate(data_off + data_len)

    # Expected expert slices, computed from first principles.
    units: dict[tuple[int, int], list[dict]] = {}
    dense = []
    for t in fx.tensors:
        nb = tensor_nbytes(t.dims, t.ty)
        parts = t.name.split(".")
        is_exp = len(parts) >= 3 and parts[0] == "blk" and parts[2].endswith("_exps")
        if is_exp:
            layer = int(parts[1])
            kind = parts[2][: -len("_exps")].removeprefix("ffn_")
            if parts[3] == "bias":
                kind += "_bias"
            n_exp = t.dims[-1]
            per = nb // n_exp
            for e in range(n_exp):
                units.setdefault((layer, e), []).append(
                    {"kind": kind, "offset": data_off + t.offset + e * per, "len": per}
                )
        else:
            dense.append({"name": t.name, "offset": data_off + t.offset, "len": nb})
    exp = {
        "version": fx.version,
        "alignment": fx.alignment,
        "data_offset": data_off,
        "file_size": data_off + data_len,
        "n_layers": fx.n_layers,
        "n_experts": fx.n_experts,
        "top_k": fx.top_k,
        "units": [
            {"layer": k[0], "expert": k[1], "slices": sorted(v, key=lambda s: s["offset"])}
            for k, v in sorted(units.items())
        ],
        "dense": dense,
    }
    path.with_suffix(".expected.json").write_text(json.dumps(exp, indent=1))
    return exp


def moe_fixture(
    *,
    version: int = 3,
    alignment: int = 32,
    n_layers: int = 2,
    n_experts: int = 4,
    top_k: int = 2,
    n_embd: int = 256,
    n_ff: int = 256,
    up_ty: str = "Q4_K",
    down_ty: tuple[str, str] = ("Q6_K", "Q4_K"),
    bias: bool = False,
    vocab: int = 300,
) -> Fixture:
    arch = "testmoe"
    fx = Fixture(version=version, alignment=alignment, arch=arch, n_layers=n_layers,
                 n_experts=n_experts, top_k=top_k)
    fx.kv = [
        ("general.architecture", STRING, arch),
        ("general.name", STRING, "moepager fixture"),
        (f"{arch}.block_count", U32, n_layers),
        (f"{arch}.embedding_length", U32, n_embd),
        (f"{arch}.expert_count", U32, n_experts),
        (f"{arch}.expert_used_count", U32, top_k),
        (f"{arch}.expert_feed_forward_length", U32, n_ff),
        ("general.file_type", U32, 15),
        ("tokenizer.ggml.scores", ARRAY, (F32, [0.5] * vocab)),
        ("tokenizer.ggml.tokens", ARRAY, (STRING, [f"tok{i}" for i in range(vocab)])),
        ("test.bool", BOOL, True),
        ("test.u64", U64, 1 << 40),
        ("test.i32", I32, -7),
    ]
    T = fx.tensors
    T.append(Tensor("token_embd.weight", [n_embd, vocab], "F16"))
    for il in range(n_layers):
        T.append(Tensor(f"blk.{il}.attn_norm.weight", [n_embd], "F32"))
        T.append(Tensor(f"blk.{il}.attn_q.weight", [n_embd, n_embd], "Q8_0"))
        dty = down_ty[il % 2]
        # down: [n_ff, n_embd, n_exp]; gate/up: [n_embd, n_ff, n_exp]
        if bias:
            T.append(Tensor(f"blk.{il}.ffn_down_exps.bias", [n_embd, n_experts], "F32"))
        T.append(Tensor(f"blk.{il}.ffn_down_exps.weight", [n_ff, n_embd, n_experts], dty))
        if bias:
            T.append(Tensor(f"blk.{il}.ffn_gate_exps.bias", [n_ff, n_experts], "F32"))
        T.append(Tensor(f"blk.{il}.ffn_gate_exps.weight", [n_embd, n_ff, n_experts], up_ty))
        T.append(Tensor(f"blk.{il}.ffn_gate_inp.weight", [n_embd, n_experts], "F32"))
        if bias:
            T.append(Tensor(f"blk.{il}.ffn_up_exps.bias", [n_ff, n_experts], "F32"))
        T.append(Tensor(f"blk.{il}.ffn_up_exps.weight", [n_embd, n_ff, n_experts], up_ty))
    T.append(Tensor("output_norm.weight", [n_embd], "F32"))
    T.append(Tensor("output.weight", [n_embd, vocab], "Q8_0"))
    return fx


def dense_fixture() -> Fixture:
    fx = Fixture(arch="testdense", n_layers=2, n_experts=0, top_k=0)
    fx.kv = [
        ("general.architecture", STRING, "testdense"),
        ("testdense.block_count", U32, 2),
    ]
    for il in range(2):
        fx.tensors.append(Tensor(f"blk.{il}.ffn_up.weight", [256, 128], "Q4_K"))
    return fx


def generate(out: Path, big: bool = False) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    made = []
    specs = {
        "tiny_moe.gguf": moe_fixture(),
        "moe_bias_mxfp4.gguf": moe_fixture(alignment=64, n_layers=3, n_experts=8, top_k=4,
                                           n_embd=128, n_ff=128, up_ty="MXFP4",
                                           down_ty=("MXFP4", "MXFP4"), bias=True),
        "v2_moe.gguf": moe_fixture(version=2, n_layers=1, n_experts=2, top_k=1),
        "dense.gguf": dense_fixture(),
    }
    if big:
        # Large enough slices (~544 KiB) for sentinel pages to be reliable;
        # used by the recorder end-to-end test.
        specs["big_moe.gguf"] = moe_fixture(n_layers=2, n_experts=8, top_k=2, n_embd=2048,
                                            n_ff=256, up_ty="Q8_0", down_ty=("Q8_0", "Q8_0"))
    for name, fx in specs.items():
        p = out / name
        write(fx, p)
        made.append(p)
    return made


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    big = "--big" in argv
    out = Path([a for a in argv if not a.startswith("--")][0])
    for p in generate(out, big=big):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
