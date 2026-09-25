import json
import struct

from moepager_tools import gguf_fixtures as gf


def test_generate_writes_consistent_files(tmp_path):
    paths = gf.generate(tmp_path)
    assert {p.name for p in paths} == {
        "tiny_moe.gguf", "moe_bias_mxfp4.gguf", "v2_moe.gguf", "dense.gguf"
    }
    for p in paths:
        exp = json.loads(p.with_suffix(".expected.json").read_text())
        raw = p.read_bytes()
        assert raw[:4] == b"GGUF"
        assert struct.unpack_from("<I", raw, 4)[0] == exp["version"]
        assert len(raw) == exp["file_size"]
        assert exp["data_offset"] % exp["alignment"] == 0


def test_expert_slices_are_strided_and_disjoint(tmp_path):
    exp = gf.write(gf.moe_fixture(n_layers=2, n_experts=4), tmp_path / "m.gguf")
    assert len(exp["units"]) == 2 * 4
    by_kind = {}
    for u in exp["units"]:
        kinds = sorted(s["kind"] for s in u["slices"])
        assert kinds == ["down", "gate", "up"]
        for s in u["slices"]:
            by_kind.setdefault((u["layer"], s["kind"]), []).append(s)
    for slices in by_kind.values():
        slices.sort(key=lambda s: s["offset"])
        for a, b in zip(slices, slices[1:], strict=False):
            assert a["offset"] + a["len"] == b["offset"], "experts must be contiguous"


def test_bias_fixture_has_bias_slices(tmp_path):
    fx = gf.moe_fixture(bias=True, up_ty="MXFP4", down_ty=("MXFP4", "MXFP4"), n_embd=128,
                        n_ff=128, alignment=64)
    exp = gf.write(fx, tmp_path / "b.gguf")
    kinds = sorted(s["kind"] for s in exp["units"][0]["slices"])
    assert kinds == ["down", "down_bias", "gate", "gate_bias", "up", "up_bias"]


def test_tensor_nbytes():
    assert gf.tensor_nbytes([256, 64, 4], "Q4_K") == 64 * 4 * 144
    assert gf.tensor_nbytes([256, 256, 4], "Q6_K") == 256 * 4 * 210
    assert gf.tensor_nbytes([32, 2], "MXFP4") == 2 * 17
