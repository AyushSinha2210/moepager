from pathlib import Path

from moepager_tools import collect as c

DATA = Path(__file__).parent / "data"


def test_llama_eval_new_and_old_formats():
    tps, n = c.parse_llama_eval((DATA / "llama_perf_new.txt").read_text())
    assert n == 127 and abs(tps - 12.7) < 1e-9
    tps, n = c.parse_llama_eval((DATA / "llama_print_timings_old.txt").read_text())
    assert n == 63 and abs(tps - 15.75) < 1e-9
    assert c.parse_llama_eval("prompt eval time = 1.0 ms / 3 tokens") is None


def test_llama_bench_rows():
    assert c.parse_llama_bench((DATA / "llama_bench.md").read_text()) == [(128, 15.23, 0.12)]


def test_kv_diskstats_psi():
    assert c.parse_kv("pgmajfault 12\nnr_free_pages 5\nbad line here\n") == {
        "pgmajfault": 12,
        "nr_free_pages": 5,
    }
    ds = "   1       0 ram0 0 0 0 0\n 259       0 nvme0n1 100 0 2048 10 5 0 8 1 0 9 11\n"
    assert c.parse_diskstats_read_bytes(ds, "nvme0n1") == 2048 * 512
    assert c.parse_diskstats_read_bytes(ds, "sda") is None
    psi = c.parse_psi(
        "some avg10=1.50 avg60=0.20 avg300=0.00 total=123456\n"
        "full avg10=0.5 avg60=0 avg300=0 total=7\n"
    )
    assert psi["some"]["avg10"] == 1.5 and psi["full"]["total"] == 7


def _snap(d: Path, sectors: int, majflt: int) -> None:
    d.mkdir(parents=True)
    (d / "device").write_text("nvme0n1\n")
    (d / "diskstats").write_text(f" 259 0 nvme0n1 1 0 {sectors} 0 0 0 0 0 0 0 0\n")
    (d / "vmstat").write_text(f"pgmajfault {majflt}\n")


def test_collect_end_to_end(tmp_path):
    run = tmp_path / "6G" / "B2"
    _snap(run / "before", 1000, 10)
    _snap(run / "after", 1000 + 127 * 2_000_000, 10 + 127 * 50)
    (run / "llama.err").write_text((DATA / "llama_perf_new.txt").read_text())
    (run / "memory.pressure").write_text("some avg10=3.00 avg60=1 avg300=0 total=5000000\n")
    (run / "memory.peak").write_text("6000000000\n")
    (run / "daemon.json").write_text('{"pin_ops": 3}')
    res = c.collect(tmp_path)
    assert len(res) == 1
    r = res[0]
    assert (r.budget, r.config, r.tokens) == ("6G", "B2", 127)
    assert abs(r.ssd_gb_per_token - 2_000_000 * 512 / 1e9) < 1e-9
    assert r.majflt_per_token == 50
    assert r.psi_some_avg10 == 3.0 and r.psi_some_stall_ms == 5000
    assert r.daemon == {"pin_ops": 3}
    md = c.markdown(res)
    assert "| 6G | B2 | 12.70 |" in md and "TBD" in md  # refault unknown without memory.stat


def test_missing_data_is_tbd(tmp_path):
    (tmp_path / "8G" / "B0" / "before").mkdir(parents=True)
    r = c.collect(tmp_path)[0]
    assert r.tok_s is None
    assert c.markdown([r]).count("TBD") == 7
