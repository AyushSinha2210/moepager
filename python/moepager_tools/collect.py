#!/usr/bin/env python3
"""Parse benchmark outputs and build the BENCHMARKS.md results table.

Layout written by bench/run_llama.sh:

    OUT/<budget>/<config>/
        before/ after/      snapshots (vmstat, diskstats, pressure_memory, device, time)
        llama.out llama.err llama.cpp output (timings are on stderr)
        memory.stat memory.pressure memory.peak   cgroup files captured at exit
        daemon.json                               moepagerd counters (if any)

Usage: collect.py OUT_DIR  → markdown table on stdout
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# llama.cpp perf lines, old (llama_print_timings) and new (llama_perf_context_print):
#   eval time =  9876.54 ms /   127 runs   (   77.77 ms per token,    12.86 tokens per second)
_EVAL = re.compile(
    r"^(?:llama_print_timings|llama_perf_context_print|common_perf_print)?:?\s*eval time\s*=\s*"
    r"([\d.]+) ms /\s*(\d+) (?:runs|tokens)",
    re.M,
)
_BENCH_ROW = re.compile(r"^\|.*\|\s*tg(\d+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*\|\s*$", re.M)


def parse_llama_eval(text: str) -> tuple[float, int] | None:
    """(tokens/s, n_tokens) of the decode phase, or None.

    The prompt-eval line also contains "eval time"; lines are matched
    anchored so "prompt eval time" is excluded.
    """
    for line in text.splitlines():
        if "prompt eval time" in line:
            continue
        m = _EVAL.search(line.strip())
        if m:
            ms, n = float(m.group(1)), int(m.group(2))
            if ms > 0 and n > 0:
                return n / (ms / 1e3), n
    return None


def parse_llama_bench(text: str) -> list[tuple[int, float, float]]:
    """llama-bench markdown rows for tg tests: (n_gen, mean t/s, stddev)."""
    return [(int(a), float(b), float(c)) for a, b, c in _BENCH_ROW.findall(text)]


def parse_kv(text: str) -> dict[str, int]:
    """`key value` files such as /proc/vmstat and memory.stat."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            out[parts[0]] = int(parts[1])
    return out


def parse_diskstats_read_bytes(text: str, dev: str) -> int | None:
    """Bytes read from `dev` (field 6 = sectors read, 512 B each)."""
    for line in text.splitlines():
        f = line.split()
        if len(f) > 5 and f[2] == dev:
            return int(f[5]) * 512
    return None


def parse_psi(text: str) -> dict[str, dict[str, float]]:
    """/proc/pressure/* or cgroup *.pressure → {"some": {...}, "full": {...}}."""
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        f = line.split()
        if not f:
            continue
        out[f[0]] = {k: float(v) for k, v in (x.split("=") for x in f[1:])}
    return out


def _read(p: Path) -> str | None:
    try:
        return p.read_text()
    except OSError:
        return None


@dataclass
class RunResult:
    budget: str
    config: str
    tok_s: float | None = None
    tokens: int | None = None
    ssd_gb_per_token: float | None = None
    majflt_per_token: float | None = None
    refault_per_token: float | None = None
    psi_some_avg10: float | None = None
    psi_some_stall_ms: float | None = None
    peak_gb: float | None = None
    daemon: dict | None = None


def collect_run(run: Path, budget: str, config: str) -> RunResult:
    r = RunResult(budget=budget, config=config)
    text = (_read(run / "llama.err") or "") + "\n" + (_read(run / "llama.out") or "")
    ev = parse_llama_eval(text)
    if ev is None:
        rows = parse_llama_bench(text)
        if rows:
            r.tok_s = rows[0][1]
            r.tokens = rows[0][0]
    else:
        r.tok_s, r.tokens = ev
    tokens = r.tokens or 0
    dev = (_read(run / "before" / "device") or "").strip()
    b, a = _read(run / "before" / "diskstats"), _read(run / "after" / "diskstats")
    if dev and b and a and tokens:
        rb, ra = parse_diskstats_read_bytes(b, dev), parse_diskstats_read_bytes(a, dev)
        if rb is not None and ra is not None:
            r.ssd_gb_per_token = (ra - rb) / tokens / 1e9
    ms = _read(run / "memory.stat")
    if ms and tokens:
        st = parse_kv(ms)
        if "pgmajfault" in st:
            r.majflt_per_token = st["pgmajfault"] / tokens
        if "workingset_refault_file" in st:
            r.refault_per_token = st["workingset_refault_file"] / tokens
    elif tokens:
        vb, va = _read(run / "before" / "vmstat"), _read(run / "after" / "vmstat")
        if vb and va:
            r.majflt_per_token = (parse_kv(va)["pgmajfault"] - parse_kv(vb)["pgmajfault"]) / tokens
    pr = _read(run / "memory.pressure")
    if pr:
        some = parse_psi(pr).get("some", {})
        r.psi_some_avg10 = some.get("avg10")
        if "total" in some:
            r.psi_some_stall_ms = some["total"] / 1e3
    peak = _read(run / "memory.peak")
    if peak and peak.strip().isdigit():
        r.peak_gb = int(peak) / 1e9
    d = _read(run / "daemon.json")
    if d:
        try:
            r.daemon = json.loads(d)
        except json.JSONDecodeError:
            pass
    return r


def collect(out: Path) -> list[RunResult]:
    res = []
    for bdir in sorted(p for p in out.iterdir() if p.is_dir()):
        for cdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
            if (cdir / "before").exists() or (cdir / "llama.err").exists():
                res.append(collect_run(cdir, bdir.name, cdir.name))
    return res


def fmt(x: float | None, nd: int = 2) -> str:
    return "TBD" if x is None else f"{x:.{nd}f}"


def markdown(results: list[RunResult]) -> str:
    lines = [
        "| budget | config | tok/s | SSD GB/token | majflt/token | refault/token "
        "| PSI some avg10 | PSI stall ms | peak GB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.budget} | {r.config} | {fmt(r.tok_s)} | {fmt(r.ssd_gb_per_token, 3)} "
            f"| {fmt(r.majflt_per_token, 1)} | {fmt(r.refault_per_token, 1)} "
            f"| {fmt(r.psi_some_avg10)} | {fmt(r.psi_some_stall_ms, 0)} | {fmt(r.peak_gb)} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    out = Path(argv[0])
    res = collect(out)
    sys.stdout.write(markdown(res))
    (out / "results.json").write_text(json.dumps([asdict(r) for r in res], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
