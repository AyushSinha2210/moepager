#!/usr/bin/env python3
"""Co-tenant probe: how much does the model run slow down another app?

Keeps an anonymous working set of N MiB and rewrites one byte per page every
period, timing each sweep. Under memory pressure the sweep stalls (reclaim,
swap-in), so p95 sweep latency relative to a solo run measures co-tenant
slowdown (BENCHMARKS.md).

Usage: cotenant.py --mib 512 --period-ms 250 --duration-s 60 --out probe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, round((len(s) - 1) * p))]


def run(mib: int, period_ms: float, duration_s: float) -> dict:
    buf = bytearray(mib << 20)
    pages = len(buf) // 4096
    lat = []
    end = time.monotonic() + duration_s
    v = 1
    while time.monotonic() < end:
        t = time.perf_counter()
        buf[::4096] = bytes([v]) * pages  # touch every page (C speed)
        lat.append((time.perf_counter() - t) * 1e3)
        v = v % 255 + 1
        time.sleep(max(0.0, period_ms / 1e3 - lat[-1] / 1e3))
    return {
        "mib": mib,
        "sweeps": len(lat),
        "p50_ms": percentile(lat, 0.5),
        "p95_ms": percentile(lat, 0.95),
        "max_ms": max(lat) if lat else 0.0,
        "mean_ms": statistics.fmean(lat) if lat else 0.0,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=512)
    ap.add_argument("--period-ms", type=float, default=250)
    ap.add_argument("--duration-s", type=float, default=60)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    r = run(a.mib, a.period_ms, a.duration_s)
    s = json.dumps(r, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s)
    print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
