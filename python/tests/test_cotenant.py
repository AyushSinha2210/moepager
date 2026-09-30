from moepager_tools import cotenant


def test_percentile():
    assert cotenant.percentile([], 0.5) == 0.0
    assert cotenant.percentile([3, 1, 2], 0.5) == 2
    assert cotenant.percentile(list(range(101)), 0.95) == 95


def test_short_run_produces_stats():
    r = cotenant.run(mib=4, period_ms=5, duration_s=0.1)
    assert r["sweeps"] >= 2
    assert 0 <= r["p50_ms"] <= r["p95_ms"] <= r["max_ms"]
