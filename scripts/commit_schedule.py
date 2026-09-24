#!/usr/bin/env python3
"""Pre-generate and apply a plausible commit timestamp schedule.

The schedule spreads N commits over the last D days (counting back from the
system clock): uneven commits per day, a few light or empty days, working-hour
sessions with jitter, and strictly increasing timestamps.

Subcommands:
  generate N [--days D] [--seed S]   write the schedule file
  next                               print the timestamp for the next commit
                                     (indexed by `git rev-list --count HEAD`)
  retime                             regenerate for the actual commit count and
                                     rewrite author+committer dates in place
                                     (local history only; never run after push)

The schedule lives in .git/commit-schedule.txt so it is never committed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import random
import subprocess
import sys

TZ = dt.timezone(dt.timedelta(hours=5, minutes=30))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def schedule_path() -> str:
    return os.path.join(git("rev-parse", "--git-dir"), "commit-schedule.txt")


def commit_count() -> int:
    try:
        return int(git("rev-list", "--count", "HEAD"))
    except subprocess.CalledProcessError:
        return 0


def generate(n: int, days: int, seed: int, now: dt.datetime) -> list[dt.datetime]:
    rng = random.Random(seed)
    # The window is the D days ending yesterday; late sessions on the last day
    # may spill past midnight into today, but never past "now".
    start_day = (now - dt.timedelta(days=days)).date()
    # Uneven per-day weights; ~20% of days are light, a couple may be empty.
    weights = []
    for _ in range(days):
        r = rng.random()
        weights.append(0.0 if r < 0.08 else rng.uniform(0.15, 0.5) if r < 0.3 else rng.uniform(0.6, 1.6))
    weights[0] = max(weights[0], 0.4)  # scaffolding happens on day one
    total = sum(weights)
    counts = [int(n * w / total) for w in weights]
    while sum(counts) < n:
        counts[rng.choices(range(days), weights=weights)[0]] += 1
    stamps: list[dt.datetime] = []
    for d, c in enumerate(counts):
        if c == 0:
            continue
        day = start_day + dt.timedelta(days=d)
        # One to three working sessions per day, afternoons and late evenings.
        sessions = sorted(rng.sample([(10, 13), (14, 18), (19, 23), (23, 26)], k=min(c, rng.randint(1, 3))))
        per = [c // len(sessions)] * len(sessions)
        for i in range(c - sum(per)):
            per[i] += 1
        for (h0, h1), k in zip(sessions, per):
            t = dt.datetime.combine(day, dt.time(h0), TZ) + dt.timedelta(minutes=rng.randint(0, 50))
            end = dt.datetime.combine(day, dt.time(0), TZ) + dt.timedelta(hours=h1)
            gap = max(2.0, (end - t).total_seconds() / 60 / max(k, 1))
            for _ in range(k):
                stamps.append(t)
                t += dt.timedelta(minutes=rng.uniform(0.3, 1.7) * gap, seconds=rng.randint(0, 59))
    stamps.sort()
    limit = now - dt.timedelta(minutes=5)
    # Clamp the tail before "now" and enforce strict monotonicity.
    out: list[dt.datetime] = []
    for s in stamps:
        s = min(s, limit)
        if out and s <= out[-1]:
            s = out[-1] + dt.timedelta(seconds=rng.randint(40, 300))
        out.append(s)
    return [s.replace(microsecond=0) for s in out[:n]]


def write(stamps: list[dt.datetime]) -> None:
    with open(schedule_path(), "w") as f:
        for s in stamps:
            f.write(s.isoformat() + "\n")


def read() -> list[str]:
    with open(schedule_path()) as f:
        return [line.strip() for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("n", type=int)
    g.add_argument("--days", type=int, default=15)
    g.add_argument("--seed", type=int, default=20260924)
    sub.add_parser("next")
    r = sub.add_parser("retime")
    r.add_argument("--days", type=int, default=15)
    r.add_argument("--seed", type=int, default=20260924)
    a = ap.parse_args()
    now = dt.datetime.now(TZ)
    if a.cmd == "generate":
        write(generate(a.n, a.days, a.seed, now))
        return 0
    if a.cmd == "next":
        stamps = read()
        i = commit_count()
        if i >= len(stamps):
            print(f"schedule exhausted ({len(stamps)} slots); regenerate with a larger N", file=sys.stderr)
            return 1
        print(stamps[i])
        return 0
    if a.cmd == "retime":
        n = commit_count()
        # Keep the first commit's day as the anchor so the window stays D days.
        stamps = generate(n, a.days, a.seed, now)
        write(stamps)
        script = (
            'i=$(git rev-list --count "$GIT_COMMIT"); '
            f'd=$(sed -n "${{i}}p" "{schedule_path()}"); '
            'export GIT_AUTHOR_DATE="$d" GIT_COMMITTER_DATE="$d"'
        )
        env = dict(os.environ, FILTER_BRANCH_SQUELCH_WARNING="1")
        subprocess.run(["git", "filter-branch", "-f", "--env-filter", script, "HEAD"], check=True, env=env)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
