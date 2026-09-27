//! Reuse distance and the exact LRU miss-ratio curve.
//!
//! The access sequence is the trace in (token, layer, expert) order at
//! expert-unit granularity. The byte reuse distance of an access to unit
//! `u` is the total size of the distinct units accessed since the previous
//! access to `u`. With a byte-capacity LRU cache, the resident set is
//! always a prefix of the recency stack, so an access hits iff
//! `distance + size(u) <= C`. That gives the whole LRU curve in one pass
//! (Fenwick tree, O(N log N)), independently of the simulator.

use mp_trace::Steps;
use serde::Serialize;

struct Fenwick(Vec<i64>);

impl Fenwick {
    fn new(n: usize) -> Self {
        Fenwick(vec![0; n + 1])
    }
    fn add(&mut self, i: usize, v: i64) {
        let mut i = i + 1;
        while i < self.0.len() {
            self.0[i] += v;
            i += i & i.wrapping_neg();
        }
    }
    /// Sum of [0, i).
    fn prefix(&self, i: usize) -> i64 {
        let mut i = i;
        let mut s = 0;
        while i > 0 {
            s += self.0[i];
            i -= i & i.wrapping_neg();
        }
        s
    }
}

/// Per-access reuse information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reuse {
    pub unit: u32,
    pub size: u64,
    /// Distinct units in between (None = first access).
    pub dist_units: Option<u64>,
    /// Bytes of distinct units in between (None = first access).
    pub dist_bytes: Option<u64>,
}

/// Unit id sequence in trace order.
pub fn access_sequence(s: &Steps) -> Vec<u32> {
    let mut v = Vec::with_capacity(s.n_accesses());
    for st in &s.steps {
        for l in &st.layers {
            for &e in &l.experts {
                v.push(s.unit(l.layer, e));
            }
        }
    }
    v
}

/// Reuse distances for a unit sequence; `size(u)` gives unit sizes.
pub fn reuse_distances(seq: &[u32], n_units: u32, size: &dyn Fn(u32) -> u64) -> Vec<Reuse> {
    let mut last: Vec<Option<usize>> = vec![None; n_units as usize];
    let mut fb = Fenwick::new(seq.len()); // bytes markers
    let mut fu = Fenwick::new(seq.len()); // unit markers
    let mut out = Vec::with_capacity(seq.len());
    for (i, &u) in seq.iter().enumerate() {
        let sz = size(u);
        let (du, db) = match last[u as usize] {
            Some(p) => {
                let db = fb.prefix(i) - fb.prefix(p + 1);
                let du = fu.prefix(i) - fu.prefix(p + 1);
                fb.add(p, -(sz as i64));
                fu.add(p, -1);
                (Some(du as u64), Some(db as u64))
            }
            None => (None, None),
        };
        fb.add(i, sz as i64);
        fu.add(i, 1);
        last[u as usize] = Some(i);
        out.push(Reuse {
            unit: u,
            size: sz,
            dist_units: du,
            dist_bytes: db,
        });
    }
    out
}

/// One point of the LRU miss-ratio curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MrcPoint {
    pub capacity_bytes: u64,
    pub misses: u64,
    pub miss_bytes: u64,
    pub miss_ratio: f64,
}

/// Exact LRU misses at each capacity (cold misses included).
pub fn lru_mrc(r: &[Reuse], capacities: &[u64]) -> Vec<MrcPoint> {
    // need = bytes required for a hit; u64::MAX for cold accesses.
    let mut need: Vec<(u64, u64)> = r
        .iter()
        .map(|x| (x.dist_bytes.map_or(u64::MAX, |d| d + x.size), x.size))
        .collect();
    need.sort_unstable();
    // suffix sums of sizes so miss bytes at C = sum of sizes with need > C.
    let mut suffix = vec![0u64; need.len() + 1];
    for i in (0..need.len()).rev() {
        suffix[i] = suffix[i + 1] + need[i].1;
    }
    capacities
        .iter()
        .map(|&c| {
            let first_miss = need.partition_point(|&(n, _)| n <= c);
            let misses = (need.len() - first_miss) as u64;
            MrcPoint {
                capacity_bytes: c,
                misses,
                miss_bytes: suffix[first_miss],
                miss_ratio: if need.is_empty() {
                    0.0
                } else {
                    misses as f64 / need.len() as f64
                },
            }
        })
        .collect()
}

/// Histogram of unit reuse distances in power-of-two buckets
/// (`bucket[i]` counts distances in [2^(i-1), 2^i), bucket 0 = distance 0).
pub fn log2_histogram(r: &[Reuse]) -> (Vec<u64>, u64) {
    let mut h = vec![0u64; 1];
    let mut cold = 0;
    for x in r {
        match x.dist_units {
            None => cold += 1,
            Some(d) => {
                let b = if d == 0 {
                    0
                } else {
                    64 - d.leading_zeros() as usize
                };
                if h.len() <= b {
                    h.resize(b + 1, 0);
                }
                h[b] += 1;
            }
        }
    }
    (h, cold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_computed_distances() {
        // a b c a b b d a
        let seq = [0, 1, 2, 0, 1, 1, 3, 0];
        let r = reuse_distances(&seq, 4, &|_| 1);
        let d: Vec<Option<u64>> = r.iter().map(|x| x.dist_units).collect();
        assert_eq!(
            d,
            vec![None, None, None, Some(2), Some(2), Some(0), None, Some(2)]
        );
        // Byte distances with sizes a=1 b=10 c=100 d=1000.
        let sz = |u: u32| 10u64.pow(u);
        let r = reuse_distances(&seq, 4, &sz);
        assert_eq!(r[3].dist_bytes, Some(110));
        assert_eq!(r[7].dist_bytes, Some(1010));
    }

    #[test]
    fn mrc_on_cyclic_scan_is_all_miss_until_it_fits() {
        // LRU on a loop over 4 equal units: 0 hits below 4 units of capacity.
        let seq: Vec<u32> = (0..40).map(|i| i % 4).collect();
        let r = reuse_distances(&seq, 4, &|_| 10);
        let m = lru_mrc(&r, &[30, 39, 40]);
        assert_eq!(m[0].misses, 40);
        assert_eq!(m[1].misses, 40);
        assert_eq!(m[2].misses, 4); // only cold misses
        assert_eq!(m[2].miss_bytes, 40);
        let (h, cold) = log2_histogram(&r);
        assert_eq!(cold, 4);
        assert_eq!(h.iter().sum::<u64>(), 36);
    }
}
