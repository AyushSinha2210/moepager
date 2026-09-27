//! Offline trace analysis: reuse distance, exact LRU miss-ratio curve,
//! routing statistics and timing.

pub mod reuse;
pub mod stats;

use mp_trace::Steps;
use serde::Serialize;

pub use reuse::{access_sequence, log2_histogram, lru_mrc, reuse_distances, MrcPoint, Reuse};

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub tokens: usize,
    pub accesses: usize,
    pub footprint_bytes: u64,
    pub cold_misses: u64,
    /// Unit reuse distances, log2 buckets (bucket 0 = distance 0).
    pub reuse_hist_log2: Vec<u64>,
    pub median_reuse_units: Option<u64>,
    pub lru_mrc: Vec<MrcPoint>,
    pub token_reuse: stats::ReuseStats,
    pub popularity: stats::Popularity,
    pub predictability: stats::Predictability,
    pub timing: stats::Timing,
}

/// Analyze a trace. `unit_bytes[layer * n_experts + expert]` gives unit
/// sizes (uniform 1 byte when `None`). MRC capacities are the given
/// fractions of the footprint (distinct bytes touched).
pub fn analyze(s: &Steps, unit_bytes: Option<&[u64]>, mrc_fractions: &[f64]) -> Report {
    let size = |u: u32| {
        unit_bytes
            .and_then(|b| b.get(u as usize).copied())
            .unwrap_or(1)
    };
    let seq = access_sequence(s);
    let r = reuse_distances(&seq, s.n_units(), &size);
    let mut seen = vec![false; s.n_units() as usize];
    let mut footprint = 0;
    for &u in &seq {
        if !std::mem::replace(&mut seen[u as usize], true) {
            footprint += size(u);
        }
    }
    let caps: Vec<u64> = mrc_fractions
        .iter()
        .map(|f| (footprint as f64 * f).round() as u64)
        .collect();
    let (hist, cold) = log2_histogram(&r);
    let mut d: Vec<u64> = r.iter().filter_map(|x| x.dist_units).collect();
    d.sort_unstable();
    Report {
        tokens: s.steps.len(),
        accesses: seq.len(),
        footprint_bytes: footprint,
        cold_misses: cold,
        reuse_hist_log2: hist,
        median_reuse_units: d.get(d.len() / 2).copied(),
        lru_mrc: lru_mrc(&r, &caps),
        token_reuse: stats::token_reuse(s),
        popularity: stats::popularity(s),
        predictability: stats::predictability(s),
        timing: stats::timing(s),
    }
}

/// Default MRC capacity grid: 5 %, 10 %, …, 100 % of the footprint.
pub fn default_fractions() -> Vec<f64> {
    (1..=20).map(|i| i as f64 * 0.05).collect()
}
