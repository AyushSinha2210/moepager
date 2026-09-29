//! I/O cost model shared by the simulator and the prefetch deadline logic.

use serde::{Deserialize, Serialize};

/// Bandwidths in bytes/s (of *touched* bytes, so read amplification is
/// folded in), latencies in ns. Bandwidth defaults are the dev laptop's
/// `moepager fault-io` smoke results (BENCHMARKS.md, Exp. C: ≈0.45 GB/s
/// fault-driven, ≈2.0 GB/s bulk). Re-measure on each target machine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CostModel {
    /// Fixed per-unit overhead of a demand miss (fault entry, first I/O latency).
    pub t_fault_ns: f64,
    /// Effective bandwidth of fault-driven read-around.
    pub demand_bw: f64,
    /// Effective bandwidth of large readahead requests (WILLNEED / pread).
    pub bulk_bw: f64,
    /// Time to notice a miss and issue completion readahead.
    pub detect_ns: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        CostModel {
            t_fault_ns: 100_000.0,
            demand_bw: 0.45e9,
            bulk_bw: 2.0e9,
            detect_ns: 200_000.0,
        }
    }
}

impl CostModel {
    /// Stall for a demand miss of `bytes`, with or without expert-completion readahead.
    pub fn demand_ns(&self, bytes: u64, completion: bool) -> f64 {
        if completion {
            self.t_fault_ns + self.detect_ns + bytes as f64 / self.bulk_bw * 1e9
        } else {
            self.t_fault_ns + bytes as f64 / self.demand_bw * 1e9
        }
    }

    /// Channel time for a bulk prefetch of `bytes`.
    pub fn bulk_ns(&self, bytes: u64) -> f64 {
        bytes as f64 / self.bulk_bw * 1e9
    }

    /// Stall time saved per byte held resident, for a unit missed `rate`
    /// times per token: V(e) = rate · (t_fault + s/B_demand) / s.
    pub fn value_per_byte(&self, rate: f64, bytes: u64) -> f64 {
        if bytes == 0 {
            return 0.0;
        }
        rate * self.demand_ns(bytes, false) / bytes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn costs() {
        let c = CostModel {
            t_fault_ns: 100.0,
            demand_bw: 1e9,
            bulk_bw: 4e9,
            detect_ns: 50.0,
        };
        assert_eq!(c.demand_ns(1_000_000, false), 100.0 + 1e6);
        assert_eq!(c.demand_ns(1_000_000, true), 150.0 + 2.5e5);
        assert_eq!(c.bulk_ns(4_000), 1_000.0);
        // Small units carry relatively more fixed overhead per byte.
        assert!(c.value_per_byte(1.0, 1_000) > c.value_per_byte(1.0, 1_000_000));
        assert_eq!(c.value_per_byte(1.0, 0), 0.0);
    }
}
