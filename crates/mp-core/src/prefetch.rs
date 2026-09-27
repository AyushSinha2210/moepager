//! Prefetch planning: expert-completion readahead (prediction-free) and
//! deadline-bounded cross-layer prediction.

use serde::{Deserialize, Serialize};

use crate::{residency::Action, CostModel, OnlineStats};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PrefetchConfig {
    /// On a miss of any slice of a unit, read the rest of the unit in bulk.
    pub completion: bool,
    /// Predict next-layer experts from transition statistics.
    pub predict: bool,
    /// Max predicted units per target layer.
    pub max_per_layer: u32,
    /// Ignore candidates whose score is below this fraction of the best one.
    pub min_rel_score: f64,
    /// Used when no layer-time estimate exists yet (ns).
    pub default_t_layer_ns: f64,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        PrefetchConfig {
            completion: true,
            predict: false,
            max_per_layer: 4,
            min_rel_score: 0.2,
            default_t_layer_ns: 1_000_000.0,
        }
    }
}

pub struct PrefetchPlanner {
    pub cfg: PrefetchConfig,
    sizes: Vec<u64>,
    n_experts: u32,
}

impl PrefetchPlanner {
    pub fn new(cfg: PrefetchConfig, sizes: Vec<u64>, n_experts: u32) -> Self {
        PrefetchPlanner {
            cfg,
            sizes,
            n_experts,
        }
    }

    /// A unit just missed: complete it with one bulk request.
    pub fn on_miss(&self, unit: u32) -> Option<Action> {
        self.cfg.completion.then_some(Action::Prefetch(unit))
    }

    /// Byte budget that can be read before the next layer needs it:
    /// `B_bulk · t_layer_est`.
    pub fn deadline_budget(&self, stats: &OnlineStats, cost: &CostModel) -> u64 {
        let t = if stats.t_layer_ns() > 0.0 {
            stats.t_layer_ns()
        } else {
            self.cfg.default_t_layer_ns
        };
        (cost.bulk_bw * t / 1e9) as u64
    }

    /// After layer `layer`'s experts were observed in this token, choose
    /// next-layer units to prefetch. `resident(u)` filters units already in RAM.
    pub fn after_layer(
        &self,
        stats: &OnlineStats,
        layer: u16,
        cost: &CostModel,
        resident: &dyn Fn(u32) -> bool,
    ) -> Vec<Action> {
        if !self.cfg.predict {
            return Vec::new();
        }
        let scores = stats.predict_next(layer);
        let Some(&(_, best)) = scores.first() else {
            return Vec::new();
        };
        let mut budget = self.deadline_budget(stats, cost);
        let mut out = Vec::new();
        for (e, sc) in scores {
            if out.len() as u32 >= self.cfg.max_per_layer || sc < best * self.cfg.min_rel_score {
                break;
            }
            let u = (layer as u32 + 1) * self.n_experts + e as u32;
            let s = self.sizes[u as usize];
            if resident(u) {
                continue;
            }
            if s > budget {
                break;
            }
            budget -= s;
            out.push(Action::Prefetch(u));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Seen;
    use mp_trace::Observation;

    fn trained() -> OnlineStats {
        let mut s = OnlineStats::new(2, 4, Observation::Full, 100.0, 0.0);
        for t in 0..10u64 {
            s.observe(0, 0, Seen::Hit, t * 10_000);
            s.observe(1, 1, Seen::Hit, t * 10_000 + 1_000_000);
            s.observe(1, 2, Seen::Hit, t * 10_000 + 1_000_000);
            s.end_token(&|_| false);
        }
        s.observe(0, 0, Seen::Hit, 999_000_000);
        s
    }

    #[test]
    fn predicts_within_deadline_budget() {
        let s = trained();
        let cost = CostModel {
            bulk_bw: 2e9,
            ..Default::default()
        };
        let cfg = PrefetchConfig {
            predict: true,
            default_t_layer_ns: 1e6,
            ..Default::default()
        };
        // Budget 2e9 B/s × 1 ms = 2 MB: room for one 1.5 MB unit only.
        let p = PrefetchPlanner::new(cfg, vec![1_500_000; 8], 4);
        assert_eq!(p.deadline_budget(&s, &cost), 2_000_000);
        let a = p.after_layer(&s, 0, &cost, &|_| false);
        assert_eq!(a, vec![Action::Prefetch(5)]);
        // Already-resident candidates are skipped.
        let a = p.after_layer(&s, 0, &cost, &|u| u == 5);
        assert_eq!(a, vec![Action::Prefetch(6)]);
    }

    #[test]
    fn disabled_modes() {
        let s = trained();
        let p = PrefetchPlanner::new(
            PrefetchConfig {
                completion: false,
                ..Default::default()
            },
            vec![1; 8],
            4,
        );
        assert!(p
            .after_layer(&s, 0, &CostModel::default(), &|_| false)
            .is_empty());
        assert_eq!(p.on_miss(3), None);
        let p = PrefetchPlanner::new(PrefetchConfig::default(), vec![1; 8], 4);
        assert_eq!(p.on_miss(3), Some(Action::Prefetch(3)));
    }
}
