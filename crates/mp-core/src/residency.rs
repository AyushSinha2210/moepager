//! V(e)-residency: choose which expert units to pin within a byte budget.

use mp_trace::Observation;
use serde::{Deserialize, Serialize};

use crate::{CostModel, OnlineStats, Rng};

/// An action for the OS layer (daemon) or the simulated cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Pin(u32),
    Unpin(u32),
    Prefetch(u32),
    Demote(u32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResidencyConfig {
    pub pin_budget_bytes: u64,
    /// Re-plan every this many tokens.
    pub replan_every: u32,
    /// Pinned units keep their slot unless a newcomer's value beats theirs by this factor.
    pub hysteresis: f64,
    /// In the miss-only regime, unpin this fraction of pinned units per
    /// re-plan to refresh their (frozen) estimates.
    pub explore_frac: f64,
    /// Emit `Demote` for units that drop out of the pinned set.
    pub demote_unpinned: bool,
    pub seed: u64,
}

impl Default for ResidencyConfig {
    fn default() -> Self {
        ResidencyConfig {
            pin_budget_bytes: 0,
            replan_every: 4,
            hysteresis: 1.25,
            explore_frac: 0.02,
            demote_unpinned: false,
            seed: 7,
        }
    }
}

pub struct ResidencyEngine {
    pub cfg: ResidencyConfig,
    sizes: Vec<u64>,
    pinned: Vec<bool>,
    pinned_bytes: u64,
    since: u32,
    rng: Rng,
    churn: u64,
}

impl ResidencyEngine {
    pub fn new(cfg: ResidencyConfig, sizes: Vec<u64>) -> Self {
        let n = sizes.len();
        let rng = Rng::new(cfg.seed);
        ResidencyEngine {
            cfg,
            sizes,
            pinned: vec![false; n],
            pinned_bytes: 0,
            since: 0,
            rng,
            churn: 0,
        }
    }

    pub fn is_pinned(&self, u: u32) -> bool {
        self.pinned[u as usize]
    }

    pub fn pinned_bytes(&self) -> u64 {
        self.pinned_bytes
    }

    /// Pin + unpin actions issued so far.
    pub fn churn(&self) -> u64 {
        self.churn
    }

    /// Change the budget (e.g. under memory pressure); takes effect at the next plan.
    pub fn set_budget(&mut self, bytes: u64) {
        self.cfg.pin_budget_bytes = bytes;
        self.since = self.cfg.replan_every;
    }

    /// Called once per token; returns actions when a re-plan is due.
    pub fn on_token(&mut self, stats: &OnlineStats, cost: &CostModel) -> Vec<Action> {
        self.since += 1;
        if self.since < self.cfg.replan_every {
            return Vec::new();
        }
        self.since = 0;
        self.plan(stats, cost)
    }

    /// Compute the desired pinned set and return the diff as actions.
    // Several parallel per-unit vectors are indexed together; a range loop is clearest.
    #[allow(clippy::needless_range_loop)]
    pub fn plan(&mut self, stats: &OnlineStats, cost: &CostModel) -> Vec<Action> {
        let n = self.sizes.len();
        // Exploration: force-release a few pinned units so their estimates refresh.
        let mut excluded = vec![false; n];
        if stats.regime == Observation::MissOnly && self.cfg.explore_frac > 0.0 {
            for u in 0..n {
                if self.pinned[u] && self.rng.chance(self.cfg.explore_frac) {
                    excluded[u] = true;
                }
            }
        }
        let mut cand: Vec<(f64, u32)> = (0..n as u32)
            .filter(|&u| !excluded[u as usize] && self.sizes[u as usize] > 0)
            .map(|u| {
                let mut v = cost.value_per_byte(stats.rate(u), self.sizes[u as usize]);
                if self.pinned[u as usize] {
                    v *= self.cfg.hysteresis;
                }
                (v, u)
            })
            .filter(|&(v, _)| v > 0.0)
            .collect();
        cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut want = vec![false; n];
        let mut used = 0u64;
        for &(_, u) in &cand {
            let s = self.sizes[u as usize];
            if used + s <= self.cfg.pin_budget_bytes {
                want[u as usize] = true;
                used += s;
            }
        }
        let mut actions = Vec::new();
        // Unpins first so the budget is never exceeded mid-way.
        for u in 0..n {
            if self.pinned[u] && !want[u] {
                self.pinned[u] = false;
                self.pinned_bytes -= self.sizes[u];
                actions.push(Action::Unpin(u as u32));
                if self.cfg.demote_unpinned {
                    actions.push(Action::Demote(u as u32));
                }
            }
        }
        for u in 0..n {
            if !self.pinned[u] && want[u] {
                self.pinned[u] = true;
                self.pinned_bytes += self.sizes[u];
                actions.push(Action::Pin(u as u32));
            }
        }
        self.churn += actions
            .iter()
            .filter(|a| matches!(a, Action::Pin(_) | Action::Unpin(_)))
            .count() as u64;
        debug_assert!(self.pinned_bytes <= self.cfg.pin_budget_bytes);
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Seen;

    fn stats_with(rates: &[f64], regime: Observation) -> OnlineStats {
        let mut s = OnlineStats::new(1, rates.len() as u32, regime, 1e9, 0.0);
        let mut acc = vec![0.0; rates.len()];
        for t in 0..200 {
            for (e, r) in rates.iter().enumerate() {
                acc[e] += r;
                if acc[e] >= 1.0 {
                    acc[e] -= 1.0;
                    s.observe(0, e as u16, Seen::Miss, t);
                }
            }
            s.end_token(&|_| false);
        }
        s
    }

    #[test]
    fn pins_highest_value_within_budget() {
        let s = stats_with(&[0.9, 0.1, 0.5, 0.0], Observation::Full);
        let cfg = ResidencyConfig {
            pin_budget_bytes: 2_000,
            ..Default::default()
        };
        let mut e = ResidencyEngine::new(cfg, vec![1_000; 4]);
        let a = e.plan(&s, &CostModel::default());
        assert_eq!(a, vec![Action::Pin(0), Action::Pin(2)]);
        assert_eq!(e.pinned_bytes(), 2_000);
        // Stable: re-planning with the same stats does nothing.
        assert!(e.plan(&s, &CostModel::default()).is_empty());
    }

    #[test]
    fn hysteresis_limits_churn() {
        let cost = CostModel::default();
        let cfg = ResidencyConfig {
            pin_budget_bytes: 1_000,
            hysteresis: 1.5,
            ..Default::default()
        };
        let mut e = ResidencyEngine::new(cfg, vec![1_000; 2]);
        e.plan(&stats_with(&[0.5, 0.4], Observation::Full), &cost);
        assert!(e.is_pinned(0));
        // 0.55 vs 0.5: not enough to displace under hysteresis 1.5.
        assert!(e
            .plan(&stats_with(&[0.5, 0.55], Observation::Full), &cost)
            .is_empty());
        // 0.9 vs 0.5: displaces.
        let a = e.plan(&stats_with(&[0.5, 0.9], Observation::Full), &cost);
        assert_eq!(a, vec![Action::Unpin(0), Action::Pin(1)]);
        assert_eq!(e.churn(), 3);
    }

    #[test]
    fn budget_shrink_and_on_token_cadence() {
        let s = stats_with(&[0.9, 0.8, 0.7], Observation::Full);
        let cfg = ResidencyConfig {
            pin_budget_bytes: 3_000,
            replan_every: 3,
            ..Default::default()
        };
        let mut e = ResidencyEngine::new(cfg, vec![1_000; 3]);
        let c = CostModel::default();
        assert!(e.on_token(&s, &c).is_empty());
        assert!(e.on_token(&s, &c).is_empty());
        assert_eq!(e.on_token(&s, &c).len(), 3);
        e.set_budget(1_000);
        let a = e.on_token(&s, &c);
        assert_eq!(a, vec![Action::Unpin(1), Action::Unpin(2)]);
        assert_eq!(e.pinned_bytes(), 1_000);
    }

    #[test]
    fn exploration_only_in_miss_only_regime() {
        let c = CostModel::default();
        let cfg = ResidencyConfig {
            pin_budget_bytes: 50_000,
            explore_frac: 0.5,
            ..Default::default()
        };
        let rates = vec![0.5; 50];
        let full = stats_with(&rates, Observation::Full);
        let mut e = ResidencyEngine::new(cfg.clone(), vec![1_000; 50]);
        e.plan(&full, &c);
        assert!(e.plan(&full, &c).is_empty());
        let miss = stats_with(&rates, Observation::MissOnly);
        let mut e = ResidencyEngine::new(cfg, vec![1_000; 50]);
        e.plan(&miss, &c);
        assert!(e
            .plan(&miss, &c)
            .iter()
            .any(|a| matches!(a, Action::Unpin(_))));
    }
}
