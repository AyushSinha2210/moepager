//! moepagerd core: turns black-box expert observations into page-cache
//! actions through the same mp-core logic the simulator evaluates.
//!
//! Event flow: `EventSource` → [`Daemon::on_event`] → `OnlineStats` →
//! (`ResidencyEngine` at token boundaries, `PrefetchPlanner` per layer and
//! per miss) → `Action`s → `PageCacheOps` on the unit's file slices.

use std::io;

use mp_core::{
    Action, CostModel, OnlineStats, PrefetchConfig, PrefetchPlanner, ResidencyConfig,
    ResidencyEngine, Seen,
};
use mp_gguf::ExpertMap;
use mp_os::PageCacheOps;
use mp_trace::{ExpertEvent, Observation};
use serde::{Deserialize, Serialize};

pub mod source;

/// Residency policy run by the daemon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Policy {
    /// Observe and log only.
    None,
    /// Baseline B5: WILLNEED every expert of layer l+1 when layer l is seen.
    WillneedAll,
    /// Baseline B6: pin the most recently missed units within the budget.
    LruPin,
    /// moepager: V(e)-residency from miss-only statistics.
    V,
}

impl std::str::FromStr for Policy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Policy::None),
            "willneed-all" => Ok(Policy::WillneedAll),
            "lru-pin" => Ok(Policy::LruPin),
            "v" => Ok(Policy::V),
            _ => Err(format!("unknown policy {s} (none|willneed-all|lru-pin|v)")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DaemonConfig {
    pub policy: Policy,
    /// Bytes the daemon may pin.
    pub pin_budget_bytes: u64,
    pub half_life_tokens: f64,
    pub residency: ResidencyConfig,
    pub prefetch: PrefetchConfig,
    pub cost: CostModel,
    /// Layer drop that marks a new token (see `mp_trace::infer_tokens`).
    pub token_slack: u16,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        DaemonConfig {
            policy: Policy::V,
            pin_budget_bytes: 0,
            half_life_tokens: 64.0,
            residency: ResidencyConfig::default(),
            prefetch: PrefetchConfig::default(),
            cost: CostModel::default(),
            token_slack: 2,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Counters {
    pub events: u64,
    pub tokens: u64,
    pub prefetch_ops: u64,
    pub pin_ops: u64,
    pub unpin_ops: u64,
    pub demote_ops: u64,
    pub failed_ops: u64,
    /// Times the pin budget was cut because pinning failed (RLIMIT_MEMLOCK).
    pub budget_clamps: u64,
}

pub struct Daemon<O: PageCacheOps> {
    pub cfg: DaemonConfig,
    map: ExpertMap,
    n_experts: u32,
    /// Dense unit id (layer * n_experts + expert) → map unit index.
    dense_to_map: Vec<Option<u32>>,
    stats: OnlineStats,
    engine: ResidencyEngine,
    planner: PrefetchPlanner,
    pub ops: O,
    pub counters: Counters,
    cur_max_layer: Option<u16>,
    last_layer: Option<u16>,
    /// LruPin: pinned units in recency order (front = oldest).
    lru: std::collections::VecDeque<u32>,
    lru_bytes: u64,
    sizes: Vec<u64>,
}

impl<O: PageCacheOps> Daemon<O> {
    pub fn new(map: ExpertMap, cfg: DaemonConfig, ops: O) -> Self {
        let (nl, ne) = (map.n_layers, map.n_experts);
        let mut dense_to_map = vec![None; (nl * ne) as usize];
        let mut sizes = vec![0u64; (nl * ne) as usize];
        for (i, u) in map.units.iter().enumerate() {
            let d = (u.layer * ne + u.expert) as usize;
            dense_to_map[d] = Some(i as u32);
            sizes[d] = u.bytes;
        }
        let prior = map.top_k.unwrap_or(1) as f64 / ne.max(1) as f64;
        let stats = OnlineStats::new(nl, ne, Observation::MissOnly, cfg.half_life_tokens, prior);
        let mut rc = cfg.residency.clone();
        rc.pin_budget_bytes = cfg.pin_budget_bytes;
        let engine = ResidencyEngine::new(rc, sizes.clone());
        let planner = PrefetchPlanner::new(cfg.prefetch.clone(), sizes.clone(), ne);
        Daemon {
            cfg,
            map,
            n_experts: ne,
            dense_to_map,
            stats,
            engine,
            planner,
            ops,
            counters: Counters::default(),
            cur_max_layer: None,
            last_layer: None,
            lru: Default::default(),
            lru_bytes: 0,
            sizes,
        }
    }

    pub fn pinned_bytes(&self) -> u64 {
        match self.cfg.policy {
            Policy::LruPin => self.lru_bytes,
            _ => self.engine.pinned_bytes(),
        }
    }

    fn dense(&self, layer: u16, expert: u16) -> u32 {
        layer as u32 * self.n_experts + expert as u32
    }

    /// Feed one observed expert miss.
    pub fn on_event(&mut self, ev: &ExpertEvent) {
        self.counters.events += 1;
        if let Some(m) = self.cur_max_layer {
            if ev.layer.saturating_add(self.cfg.token_slack) < m {
                self.end_token();
            }
        }
        // A newly reached layer closes the previous one for prediction.
        if let Some(prev) = self.last_layer {
            if ev.layer > prev && self.cfg.policy == Policy::V {
                let acts = self
                    .planner
                    .after_layer(&self.stats, prev, &self.cfg.cost, &|_| false);
                self.apply(acts);
            }
        }
        self.cur_max_layer = Some(self.cur_max_layer.map_or(ev.layer, |m| m.max(ev.layer)));
        self.last_layer = Some(ev.layer);
        self.stats.observe(ev.layer, ev.expert, Seen::Miss, ev.t_ns);
        let u = self.dense(ev.layer, ev.expert);
        if let Some(a) = self.planner.on_miss(u) {
            self.apply(vec![a]);
        }
        match self.cfg.policy {
            Policy::WillneedAll => {
                let next = ev.layer as u32 + 1;
                if next < self.map.n_layers {
                    let acts = (0..self.n_experts)
                        .map(|e| Action::Prefetch(next * self.n_experts + e))
                        .collect();
                    self.apply(acts);
                }
            }
            Policy::LruPin => self.lru_touch(u),
            Policy::None | Policy::V => {}
        }
    }

    /// Close the current token (also called on inferred boundaries).
    pub fn end_token(&mut self) {
        self.counters.tokens += 1;
        self.cur_max_layer = None;
        self.last_layer = None;
        let engine = &self.engine;
        self.stats.end_token(&|u| engine.is_pinned(u));
        if self.cfg.policy == Policy::V {
            let acts = self.engine.on_token(&self.stats, &self.cfg.cost);
            self.apply(acts);
        }
    }

    fn lru_touch(&mut self, u: u32) {
        if let Some(i) = self.lru.iter().position(|&x| x == u) {
            self.lru.remove(i);
            self.lru.push_back(u);
            return;
        }
        let s = self.sizes[u as usize];
        if s > self.cfg.pin_budget_bytes {
            return;
        }
        while self.lru_bytes + s > self.cfg.pin_budget_bytes {
            let Some(old) = self.lru.pop_front() else {
                break;
            };
            self.lru_bytes -= self.sizes[old as usize];
            self.apply(vec![Action::Unpin(old)]);
        }
        if self.apply_one(Action::Pin(u)) {
            self.lru.push_back(u);
            self.lru_bytes += s;
        }
    }

    fn apply(&mut self, acts: Vec<Action>) {
        for a in acts {
            self.apply_one(a);
        }
    }

    /// Execute one action on every slice of the unit. Returns success.
    fn apply_one(&mut self, a: Action) -> bool {
        let u = match a {
            Action::Pin(u) | Action::Unpin(u) | Action::Prefetch(u) | Action::Demote(u) => u,
        };
        let Some(Some(mi)) = self.dense_to_map.get(u as usize).copied() else {
            return false;
        };
        let slices: Vec<(u64, u64)> = self.map.units[mi as usize]
            .slices
            .iter()
            .map(|s| (s.offset, s.len))
            .collect();
        let mut ok = true;
        for (off, len) in slices {
            let r: io::Result<()> = match a {
                Action::Prefetch(_) => self.ops.prefetch(off, len),
                Action::Pin(_) => self.ops.pin(off, len),
                Action::Unpin(_) => self.ops.unpin(off, len),
                Action::Demote(_) => self.ops.demote(off, len),
            };
            if r.is_err() {
                ok = false;
                self.counters.failed_ops += 1;
            }
        }
        match a {
            Action::Prefetch(_) => self.counters.prefetch_ops += 1,
            Action::Pin(_) => self.counters.pin_ops += 1,
            Action::Unpin(_) => self.counters.unpin_ops += 1,
            Action::Demote(_) => self.counters.demote_ops += 1,
        }
        if !ok && matches!(a, Action::Pin(_)) {
            // Most likely RLIMIT_MEMLOCK: undo partial pins and stop asking for more.
            let _ = self.apply_unpin_quiet(u);
            let now = self.pinned_bytes();
            if self.cfg.pin_budget_bytes > now {
                self.cfg.pin_budget_bytes = now;
                self.engine.set_budget(now);
                self.counters.budget_clamps += 1;
            }
        }
        ok
    }

    fn apply_unpin_quiet(&mut self, u: u32) -> io::Result<()> {
        if let Some(Some(mi)) = self.dense_to_map.get(u as usize).copied() {
            for s in &self.map.units[mi as usize].slices {
                self.ops.unpin(s.offset, s.len)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mp_gguf::{Slice, Unit};
    use mp_os::{MockOps, OpCall};

    /// A synthetic map: `nl` layers × `ne` experts, 3 slices of 64 KiB each.
    pub(crate) fn toy_map(nl: u32, ne: u32) -> ExpertMap {
        let mut h = mp_gguf::GgufHeader {
            version: 3,
            alignment: 32,
            metadata: Default::default(),
            tensors: vec![],
            data_offset: 0,
        };
        h.metadata.insert(
            "general.architecture".into(),
            mp_gguf::MetaValue::Str("toy".into()),
        );
        let mut m = ExpertMap::from_header(&h, 4096, None);
        m.n_layers = nl;
        m.n_experts = ne;
        m.top_k = Some(2);
        let slice = 64 << 10;
        for l in 0..nl {
            for e in 0..ne {
                let base = ((l * ne + e) as u64) * 3 * slice;
                m.units.push(Unit {
                    layer: l,
                    expert: e,
                    bytes: 3 * slice,
                    slices: (0..3)
                        .map(|k| Slice {
                            kind: ["up", "gate", "down"][k as usize].into(),
                            tensor: format!("blk.{l}.t{k}"),
                            offset: base + k * slice,
                            len: slice,
                        })
                        .collect(),
                });
            }
        }
        m.reindex();
        m
    }

    fn ev(token: u32, layer: u16, expert: u16) -> ExpertEvent {
        ExpertEvent {
            t_ns: (token as u64 * 10 + layer as u64) * 1_000_000,
            token,
            layer,
            expert,
        }
    }

    #[test]
    fn completion_prefetches_all_slices_of_a_missed_unit() {
        let cfg = DaemonConfig {
            policy: Policy::None,
            ..Default::default()
        };
        let mut d = Daemon::new(toy_map(2, 4), cfg, MockOps::new(4096));
        d.on_event(&ev(0, 1, 2));
        let base = (4 + 2) * 3 * (64 << 10);
        assert_eq!(
            d.ops.calls,
            vec![
                OpCall::Prefetch(base, 65536),
                OpCall::Prefetch(base + 65536, 65536),
                OpCall::Prefetch(base + 131072, 65536)
            ]
        );
    }

    #[test]
    fn infers_token_boundaries() {
        let cfg = DaemonConfig {
            policy: Policy::None,
            token_slack: 0,
            ..Default::default()
        };
        let mut d = Daemon::new(toy_map(3, 4), cfg, MockOps::new(4096));
        for t in 0..5 {
            for l in 0..3 {
                d.on_event(&ev(t, l, 1));
            }
        }
        assert_eq!(d.counters.tokens, 4); // the 5th token is still open
    }

    #[test]
    fn v_policy_pins_hot_units_within_budget() {
        let unit = 3 * (64u64 << 10);
        let budget = 4 * unit;
        let cfg = DaemonConfig {
            policy: Policy::V,
            pin_budget_bytes: budget,
            token_slack: 0,
            residency: ResidencyConfig {
                replan_every: 1,
                explore_frac: 0.1,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut d = Daemon::new(toy_map(2, 8), cfg, MockOps::new(4096));
        let pinned = |d: &Daemon<MockOps>, l: u64, e: u64| {
            d.ops.pinned.contains(&((l * 8 + e) * unit / 4096))
        };
        // Experts 0 and 1 are used every token in both layers, others rarely.
        // Black-box reality: a pinned unit never misses, so it is never observed.
        for t in 0..400u32 {
            for l in 0..2u16 {
                let mut used = vec![0u16, 1];
                if t % 10 == 0 {
                    used.push((2 + t % 6) as u16);
                }
                for e in used {
                    if !pinned(&d, l as u64, e as u64) {
                        d.on_event(&ev(t, l, e));
                    }
                }
            }
        }
        assert!(d.pinned_bytes() <= budget);
        assert!(d.ops.pinned_bytes() <= budget);
        let hot = [(0, 0), (0, 1), (1, 0), (1, 1)]
            .iter()
            .filter(|&&(l, e)| pinned(&d, l, e))
            .count();
        // Exploration may have just released one hot unit.
        assert!(hot >= 3, "only {hot} hot units pinned: {:?}", d.counters);
        assert!(
            d.counters.unpin_ops > 0,
            "exploration should have released something"
        );
    }

    #[test]
    fn pin_failure_clamps_budget() {
        let mut ops = MockOps::new(4096);
        ops.budget_pin_bytes = 3 * (64 << 10); // room for one unit only
        let cfg = DaemonConfig {
            policy: Policy::LruPin,
            pin_budget_bytes: 10 * 3 * (64 << 10),
            ..Default::default()
        };
        let mut d = Daemon::new(toy_map(1, 8), cfg, ops);
        for e in 0..4 {
            d.on_event(&ev(0, 0, e));
        }
        assert!(d.counters.budget_clamps >= 1);
        assert!(d.ops.pinned_bytes() <= 3 * (64 << 10));
        assert_eq!(d.cfg.pin_budget_bytes, d.pinned_bytes());
    }

    #[test]
    fn willneed_all_prefetches_next_layer() {
        let cfg = DaemonConfig {
            policy: Policy::WillneedAll,
            prefetch: PrefetchConfig {
                completion: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut d = Daemon::new(toy_map(2, 4), cfg, MockOps::new(4096));
        d.on_event(&ev(0, 0, 0));
        assert_eq!(d.counters.prefetch_ops, 4);
        d.on_event(&ev(0, 1, 0)); // last layer: nothing further
        assert_eq!(d.counters.prefetch_ops, 4);
    }
}
