//! Controllers: static pinning baselines and the mp-core V(e) policy.

use mp_core::{
    Action, CostModel, OnlineStats, PrefetchConfig, PrefetchPlanner, ResidencyConfig,
    ResidencyEngine, Seen,
};
use mp_trace::{Observation, Steps};

use crate::engine::{Controller, View};

/// Pin a fixed set of units at start-up.
pub struct StaticPin {
    name: String,
    units: Vec<u32>,
}

impl StaticPin {
    /// Pin whole layers from layer 0 upward until the budget is full: the
    /// classic defence of a looping scan (keep a stable prefix, not the
    /// most recent tail).
    pub fn prefix(sizes: &[u64], budget: u64) -> Self {
        let mut used = 0;
        let units = (0..sizes.len() as u32)
            .take_while(|&u| {
                used += sizes[u as usize];
                used <= budget
            })
            .collect();
        StaticPin {
            name: "prefix-pin".into(),
            units,
        }
    }

    /// Oracle: pin the units with the most accesses per byte over the whole
    /// trace (it sees the future; an upper reference for static pinning).
    pub fn frequency_oracle(steps: &Steps, sizes: &[u64], budget: u64) -> Self {
        let mut count = vec![0u64; sizes.len()];
        for st in &steps.steps {
            for l in &st.layers {
                for &e in &l.experts {
                    count[steps.unit(l.layer, e) as usize] += 1;
                }
            }
        }
        let mut order: Vec<u32> = (0..sizes.len() as u32)
            .filter(|&u| count[u as usize] > 0)
            .collect();
        order.sort_by(|&a, &b| {
            let va = count[a as usize] as f64 / sizes[a as usize].max(1) as f64;
            let vb = count[b as usize] as f64 / sizes[b as usize].max(1) as f64;
            vb.total_cmp(&va).then(a.cmp(&b))
        });
        let mut used = 0;
        let mut units = Vec::new();
        for u in order {
            let s = sizes[u as usize];
            if used + s <= budget {
                used += s;
                units.push(u);
            }
        }
        StaticPin {
            name: "static-freq*".into(),
            units,
        }
    }
}

impl Controller for StaticPin {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn init(&mut self, _sizes: &[u64]) -> Vec<Action> {
        self.units.iter().map(|&u| Action::Pin(u)).collect()
    }
}

/// The moepager policy: OnlineStats → ResidencyEngine (+ PrefetchPlanner),
/// exactly the code the daemon runs.
pub struct VControl {
    stats: OnlineStats,
    engine: ResidencyEngine,
    planner: PrefetchPlanner,
    cost: CostModel,
}

impl VControl {
    pub fn new(
        steps: &Steps,
        sizes: &[u64],
        regime: Observation,
        half_life_tokens: f64,
        residency: ResidencyConfig,
        prefetch: PrefetchConfig,
        cost: CostModel,
    ) -> Self {
        let prior = 1.0 / steps.n_experts.max(1) as f64;
        VControl {
            stats: OnlineStats::new(
                steps.n_layers,
                steps.n_experts,
                regime,
                half_life_tokens,
                prior,
            ),
            engine: ResidencyEngine::new(residency, sizes.to_vec()),
            planner: PrefetchPlanner::new(prefetch, sizes.to_vec(), steps.n_experts),
            cost,
        }
    }
}

impl Controller for VControl {
    fn name(&self) -> String {
        let r = match self.stats.regime {
            Observation::Full => "full",
            Observation::MissOnly => "miss",
        };
        let mut n = format!("v-{r}");
        if self.engine.cfg.pin_budget_bytes == 0 {
            n = format!("stats-{r}");
        }
        if self.planner.cfg.predict {
            n.push_str("+predict");
        }
        n
    }
    fn on_access(&mut self, layer: u16, expert: u16, _unit: u32, hit: bool, t_ns: u64) {
        let seen = if hit { Seen::Hit } else { Seen::Miss };
        self.stats.observe(layer, expert, seen, t_ns);
    }
    fn after_layer(&mut self, layer: u16, view: &View) -> Vec<Action> {
        self.planner
            .after_layer(&self.stats, layer, &self.cost, &|u| {
                view.resident[u as usize]
            })
    }
    fn end_token(&mut self, _view: &View) -> Vec<Action> {
        let engine = &self.engine;
        self.stats.end_token(&|u| engine.is_pinned(u));
        self.engine.on_token(&self.stats, &self.cost)
    }
}
