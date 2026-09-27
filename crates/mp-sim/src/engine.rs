//! Simulation engine: byte-capacity cache, single FIFO I/O channel, stall
//! accounting, and a controller hook through which daemon-like policies
//! (pinning, prefetch) act. See ARCHITECTURE.md §7 for the model.

use mp_core::{Action, CostModel};
use mp_trace::Steps;
use serde::{Deserialize, Serialize};

use crate::evict::Evictor;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SimConfig {
    /// Bytes available for expert units (budget minus dense weights).
    pub capacity_bytes: u64,
    pub cost: CostModel,
    /// Demand misses complete the whole unit in bulk (completion readahead).
    pub completion: bool,
    /// Compute time per layer (ns), excluding I/O stalls.
    pub t_layer_ns: f64,
    /// Per-token time outside the MoE layers (ns).
    pub token_overhead_ns: f64,
    /// Tokens simulated but excluded from metrics.
    pub warmup_tokens: u32,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            capacity_bytes: 0,
            cost: CostModel::default(),
            completion: false,
            t_layer_ns: 800_000.0,
            token_overhead_ns: 2_000_000.0,
            warmup_tokens: 0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    pub tokens: u64,
    pub accesses: u64,
    pub hits: u64,
    pub misses: u64,
    /// Hits on units whose prefetch had not finished (partial stall).
    pub late_prefetch_hits: u64,
    pub demand_bytes: u64,
    pub prefetch_bytes: u64,
    /// Prefetched bytes evicted before any use.
    pub wasted_prefetch_bytes: u64,
    pub stall_ns: f64,
    pub compute_ns: f64,
    pub pin_actions: u64,
    /// Accesses that could not be cached because everything was pinned.
    pub bypassed: u64,
}

impl Metrics {
    pub fn miss_ratio(&self) -> f64 {
        ratio(self.misses as f64, self.accesses as f64)
    }
    pub fn ssd_bytes(&self) -> u64 {
        self.demand_bytes + self.prefetch_bytes
    }
    pub fn ssd_bytes_per_token(&self) -> f64 {
        ratio(self.ssd_bytes() as f64, self.tokens as f64)
    }
    pub fn stall_ms_per_token(&self) -> f64 {
        ratio(self.stall_ns, self.tokens as f64) / 1e6
    }
    pub fn tokens_per_s(&self) -> f64 {
        ratio(self.tokens as f64 * 1e9, self.stall_ns + self.compute_ns)
    }
}

fn ratio(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        0.0
    } else {
        a / b
    }
}

/// Read-only view of the cache handed to controllers.
pub struct View<'a> {
    pub resident: &'a [bool],
    pub pinned: &'a [bool],
}

/// Daemon-like policy logic driven by the simulator.
pub trait Controller {
    fn name(&self) -> String;
    /// Actions before the first token (e.g. static pins).
    fn init(&mut self, _sizes: &[u64]) -> Vec<Action> {
        Vec::new()
    }
    /// Every access, with its true hit/miss outcome. Controllers that model
    /// black-box observation must drop hits themselves.
    fn on_access(&mut self, _layer: u16, _expert: u16, _unit: u32, _hit: bool, _t_ns: u64) {}
    /// After all accesses of `layer` in the current token.
    fn after_layer(&mut self, _layer: u16, _view: &View) -> Vec<Action> {
        Vec::new()
    }
    /// End of a token.
    fn end_token(&mut self, _view: &View) -> Vec<Action> {
        Vec::new()
    }
}

/// No controller: the kernel's eviction order alone.
pub struct NoControl;

impl Controller for NoControl {
    fn name(&self) -> String {
        "none".into()
    }
}

struct State<'a> {
    cfg: &'a SimConfig,
    sizes: &'a [u64],
    ev: &'a mut dyn Evictor,
    resident: Vec<bool>,
    pinned: Vec<bool>,
    ready_at: Vec<f64>,
    unused_prefetch: Vec<bool>,
    used: u64,
    now: f64,
    chan_free: f64,
    m: Metrics,
    counting: bool,
}

impl State<'_> {
    fn evict_one(&mut self) -> bool {
        let Some(v) = self.ev.victim() else {
            return false;
        };
        let i = v as usize;
        self.resident[i] = false;
        self.used -= self.sizes[i];
        if self.unused_prefetch[i] {
            self.unused_prefetch[i] = false;
            if self.counting {
                self.m.wasted_prefetch_bytes += self.sizes[i];
            }
        }
        true
    }

    fn make_room(&mut self, size: u64) -> bool {
        while self.used + size > self.cfg.capacity_bytes {
            if !self.evict_one() {
                return false;
            }
        }
        true
    }

    /// Queue a read on the channel; returns its completion time.
    fn io(&mut self, dur_ns: f64) -> f64 {
        let start = self.now.max(self.chan_free);
        self.chan_free = start + dur_ns;
        self.chan_free
    }

    fn load_bulk(&mut self, u: u32, pos: usize) -> bool {
        let i = u as usize;
        if self.resident[i] {
            return true;
        }
        if !self.make_room(self.sizes[i]) {
            return false;
        }
        let done = self.io(self.cfg.cost.bulk_ns(self.sizes[i]));
        self.resident[i] = true;
        self.used += self.sizes[i];
        self.ready_at[i] = done;
        self.unused_prefetch[i] = true;
        if self.counting {
            self.m.prefetch_bytes += self.sizes[i];
        }
        if !self.pinned[i] {
            self.ev.insert(u, pos, false);
        }
        true
    }

    fn apply(&mut self, actions: Vec<Action>, pos: usize) {
        for a in actions {
            match a {
                Action::Prefetch(u) => {
                    self.load_bulk(u, pos);
                }
                Action::Pin(u) => {
                    let i = u as usize;
                    if self.pinned[i] {
                        continue;
                    }
                    if self.counting {
                        self.m.pin_actions += 1;
                    }
                    // mlock faults the unit in if needed.
                    if self.load_bulk(u, pos) {
                        self.pinned[i] = true;
                        self.ev.remove(u);
                    }
                }
                Action::Unpin(u) => {
                    let i = u as usize;
                    if !self.pinned[i] {
                        continue;
                    }
                    if self.counting {
                        self.m.pin_actions += 1;
                    }
                    self.pinned[i] = false;
                    if self.resident[i] {
                        self.ev.insert(u, pos, false);
                    }
                }
                Action::Demote(u) => {
                    // MADV_COLD: modelled as immediate eviction of an unpinned unit.
                    let i = u as usize;
                    if self.resident[i] && !self.pinned[i] {
                        self.ev.remove(u);
                        self.resident[i] = false;
                        self.used -= self.sizes[i];
                        self.unused_prefetch[i] = false;
                    }
                }
            }
        }
    }

    fn access(&mut self, u: u32, pos: usize) -> bool {
        let i = u as usize;
        let size = self.sizes[i];
        if self.counting {
            self.m.accesses += 1;
        }
        if self.resident[i] {
            if self.ready_at[i] > self.now {
                if self.counting {
                    self.m.late_prefetch_hits += 1;
                    self.m.stall_ns += self.ready_at[i] - self.now;
                }
                self.now = self.ready_at[i];
            }
            self.unused_prefetch[i] = false;
            if !self.pinned[i] {
                self.ev.touch(u, pos);
            }
            if self.counting {
                self.m.hits += 1;
            }
            return true;
        }
        let done = self.io(self.cfg.cost.demand_ns(size, self.cfg.completion));
        if self.counting {
            self.m.misses += 1;
            self.m.demand_bytes += size;
            self.m.stall_ns += done - self.now;
        }
        self.now = done;
        if self.make_room(size) {
            self.resident[i] = true;
            self.used += size;
            self.ready_at[i] = 0.0;
            self.ev.insert(u, pos, true);
        } else if self.counting {
            self.m.bypassed += 1;
        }
        false
    }
}

/// Run one simulation. `sizes[u]` are unit sizes indexed by dense unit id.
pub fn simulate(
    steps: &Steps,
    sizes: &[u64],
    cfg: &SimConfig,
    ev: &mut dyn Evictor,
    ctl: &mut dyn Controller,
) -> Metrics {
    let n = sizes.len();
    assert!(
        n >= steps.n_units() as usize,
        "sizes shorter than the unit space"
    );
    let mut st = State {
        cfg,
        sizes,
        ev,
        resident: vec![false; n],
        pinned: vec![false; n],
        ready_at: vec![0.0; n],
        unused_prefetch: vec![false; n],
        used: 0,
        now: 0.0,
        chan_free: 0.0,
        m: Metrics::default(),
        counting: cfg.warmup_tokens == 0,
    };
    let init = ctl.init(sizes);
    st.apply(init, 0);
    let mut pos = 0usize;
    for (ti, step) in steps.steps.iter().enumerate() {
        st.counting = ti as u32 >= cfg.warmup_tokens;
        for l in &step.layers {
            for &e in &l.experts {
                let u = steps.unit(l.layer, e);
                let hit = st.access(u, pos);
                ctl.on_access(l.layer, e, u, hit, st.now as u64);
                pos += 1;
            }
            let acts = ctl.after_layer(
                l.layer,
                &View {
                    resident: &st.resident,
                    pinned: &st.pinned,
                },
            );
            st.apply(acts, pos);
            st.now += cfg.t_layer_ns;
            if st.counting {
                st.m.compute_ns += cfg.t_layer_ns;
            }
        }
        st.now += cfg.token_overhead_ns;
        if st.counting {
            st.m.compute_ns += cfg.token_overhead_ns;
            st.m.tokens += 1;
        }
        let acts = ctl.end_token(&View {
            resident: &st.resident,
            pinned: &st.pinned,
        });
        st.apply(acts, pos);
        debug_assert!(st.used <= cfg.capacity_bytes);
    }
    st.m
}
