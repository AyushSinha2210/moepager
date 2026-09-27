//! Named policies and parameter sweeps.

use mp_core::{PrefetchConfig, ResidencyConfig};
use mp_trace::{Observation, Steps};
use serde::{Deserialize, Serialize};

use crate::control::{StaticPin, VControl};
use crate::engine::{simulate, Controller, Metrics, NoControl, SimConfig};
use crate::evict::{Belady, Evictor, Lfu, Lru};

/// All policy names accepted by [`run_policy`].
pub const POLICIES: &[&str] = &[
    "lru",
    "lfu",
    "belady",
    "prefix-pin",
    "static-freq",
    "v-full",
    "v-miss",
    "v-miss+predict",
    "predict-miss",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PolicyOptions {
    /// Fraction of the cache that pinning policies may pin.
    pub pin_frac: f64,
    pub half_life_tokens: f64,
    pub residency: ResidencyConfig,
    pub prefetch: PrefetchConfig,
}

impl Default for PolicyOptions {
    fn default() -> Self {
        PolicyOptions {
            pin_frac: 0.6,
            half_life_tokens: 64.0,
            residency: ResidencyConfig::default(),
            prefetch: PrefetchConfig {
                completion: false,
                ..Default::default()
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RunResult {
    pub policy: String,
    pub capacity_bytes: u64,
    pub completion: bool,
    pub metrics: Metrics,
}

fn access_seq(steps: &Steps) -> Vec<u32> {
    let mut v = Vec::with_capacity(steps.n_accesses());
    for s in &steps.steps {
        for l in &s.layers {
            for &e in &l.experts {
                v.push(steps.unit(l.layer, e));
            }
        }
    }
    v
}

/// Run one named policy.
pub fn run_policy(
    name: &str,
    steps: &Steps,
    sizes: &[u64],
    cfg: &SimConfig,
    opt: &PolicyOptions,
) -> Result<RunResult, String> {
    let n = sizes.len();
    let pin_budget = (cfg.capacity_bytes as f64 * opt.pin_frac) as u64;
    let mut ev: Box<dyn Evictor> = match name {
        "lfu" => Box::new(Lfu::new(n)),
        "belady" => Box::new(Belady::new(n, &access_seq(steps))),
        _ => Box::new(Lru::new(n)),
    };
    let v = |regime, predict: bool, budget: u64| {
        let mut r = opt.residency.clone();
        r.pin_budget_bytes = budget;
        let mut p = opt.prefetch.clone();
        p.predict = predict;
        p.default_t_layer_ns = cfg.t_layer_ns;
        VControl::new(steps, sizes, regime, opt.half_life_tokens, r, p, cfg.cost)
    };
    let mut ctl: Box<dyn Controller> = match name {
        "lru" | "lfu" | "belady" => Box::new(NoControl),
        "prefix-pin" => Box::new(StaticPin::prefix(sizes, pin_budget)),
        "static-freq" => Box::new(StaticPin::frequency_oracle(steps, sizes, pin_budget)),
        "v-full" => Box::new(v(Observation::Full, false, pin_budget)),
        "v-miss" => Box::new(v(Observation::MissOnly, false, pin_budget)),
        "v-miss+predict" => Box::new(v(Observation::MissOnly, true, pin_budget)),
        "predict-miss" => Box::new(v(Observation::MissOnly, true, 0)),
        other => {
            return Err(format!(
                "unknown policy {other:?}; known: {}",
                POLICIES.join(", ")
            ))
        }
    };
    let metrics = simulate(steps, sizes, cfg, ev.as_mut(), ctl.as_mut());
    let policy = match name {
        "belady" => "belady*".to_string(),
        "lru" | "lfu" => name.to_string(),
        _ => ctl.name(),
    };
    Ok(RunResult {
        policy,
        capacity_bytes: cfg.capacity_bytes,
        completion: cfg.completion,
        metrics,
    })
}

/// Policies × capacities (and completion on/off).
pub fn sweep(
    policies: &[&str],
    capacities: &[u64],
    completion: &[bool],
    steps: &Steps,
    sizes: &[u64],
    base: &SimConfig,
    opt: &PolicyOptions,
) -> Result<Vec<RunResult>, String> {
    let mut out = Vec::new();
    for &c in capacities {
        for &comp in completion {
            for p in policies {
                let cfg = SimConfig {
                    capacity_bytes: c,
                    completion: comp,
                    ..base.clone()
                };
                out.push(run_policy(p, steps, sizes, &cfg, opt)?);
            }
        }
    }
    Ok(out)
}

/// Markdown table of sweep results. `footprint` scales the capacity column.
pub fn markdown(results: &[RunResult], footprint: u64) -> String {
    let mut s = String::from(
        "| capacity | completion | policy | miss ratio | SSD GB/token | wasted prefetch GB/token | stall ms/token | est tok/s | pin ops/token |\n\
         |---|---|---|---|---|---|---|---|---|\n",
    );
    for r in results {
        let m = &r.metrics;
        let t = m.tokens.max(1) as f64;
        s.push_str(&format!(
            "| {:.0}% ({:.2} GB) | {} | {} | {:.3} | {:.3} | {:.3} | {:.1} | {:.2} | {:.1} |\n",
            100.0 * r.capacity_bytes as f64 / footprint.max(1) as f64,
            r.capacity_bytes as f64 / 1e9,
            if r.completion { "on" } else { "off" },
            r.policy,
            m.miss_ratio(),
            m.ssd_bytes_per_token() / 1e9,
            m.wasted_prefetch_bytes as f64 / t / 1e9,
            m.stall_ms_per_token(),
            m.tokens_per_s(),
            m.pin_actions as f64 / t,
        ));
    }
    s
}

/// CSV of sweep results.
pub fn csv(results: &[RunResult]) -> String {
    let mut s = String::from(
        "policy,capacity_bytes,completion,tokens,accesses,hits,misses,late_prefetch_hits,demand_bytes,prefetch_bytes,wasted_prefetch_bytes,stall_ns,compute_ns,pin_actions,bypassed\n",
    );
    for r in results {
        let m = &r.metrics;
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{:.0},{:.0},{},{}\n",
            r.policy,
            r.capacity_bytes,
            r.completion,
            m.tokens,
            m.accesses,
            m.hits,
            m.misses,
            m.late_prefetch_hits,
            m.demand_bytes,
            m.prefetch_bytes,
            m.wasted_prefetch_bytes,
            m.stall_ns,
            m.compute_ns,
            m.pin_actions,
            m.bypassed
        ));
    }
    s
}
