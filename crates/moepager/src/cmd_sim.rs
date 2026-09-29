//! `sim`: policy × capacity sweeps over an expert trace.

use std::path::PathBuf;

use anyhow::{bail, Result};
use mp_core::CostModel;
use mp_sim::{PolicyOptions, SimConfig, POLICIES};
use mp_trace::{Observation, Steps};

use crate::common::{load_map, unit_sizes};

#[derive(clap::Args)]
pub struct Args {
    /// Ground-truth expert trace (observation = full).
    pub trace: PathBuf,
    #[arg(long)]
    pub map: Option<PathBuf>,
    #[arg(long, default_value_t = 1 << 20)]
    pub unit_bytes: u64,
    /// Comma-separated policies (default: all).
    #[arg(long, value_delimiter = ',')]
    pub policies: Vec<String>,
    /// Capacities as fractions of total expert bytes.
    #[arg(long, value_delimiter = ',', default_value = "0.15,0.25,0.35,0.5,0.7")]
    pub caps: Vec<f64>,
    /// Expert-completion readahead: off, on or both.
    #[arg(long, default_value = "both")]
    pub completion: String,
    /// Fault-driven bandwidth of touched bytes (default: dev-laptop smoke result).
    #[arg(long, default_value_t = 0.45)]
    pub demand_gbps: f64,
    /// Bulk (WILLNEED) bandwidth (default: dev-laptop smoke result).
    #[arg(long, default_value_t = 2.0)]
    pub bulk_gbps: f64,
    #[arg(long, default_value_t = 100.0)]
    pub t_fault_us: f64,
    #[arg(long, default_value_t = 200.0)]
    pub detect_us: f64,
    /// Compute per layer; default = median layer gap in the trace.
    #[arg(long)]
    pub t_layer_us: Option<f64>,
    #[arg(long, default_value_t = 2000.0)]
    pub token_overhead_us: f64,
    #[arg(long, default_value_t = 32)]
    pub warmup: u32,
    #[arg(long, default_value_t = 0.6)]
    pub pin_frac: f64,
    #[arg(long, default_value_t = 64.0)]
    pub half_life: f64,
    #[arg(long)]
    pub csv: Option<PathBuf>,
    #[arg(long)]
    pub md: Option<PathBuf>,
}

pub fn run(a: Args) -> Result<()> {
    let (h, ev) = mp_trace::read_expert_file(&a.trace)?;
    if h.observation != Observation::Full {
        eprintln!(
            "warning: trace observation is {:?}; the simulator needs ground truth, so \
             miss-only traces understate reuse",
            h.observation
        );
    }
    let steps = Steps::from_events(&h, &ev);
    let map = a.map.as_deref().map(load_map).transpose()?;
    let sizes = unit_sizes(map.as_ref(), a.unit_bytes, h.n_layers, h.n_experts)?;
    let total: u64 = sizes.iter().sum();
    let policies: Vec<&str> = if a.policies.is_empty() {
        POLICIES.to_vec()
    } else {
        a.policies.iter().map(String::as_str).collect()
    };
    let completion: Vec<bool> = match a.completion.as_str() {
        "off" => vec![false],
        "on" => vec![true],
        "both" => vec![false, true],
        x => bail!("--completion must be off, on or both (got {x})"),
    };
    let t_layer_ns = match a.t_layer_us {
        Some(us) => us * 1e3,
        None => {
            let t = mp_analyze::stats::timing(&steps).layer_gap_p50 as f64;
            if t > 0.0 {
                t
            } else {
                800_000.0
            }
        }
    };
    let base = SimConfig {
        capacity_bytes: 0,
        cost: CostModel {
            t_fault_ns: a.t_fault_us * 1e3,
            demand_bw: a.demand_gbps * 1e9,
            bulk_bw: a.bulk_gbps * 1e9,
            detect_ns: a.detect_us * 1e3,
        },
        completion: false,
        t_layer_ns,
        token_overhead_ns: a.token_overhead_us * 1e3,
        warmup_tokens: a.warmup,
    };
    let opt = PolicyOptions {
        pin_frac: a.pin_frac,
        half_life_tokens: a.half_life,
        ..Default::default()
    };
    let caps: Vec<u64> = a.caps.iter().map(|f| (total as f64 * f) as u64).collect();
    let res = mp_sim::sweep(&policies, &caps, &completion, &steps, &sizes, &base, &opt)
        .map_err(anyhow::Error::msg)?;
    println!(
        "trace={} tokens={} (warmup {}) expert_bytes={:.3}GB t_layer={:.3}ms demand={}GB/s bulk={}GB/s",
        a.trace.display(),
        steps.steps.len(),
        a.warmup,
        total as f64 / 1e9,
        t_layer_ns / 1e6,
        a.demand_gbps,
        a.bulk_gbps
    );
    let md = mp_sim::markdown(&res, total);
    print!("{md}");
    if let Some(p) = a.md {
        std::fs::write(p, &md)?;
    }
    if let Some(p) = a.csv {
        std::fs::write(p, mp_sim::csv(&res))?;
    }
    Ok(())
}
