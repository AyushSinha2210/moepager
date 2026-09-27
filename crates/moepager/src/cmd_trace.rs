//! `synth`, `analyze` and `trace-csv`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use mp_synth::SynthParams;
use mp_trace::Steps;

use crate::common::{gb, load_map, unit_sizes};

#[derive(clap::Args)]
pub struct SynthArgs {
    /// Output trace (.mpt).
    #[arg(short, long)]
    pub out: PathBuf,
    /// Start from parameters in a JSON file (fields of SynthParams).
    #[arg(long)]
    pub params: Option<PathBuf>,
    /// Take layers/experts/top-k from an expert map (map.json).
    #[arg(long)]
    pub like: Option<PathBuf>,
    #[arg(long)]
    pub layers: Option<u32>,
    #[arg(long)]
    pub experts: Option<u32>,
    #[arg(long)]
    pub top_k: Option<u32>,
    #[arg(long)]
    pub tokens: Option<u32>,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long)]
    pub skew: Option<f64>,
    #[arg(long)]
    pub reuse: Option<f64>,
    #[arg(long)]
    pub affinity: Option<f64>,
    #[arg(long)]
    pub drift_every: Option<u32>,
    #[arg(long)]
    pub t_layer_us: Option<u64>,
}

pub fn synth(a: SynthArgs) -> Result<()> {
    let mut p: SynthParams = match &a.params {
        Some(f) => serde_json::from_slice(&std::fs::read(f)?).context("params json")?,
        None => SynthParams::default(),
    };
    if let Some(m) = &a.like {
        let m = load_map(m)?;
        p.n_layers = m.n_layers;
        p.n_experts = m.n_experts;
        p.top_k = m.top_k.unwrap_or(p.top_k);
    }
    macro_rules! set {
        ($f:ident, $v:expr) => {
            if let Some(v) = $v {
                p.$f = v;
            }
        };
    }
    set!(n_layers, a.layers);
    set!(n_experts, a.experts);
    set!(top_k, a.top_k);
    set!(tokens, a.tokens);
    set!(seed, a.seed);
    set!(skew, a.skew);
    set!(reuse, a.reuse);
    set!(affinity, a.affinity);
    set!(drift_every, a.drift_every);
    set!(t_layer_ns, a.t_layer_us.map(|u| u * 1000));
    p.validate().map_err(anyhow::Error::msg)?;
    let (h, ev) = mp_synth::generate(&p);
    mp_trace::write_expert_file(&a.out, &h, &ev)?;
    println!(
        "wrote {} events ({} tokens x {} layers x top-{}) to {}",
        ev.len(),
        p.tokens,
        p.n_layers,
        p.top_k,
        a.out.display()
    );
    Ok(())
}

#[derive(clap::Args)]
pub struct AnalyzeArgs {
    pub trace: PathBuf,
    /// Expert map for real unit sizes.
    #[arg(long)]
    pub map: Option<PathBuf>,
    /// Uniform unit size in bytes when no map is given.
    #[arg(long, default_value_t = 1 << 20)]
    pub unit_bytes: u64,
    /// Write the full report as JSON.
    #[arg(long)]
    pub json: Option<PathBuf>,
}

pub fn analyze(a: AnalyzeArgs) -> Result<()> {
    let (h, ev) = mp_trace::read_expert_file(&a.trace)?;
    let s = Steps::from_events(&h, &ev);
    let map = a.map.as_deref().map(load_map).transpose()?;
    let sizes = unit_sizes(map.as_ref(), a.unit_bytes, h.n_layers, h.n_experts)?;
    let r = mp_analyze::analyze(&s, Some(&sizes), &mp_analyze::default_fractions());
    println!(
        "source={} observation={:?} tokens={} accesses={}",
        h.source, h.observation, r.tokens, r.accesses
    );
    println!(
        "footprint={}GB cold_misses={} median_reuse_units={:?}",
        gb(r.footprint_bytes as f64),
        r.cold_misses,
        r.median_reuse_units
    );
    println!(
        "token_reuse={:.3} top10_share={:.3} norm_entropy={:.3} units_touched={:.3}",
        r.token_reuse.mean,
        r.popularity.top10_share,
        r.popularity.norm_entropy,
        r.popularity.units_touched
    );
    let p = &r.predictability;
    println!(
        "next-layer recall@k: transitions={:.3} popularity={:.3} prev-token={:.3} (concentration top4={:.3})",
        p.transition_recall, p.popularity_recall, p.reuse_recall, p.transition_concentration
    );
    println!(
        "layer gap p10/p50/p90 = {:.3}/{:.3}/{:.3} ms; token p50 = {:.3} ms",
        r.timing.layer_gap_p10 as f64 / 1e6,
        r.timing.layer_gap_p50 as f64 / 1e6,
        r.timing.layer_gap_p90 as f64 / 1e6,
        r.timing.token_p50 as f64 / 1e6
    );
    println!("LRU miss-ratio curve (capacity as % of footprint):");
    for (i, m) in r.lru_mrc.iter().enumerate() {
        if i % 2 == 1 {
            println!(
                "  {:>4.0}%  cap={}GB  miss_ratio={:.3}  miss_bytes/token={}GB",
                100.0 * m.capacity_bytes as f64 / r.footprint_bytes.max(1) as f64,
                gb(m.capacity_bytes as f64),
                m.miss_ratio,
                gb(m.miss_bytes as f64 / r.tokens.max(1) as f64)
            );
        }
    }
    if let Some(j) = a.json {
        std::fs::write(&j, serde_json::to_vec_pretty(&r)?)?;
        eprintln!("wrote {}", j.display());
    }
    Ok(())
}

#[derive(clap::Args)]
pub struct CsvArgs {
    pub trace: PathBuf,
}

pub fn csv(a: CsvArgs) -> Result<()> {
    let f = std::io::BufReader::new(std::fs::File::open(&a.trace)?);
    let out = std::io::BufWriter::new(std::io::stdout().lock());
    match mp_trace::to_csv(f, out) {
        // `moepager trace-csv x | head` closes the pipe early; that's fine.
        Err(mp_trace::TraceError::Io(e)) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        r => r.map(|_| ()).map_err(Into::into),
    }
}
