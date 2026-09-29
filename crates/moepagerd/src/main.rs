//! moepagerd: expert-aware page-cache daemon for an unmodified MoE engine.
//!
//! Status: skeleton. `--ops mock` dry runs are tested; `--ops linux` with
//! `--source sentinel` against a live llama.cpp is UNTESTED-ON-HW.

use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;
use moepagerd::source::{EventSource, SentinelSource, TraceSource};
use moepagerd::{Daemon, DaemonConfig, Policy};
use mp_gguf::{parse_file, ExpertMap};
use mp_os::{
    find_file_mapping, memlock_limit, LinuxOps, MappedFile, MockOps, PageCacheOps, TargetMapping,
};

#[derive(Parser)]
#[command(name = "moepagerd", version, about)]
struct Args {
    /// Model file used by the engine.
    gguf: PathBuf,
    /// none | willneed-all | lru-pin | v
    #[arg(long, default_value = "v")]
    policy: Policy,
    /// Pin budget in MiB.
    #[arg(long, default_value_t = 0)]
    budget_mb: u64,
    /// `sentinel` (live, no root) or `trace:<file.mpt>` (dry run).
    #[arg(long, default_value = "sentinel")]
    source: String,
    /// `mock` (log only) or `linux` (real fadvise/mlock/process_madvise).
    #[arg(long, default_value = "mock")]
    ops: String,
    /// Engine pid, enables MADV_COLD demotion (needs CAP_SYS_NICE).
    #[arg(long)]
    pid: Option<u32>,
    #[arg(long, default_value_t = 500)]
    interval_us: u64,
    #[arg(long)]
    duration_s: Option<f64>,
    /// Disable expert-completion readahead.
    #[arg(long)]
    no_completion: bool,
    /// Enable cross-layer predictive prefetch (V policy).
    #[arg(long)]
    predict: bool,
    /// Write final counters as JSON.
    #[arg(long)]
    stats_json: Option<PathBuf>,
}

fn run<O: PageCacheOps>(
    map: ExpertMap,
    cfg: DaemonConfig,
    ops: O,
    mut src: Box<dyn EventSource>,
) -> Result<Daemon<O>> {
    let mut d = Daemon::new(map, cfg, ops);
    while let Some(batch) = src.poll()? {
        for e in &batch {
            d.on_event(e);
        }
    }
    Ok(d)
}

fn main() -> Result<()> {
    let a = Args::parse();
    let h = parse_file(&a.gguf).with_context(|| format!("parsing {}", a.gguf.display()))?;
    let map = ExpertMap::from_header(&h, 4096, std::fs::metadata(&a.gguf).ok().map(|m| m.len()));
    if map.units.is_empty() {
        bail!("no expert tensors in {}", a.gguf.display());
    }
    let mut cfg = DaemonConfig {
        policy: a.policy,
        pin_budget_bytes: a.budget_mb << 20,
        ..Default::default()
    };
    cfg.prefetch.completion = !a.no_completion;
    cfg.prefetch.predict = a.predict;
    cfg.token_slack = (map.n_layers / 8) as u16;
    let src: Box<dyn EventSource> = match a.source.split_once(':') {
        Some(("trace", p)) => Box::new(TraceSource::open(p.as_ref())?),
        None if a.source == "sentinel" => Box::new(SentinelSource::open(
            &a.gguf,
            &map,
            Duration::from_micros(a.interval_us),
            a.duration_s.map(Duration::from_secs_f64),
        )?),
        _ => bail!("--source must be sentinel or trace:<file>"),
    };
    eprintln!(
        "moepagerd: {} | policy {:?} | pin budget {} MiB | ops {}",
        map.summary(),
        cfg.policy,
        a.budget_mb,
        a.ops
    );
    let counters = match a.ops.as_str() {
        "mock" => {
            let d = run(map, cfg, MockOps::new(4096), src)?;
            eprintln!(
                "mock: {} op calls, pinned {} MiB at exit",
                d.ops.calls.len(),
                d.ops.pinned_bytes() >> 20
            );
            d.counters
        }
        "linux" => {
            let lim = memlock_limit()?;
            if cfg.pin_budget_bytes > lim {
                eprintln!("warning: RLIMIT_MEMLOCK is {} MiB; pin budget clamped (see docs/PRIVILEGES.md)", lim >> 20);
                cfg.pin_budget_bytes = lim;
            }
            let file = MappedFile::open(&a.gguf)?;
            let mut ops = LinuxOps::new(file);
            if let Some(pid) = a.pid {
                let md = std::fs::metadata(&a.gguf)?;
                let dev = (libc_major(md.dev()), libc_minor(md.dev()));
                let maps = find_file_mapping(pid, dev, md.ino(), Some(&a.gguf))?;
                match maps.iter().max_by_key(|m| m.end - m.start) {
                    Some(m) => {
                        ops.target = Some(TargetMapping::open(
                            pid,
                            m.start,
                            m.offset,
                            m.end - m.start,
                        )?)
                    }
                    None => {
                        eprintln!("warning: pid {pid} has not mapped the model; demotion disabled")
                    }
                }
            }
            run(map, cfg, ops, src)?.counters
        }
        o => bail!("--ops must be mock or linux (got {o})"),
    };
    eprintln!("{}", serde_json::to_string(&counters)?);
    if let Some(p) = a.stats_json {
        std::fs::write(p, serde_json::to_vec_pretty(&counters)?)?;
    }
    Ok(())
}

fn libc_major(dev: u64) -> u32 {
    ((dev >> 8) & 0xfff) as u32 | ((dev >> 32) & !0xfff) as u32
}

fn libc_minor(dev: u64) -> u32 {
    (dev & 0xff) as u32 | ((dev >> 12) & !0xff) as u32
}
