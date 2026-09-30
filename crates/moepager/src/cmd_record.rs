//! `record`, `ingest-bpftrace`, `page2expert`, `replay`.

use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use mp_gguf::{parse_file, ExpertMap};
use mp_os::{drop_file_cache, kernel_dev_of_mapping, MappedFile};
use mp_recorder::{
    page_to_expert, ConvertConfig, ReplayConfig, ScanRecorder, SentinelRecorder, TouchMode,
};
use mp_trace::{infer_tokens, Observation, RecordKind, Steps, TraceHeader};

fn map_of(gguf: &Path) -> Result<ExpertMap> {
    let h = parse_file(gguf).with_context(|| format!("parsing {}", gguf.display()))?;
    let size = std::fs::metadata(gguf)?.len();
    let m = ExpertMap::from_header(&h, 4096, Some(size));
    if m.units.is_empty() {
        bail!("{} has no expert tensors", gguf.display());
    }
    Ok(m)
}

fn header(source: &str, m: &ExpertMap, gguf: &Path, obs: Observation) -> TraceHeader {
    let mut h = TraceHeader::new(source, m.n_layers, m.n_experts);
    h.top_k = m.top_k;
    h.observation = obs;
    h.model_file = gguf.file_name().map(|f| f.to_string_lossy().into_owned());
    h.model_size = std::fs::metadata(gguf).ok().map(|x| x.len());
    h
}

#[derive(clap::Args)]
pub struct RecordArgs {
    /// Model file to watch (the engine runs separately and unmodified).
    pub gguf: PathBuf,
    /// sentinel: expert events from per-slice probe pages (cheap);
    /// scan: page events from full-file mincore diffs (expensive).
    #[arg(long, default_value = "sentinel")]
    pub mode: String,
    #[arg(short, long)]
    pub out: Option<PathBuf>,
    #[arg(long, default_value_t = 500)]
    pub interval_us: u64,
    #[arg(long, default_value_t = 30.0)]
    pub duration_s: f64,
    /// Print the inode and kernel device for bpf/moepager.bt and exit.
    #[arg(long)]
    pub print_bpftrace_args: bool,
}

pub fn record(a: RecordArgs) -> Result<()> {
    let probe = MappedFile::open(&a.gguf)?;
    if a.print_bpftrace_args {
        let ino = std::fs::metadata(&a.gguf)?.ino();
        let (maj, min) = kernel_dev_of_mapping(ino, &a.gguf)?.context("own mapping not found")?;
        println!("{ino} {}", ((maj as u64) << 20) | min as u64);
        eprintln!(
            "usage: sudo bpftrace bpf/moepager.bt {ino} {}",
            ((maj as u64) << 20) | min as u64
        );
        return Ok(());
    }
    let out = a.out.context("--out is required")?;
    let map = map_of(&a.gguf)?;
    let t0 = Instant::now();
    let deadline = Duration::from_secs_f64(a.duration_s);
    let interval = Duration::from_micros(a.interval_us);
    match a.mode.as_str() {
        "sentinel" => {
            let mut rec = SentinelRecorder::new(&map);
            eprintln!(
                "watching {} sentinel pages ({} units with small slices) for {:.0}s",
                rec.len(),
                rec.unreliable_units,
                a.duration_s
            );
            rec.baseline(&probe)?;
            let mut ev = Vec::new();
            while t0.elapsed() < deadline {
                ev.extend(rec.tick(&probe, t0.elapsed().as_nanos() as u64)?);
                std::thread::sleep(interval);
            }
            let slack = (map.n_layers / 8) as u16;
            let n = infer_tokens(&mut ev, slack);
            let h = header("sentinel", &map, &a.gguf, Observation::MissOnly);
            mp_trace::write_expert_file(&out, &h, &ev)?;
            println!(
                "{} expert miss events, {} inferred tokens -> {}",
                ev.len(),
                n,
                out.display()
            );
        }
        "scan" => {
            let mut rec = ScanRecorder::new(probe.len(), 4096);
            rec.baseline(&probe)?;
            let h = header("mincore-scan", &map, &a.gguf, Observation::MissOnly);
            let mut w = mp_trace::create(&out, RecordKind::Page, &h)?;
            let mut scans = 0;
            while t0.elapsed() < deadline {
                for e in rec.tick(&probe, t0.elapsed().as_nanos() as u64)? {
                    w.page(&e)?;
                }
                scans += 1;
                std::thread::sleep(interval);
            }
            let n = w.count();
            w.finish()?;
            println!("{scans} scans, {n} page events -> {}", out.display());
        }
        m => bail!("unknown mode {m} (sentinel|scan)"),
    }
    Ok(())
}

#[derive(clap::Args)]
pub struct IngestArgs {
    /// Output of `bpftrace bpf/moepager.bt`.
    pub input: PathBuf,
    /// Model file (for the inode and header).
    #[arg(long)]
    pub gguf: PathBuf,
    #[arg(short, long)]
    pub out: PathBuf,
}

pub fn ingest(a: IngestArgs) -> Result<()> {
    let ino = std::fs::metadata(&a.gguf)?.ino();
    let text = std::fs::read_to_string(&a.input)?;
    let (ev, bad) = mp_recorder::bpftrace::parse(&text, ino);
    let map = map_of(&a.gguf)?;
    let h = header("bpftrace", &map, &a.gguf, Observation::MissOnly);
    let mut w = mp_trace::create(&a.out, RecordKind::Page, &h)?;
    for e in &ev {
        w.page(e)?;
    }
    w.finish()?;
    println!(
        "{} page events ({bad} unparsable lines) -> {}",
        ev.len(),
        a.out.display()
    );
    Ok(())
}

#[derive(clap::Args)]
pub struct Page2ExpertArgs {
    pub pages: PathBuf,
    #[arg(long)]
    pub gguf: PathBuf,
    #[arg(short, long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 8)]
    pub min_pages: u32,
}

pub fn page2expert(a: Page2ExpertArgs) -> Result<()> {
    let (ph, pages) = mp_trace::read_page_file(&a.pages)?;
    let map = map_of(&a.gguf)?;
    let cfg = ConvertConfig {
        min_pages: a.min_pages,
        ..Default::default()
    };
    let mut ev = page_to_expert(&pages, &map, &cfg);
    let n = infer_tokens(&mut ev, (map.n_layers / 8) as u16);
    let mut h = header(&ph.source, &map, &a.gguf, Observation::MissOnly);
    h.params = serde_json::json!({ "converted_from": a.pages.display().to_string() });
    mp_trace::write_expert_file(&a.out, &h, &ev)?;
    println!(
        "{} expert events, {n} inferred tokens -> {}",
        ev.len(),
        a.out.display()
    );
    Ok(())
}

#[derive(clap::Args)]
pub struct ReplayArgs {
    pub gguf: PathBuf,
    /// Expert trace to replay (layers/experts must match the model).
    pub trace: PathBuf,
    #[arg(long, default_value = "mmap")]
    pub mode: String,
    #[arg(long, default_value_t = 4)]
    pub threads: usize,
    #[arg(long, default_value_t = 0.0)]
    pub t_layer_us: f64,
    #[arg(long, default_value_t = 0)]
    pub max_tokens: usize,
    /// Evict the model from the page cache before starting.
    #[arg(long)]
    pub cold: bool,
    #[arg(long)]
    pub no_readahead: bool,
    /// Evict the file's unmapped pages after every token (pread mode), so
    /// every expert use is a miss a black-box recorder can see.
    #[arg(long)]
    pub drop_each_token: bool,
    /// Idle time at the start of each token.
    #[arg(long, default_value_t = 0.0)]
    pub token_gap_us: f64,
    #[arg(long)]
    pub json: Option<PathBuf>,
}

pub fn replay(a: ReplayArgs) -> Result<()> {
    let map = map_of(&a.gguf)?;
    let (h, ev) = mp_trace::read_expert_file(&a.trace)?;
    if h.n_experts != map.n_experts || h.n_layers > map.n_layers {
        bail!(
            "trace shape {}x{} does not fit model {}x{}",
            h.n_layers,
            h.n_experts,
            map.n_layers,
            map.n_experts
        );
    }
    let steps = Steps::from_events(&h, &ev);
    if a.cold {
        drop_file_cache(&a.gguf)?;
    }
    let cfg = ReplayConfig {
        mode: match a.mode.as_str() {
            "mmap" => TouchMode::Mmap,
            "pread" => TouchMode::Pread,
            m => bail!("unknown mode {m} (mmap|pread)"),
        },
        threads: a.threads,
        t_layer: Duration::from_secs_f64(a.t_layer_us / 1e6),
        max_tokens: a.max_tokens,
        no_readahead: a.no_readahead,
        drop_each_token: a.drop_each_token,
        t_token_gap: Duration::from_secs_f64(a.token_gap_us / 1e6),
    };
    let io0 = crate::cmd_faultio::proc_read_bytes();
    let s = mp_recorder::replay(&a.gguf, &map, &steps, &cfg, |_| {})?;
    let read = crate::cmd_faultio::proc_read_bytes().saturating_sub(io0);
    let tok = s.tokens.max(1) as f64;
    println!(
        "tokens={} wall={:.2}s tok/s={:.2} majflt/token={:.1} storage_read/token={:.3}GB touched/token={:.3}GB",
        s.tokens,
        s.wall_s,
        s.tokens as f64 / s.wall_s.max(1e-9),
        s.major_faults as f64 / tok,
        read as f64 / tok / 1e9,
        s.bytes_touched as f64 / tok / 1e9
    );
    if let Some(j) = a.json {
        std::fs::write(
            j,
            serde_json::to_vec_pretty(&serde_json::json!({
                "tokens": s.tokens, "wall_s": s.wall_s, "major_faults": s.major_faults,
                "storage_read_bytes": read, "bytes_touched": s.bytes_touched,
            }))?,
        )?;
    }
    Ok(())
}
