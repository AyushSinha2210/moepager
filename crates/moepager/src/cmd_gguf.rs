use std::path::PathBuf;

use anyhow::{Context, Result};
use mp_gguf::repack::{repack_report, CpuFeatures};
use mp_gguf::{parse_file, ExpertMap};

#[derive(clap::Args)]
pub struct Args {
    /// GGUF file (a truncated header-only prefix also works).
    pub gguf: PathBuf,
    /// Write the map as JSON here.
    #[arg(short, long)]
    pub out: Option<PathBuf>,
    #[arg(long, default_value_t = 4096)]
    pub page_size: u64,
    /// Real file size, when `gguf` is a truncated header prefix.
    #[arg(long)]
    pub file_size: Option<u64>,
    /// Print every unit's slices.
    #[arg(long)]
    pub verbose: bool,
}

pub fn run(a: Args) -> Result<()> {
    let h = parse_file(&a.gguf).with_context(|| format!("parsing {}", a.gguf.display()))?;
    let size = a.file_size.or_else(|| std::fs::metadata(&a.gguf).ok().map(|m| m.len()));
    let map = ExpertMap::from_header(&h, a.page_size, size);
    println!("{}", map.summary());
    for w in &map.warnings {
        eprintln!("warning: {w}");
    }
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        let cpu = CpuFeatures::from_cpuinfo(&info);
        let r = repack_report(&h, &map, &cpu);
        if !r.expert_tensors.is_empty() {
            eprintln!(
                "warning: on this CPU, default llama.cpp repacks {} expert tensors ({:.0}% of \
                 expert bytes) into anonymous memory; run llama.cpp with --no-repack (-nr) so \
                 experts are served from the page cache",
                r.expert_tensors.len(),
                100.0 * r.fraction()
            );
        }
    }
    if a.verbose {
        for u in &map.units {
            let s: Vec<String> =
                u.slices.iter().map(|s| format!("{}@{}+{}", s.kind, s.offset, s.len)).collect();
            println!("L{:>3} E{:>4} {:>10}B {}", u.layer, u.expert, u.bytes, s.join(" "));
        }
    }
    if let Some(out) = a.out {
        std::fs::write(&out, serde_json::to_vec(&map)?)?;
        eprintln!("wrote {}", out.display());
    }
    Ok(())
}
