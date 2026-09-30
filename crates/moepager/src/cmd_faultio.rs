//! `fault-io`: how fast do cold expert units arrive when faulted in through
//! mmap (llama.cpp's path) versus one bulk WILLNEED per slice versus pread?
//! This measures the bandwidth lever behind expert-completion readahead
//! (IDEA_REVIEW §4, Experiment C), plus read amplification
//! (storage bytes read / bytes touched).

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use mp_core::Rng;

#[derive(clap::Args)]
pub struct Args {
    /// Existing file to read (e.g. a GGUF). Mutually exclusive with --create.
    #[arg(long)]
    pub file: Option<PathBuf>,
    /// Create a file of this many MiB with incompressible data at this path.
    #[arg(long, value_names = ["PATH", "MIB"], num_args = 2)]
    pub create: Vec<String>,
    /// Only evict the file from the page cache and exit.
    #[arg(long)]
    pub drop: Option<PathBuf>,
    /// Bytes per expert unit (split into `slices` slices in different file regions).
    #[arg(long, default_value_t = 2_856_960)]
    pub unit_bytes: u64,
    #[arg(long, default_value_t = 3)]
    pub slices: u64,
    #[arg(long, default_value_t = 48)]
    pub units: u64,
    #[arg(long, default_value_t = 6)]
    pub threads: usize,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    /// WILLNEED request size in KiB (0 = one call per slice, which the kernel
    /// truncates to the device's readahead size; see mp_os::WILLNEED_CHUNK).
    #[arg(long, default_value_t = 128)]
    pub willneed_chunk_kb: u64,
    /// Modes to run (comma-separated): fault, willneed, pread.
    #[arg(long, value_delimiter = ',', default_value = "fault,willneed,pread")]
    pub modes: Vec<String>,
}

/// Bytes this process caused to be fetched from storage (/proc/self/io).
pub fn proc_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                l.strip_prefix("read_bytes:")
                    .map(|v| v.trim().parse().unwrap_or(0))
            })
        })
        .unwrap_or(0)
}

fn create_file(path: &str, mib: u64) -> Result<PathBuf> {
    let p = PathBuf::from(path);
    let mut f = std::io::BufWriter::new(File::create(&p)?);
    let mut r = Rng::new(0xF417);
    let mut buf = vec![0u8; 1 << 20];
    for _ in 0..mib {
        for c in buf.chunks_exact_mut(8) {
            c.copy_from_slice(&r.next_u64().to_le_bytes());
        }
        f.write_all(&buf)?;
    }
    f.flush()?;
    f.get_ref().sync_all()?;
    Ok(p)
}

/// Touch one byte per page of each range with `threads` threads, 64 KiB
/// chunks interleaved (roughly how ggml splits expert rows across threads).
fn touch(base: *const u8, ranges: &[(u64, u64)], threads: usize) {
    struct P(*const u8);
    unsafe impl Sync for P {}
    let p = P(base);
    let chunks: Vec<(u64, u64)> = ranges
        .iter()
        .flat_map(|&(o, n)| {
            (0..n.div_ceil(1 << 16)).map(move |i| (o + (i << 16), (n - (i << 16)).min(1 << 16)))
        })
        .collect();
    std::thread::scope(|s| {
        for k in 0..threads {
            let (p, chunks) = (&p, &chunks);
            s.spawn(move || {
                let mut acc = 0u8;
                for &(o, n) in chunks.iter().skip(k).step_by(threads) {
                    let mut a = o;
                    while a < o + n {
                        // SAFETY: ranges lie within the mapping.
                        acc ^= unsafe { std::ptr::read_volatile(p.0.add(a as usize)) };
                        a = (a / 4096 + 1) * 4096;
                    }
                }
                std::hint::black_box(acc);
            });
        }
    });
}

pub fn run(a: Args) -> Result<()> {
    if let Some(p) = &a.drop {
        mp_os::drop_file_cache(p)?;
        println!("dropped {}", p.display());
        return Ok(());
    }
    let path = match (&a.file, a.create.as_slice()) {
        (Some(f), []) => f.clone(),
        (None, [p, mib]) => create_file(p, mib.parse().context("MIB")?)?,
        _ => bail!("give exactly one of --file or --create PATH MIB"),
    };
    let file = File::open(&path)?;
    let len = file.metadata()?.len();
    let slice = a.unit_bytes / a.slices;
    let region = len / a.slices;
    let slots = region / slice;
    if slots < a.units {
        bail!(
            "file too small: {} slots per region for {} units",
            slots,
            a.units
        );
    }
    // Choose distinct random slots; unit i uses slot s_i in every region.
    let mut r = Rng::new(a.seed);
    let mut all: Vec<u64> = (0..slots).collect();
    r.shuffle(&mut all);
    let units: Vec<Vec<(u64, u64)>> = all[..a.units as usize]
        .iter()
        .map(|&s| {
            (0..a.slices)
                .map(|k| (k * region + s * slice, slice))
                .collect()
        })
        .collect();
    let touched = a.units * slice * a.slices;
    // SAFETY: read-only shared mapping of a valid fd, checked below.
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len as usize,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if base == libc::MAP_FAILED {
        bail!("mmap: {}", std::io::Error::last_os_error());
    }
    let base = base as *const u8;
    println!(
        "file={} size={:.2}GB units={} unit={:.2}MB slices={} threads={} read_ahead: see /sys/class/bdi/*/read_ahead_kb",
        path.display(),
        len as f64 / 1e9,
        a.units,
        a.unit_bytes as f64 / 1e6,
        a.slices,
        a.threads
    );
    println!("| mode | GB/s (touched) | ms/unit | storage read / touched |");
    println!("|---|---|---|---|");
    let fd = file.as_raw_fd();
    let mut buf = vec![0u8; 1 << 20];
    for mode in &a.modes {
        // Cold start. fadvise(DONTNEED) skips pages that are mapped, so first
        // zap our own page-table entries, then evict the file.
        // SAFETY: MADV_DONTNEED on our own read-only file mapping.
        unsafe { libc::madvise(base as *mut libc::c_void, len as usize, libc::MADV_DONTNEED) };
        mp_os::drop_file_cache(&path)?;
        let io0 = proc_read_bytes();
        let t = Instant::now();
        match mode.as_str() {
            "fault" => {
                for u in &units {
                    touch(base, u, a.threads);
                }
            }
            "willneed" => {
                for u in &units {
                    for &(o, n) in u {
                        mp_os::willneed_chunked(fd, o, n, a.willneed_chunk_kb << 10)?;
                    }
                    touch(base, u, a.threads);
                }
            }
            "pread" => {
                for u in &units {
                    for &(o, n) in u {
                        let mut done = 0;
                        while done < n {
                            let k = ((n - done) as usize).min(buf.len());
                            done += file.read_at(&mut buf[..k], o + done)? as u64;
                        }
                    }
                }
            }
            m => bail!("unknown mode {m}"),
        }
        let dt = t.elapsed().as_secs_f64();
        let read = proc_read_bytes().saturating_sub(io0);
        println!(
            "| {mode} | {:.2} | {:.2} | {:.2} |",
            touched as f64 / dt / 1e9,
            dt * 1e3 / a.units as f64,
            read as f64 / touched as f64
        );
    }
    // SAFETY: unmapping our mapping.
    unsafe { libc::munmap(base as *mut libc::c_void, len as usize) };
    Ok(())
}
