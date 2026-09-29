//! Replay an expert trace against a real model file: touch every page of
//! each used unit, layer by layer, the way llama.cpp's CPU backend reads
//! expert weights. Used to validate the simulator against the real kernel
//! (Experiment B) and to exercise recorders end to end.

use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::{Duration, Instant};

use mp_gguf::ExpertMap;
use mp_trace::Steps;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TouchMode {
    /// Read one byte per page through a shared mapping (page faults; like llama.cpp).
    Mmap,
    /// `pread` the slices (no mapping; lets `drop_each_token` really evict).
    Pread,
}

#[derive(Debug, Clone)]
pub struct ReplayConfig {
    pub mode: TouchMode,
    /// Threads touching a layer's slices concurrently (mmap mode).
    pub threads: usize,
    /// Simulated compute per layer (busy-wait).
    pub t_layer: Duration,
    /// Idle time at the start of each token (sampling, tokenizer, …).
    pub t_token_gap: Duration,
    /// Evict the file's unmapped pages after each token.
    pub drop_each_token: bool,
    /// Stop after this many tokens (0 = all).
    pub max_tokens: usize,
    /// `POSIX_FADV_RANDOM` on the replayer's fd (pread mode): disables
    /// readahead so only touched pages enter the cache. btrfs uses a 4 MiB
    /// readahead window, which otherwise spills into neighbouring experts.
    pub no_readahead: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        ReplayConfig {
            mode: TouchMode::Mmap,
            threads: 4,
            t_layer: Duration::ZERO,
            t_token_gap: Duration::ZERO,
            drop_each_token: false,
            max_tokens: 0,
            no_readahead: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ReplayStats {
    pub tokens: usize,
    pub wall_s: f64,
    pub major_faults: u64,
    pub bytes_touched: u64,
}

fn majflt() -> u64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: valid out-pointer.
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru.ru_majflt as u64
}

fn busy_wait(d: Duration) {
    if d.is_zero() {
        return;
    }
    let t = Instant::now();
    while t.elapsed() < d {
        std::hint::spin_loop();
    }
}

struct Mapping {
    addr: *const u8,
    len: usize,
}

unsafe impl Sync for Mapping {}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: unmapping what we mapped.
        unsafe { libc::munmap(self.addr as *mut libc::c_void, self.len) };
    }
}

/// Replay `steps` on `path`. `on_token(i)` runs after each token, before
/// the optional cache drop (e.g. to snapshot residency or metrics).
pub fn replay(
    path: &Path,
    map: &ExpertMap,
    steps: &Steps,
    cfg: &ReplayConfig,
    mut on_token: impl FnMut(usize),
) -> io::Result<ReplayStats> {
    let file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    let mapping = if cfg.mode == TouchMode::Mmap {
        // SAFETY: read-only shared mapping of a valid fd; checked below.
        let a = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if a == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Some(Mapping {
            addr: a as *const u8,
            len,
        })
    } else {
        None
    };
    if cfg.no_readahead {
        // SAFETY: plain syscall on a valid fd.
        let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) };
        if rc != 0 {
            return Err(io::Error::from_raw_os_error(rc));
        }
    }
    let page = 4096usize;
    let f0 = majflt();
    let t0 = Instant::now();
    let mut stats = ReplayStats::default();
    let mut buf = vec![0u8; 1 << 20];
    for (ti, step) in steps.steps.iter().enumerate() {
        if cfg.max_tokens > 0 && ti >= cfg.max_tokens {
            break;
        }
        if !cfg.t_token_gap.is_zero() {
            std::thread::sleep(cfg.t_token_gap);
        }
        for l in &step.layers {
            let mut ranges: Vec<(u64, u64)> = Vec::new();
            for &e in &l.experts {
                let id = map.unit_id(l.layer as u32, e as u32).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("no unit L{} E{}", l.layer, e),
                    )
                })?;
                for s in &map.units[id as usize].slices {
                    ranges.push((s.offset, s.len));
                    stats.bytes_touched += s.len;
                }
            }
            match &mapping {
                Some(m) => {
                    // Split all pages of the layer into 64 KiB chunks, interleaved across threads.
                    let chunks: Vec<(u64, u64)> = ranges
                        .iter()
                        .flat_map(|&(o, n)| {
                            (0..n.div_ceil(1 << 16))
                                .map(move |i| (o + (i << 16), (n - (i << 16)).min(1 << 16)))
                        })
                        .collect();
                    let t = cfg.threads.max(1);
                    std::thread::scope(|sc| {
                        for k in 0..t {
                            let chunks = &chunks;
                            sc.spawn(move || {
                                let mut acc = 0u8;
                                for &(o, n) in chunks.iter().skip(k).step_by(t) {
                                    let mut p = o as usize;
                                    while p < (o + n) as usize && p < m.len {
                                        // SAFETY: p < mapping length; volatile so the read happens.
                                        acc ^= unsafe { std::ptr::read_volatile(m.addr.add(p)) };
                                        p = (p / page + 1) * page;
                                    }
                                }
                                std::hint::black_box(acc);
                            });
                        }
                    });
                }
                None => {
                    for &(o, n) in &ranges {
                        let mut done = 0u64;
                        while done < n {
                            let k = ((n - done) as usize).min(buf.len());
                            let got = file.read_at(&mut buf[..k], o + done)?;
                            if got == 0 {
                                break;
                            }
                            done += got as u64;
                        }
                    }
                }
            }
            busy_wait(cfg.t_layer);
        }
        stats.tokens += 1;
        on_token(ti);
        if cfg.drop_each_token {
            mp_os::drop_file_cache(path)?;
        }
    }
    stats.wall_s = t0.elapsed().as_secs_f64();
    stats.major_faults = majflt() - f0;
    Ok(stats)
}
