//! Real-kernel tests on a file in CARGO_TARGET_TMPDIR (must not be tmpfs:
//! tmpfs pages cannot be dropped, so these tests skip there).

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use mp_os::{
    cachestat, drop_file_cache, find_file_mapping, kernel_dev_of_mapping, memlock_limit, LinuxOps,
    MappedFile, PageCacheOps, ResidencyProbe,
};

const TMPFS_MAGIC: i64 = 0x0102_1994;

fn scratch(name: &str, bytes: usize) -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).unwrap();
    let c = std::ffi::CString::new(dir.to_str().unwrap()).unwrap();
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::statfs(c.as_ptr(), &mut st) }, 0);
    if st.f_type as i64 == TMPFS_MAGIC {
        eprintln!("{} is tmpfs; skipping page-cache test", dir.display());
        return None;
    }
    let p = dir.join(name);
    let mut f = std::fs::File::create(&p).unwrap();
    let chunk: Vec<u8> = (0..1 << 16).map(|i| (i * 7 + 1) as u8).collect();
    let mut left = bytes;
    while left > 0 {
        let n = left.min(chunk.len());
        f.write_all(&chunk[..n]).unwrap();
        left -= n;
    }
    f.sync_all().unwrap();
    drop(f);
    drop_file_cache(&p).unwrap();
    Some(p)
}

fn count(m: &MappedFile, first: u64, n: u64) -> usize {
    let mut v = Vec::new();
    m.resident(first, n, &mut v).unwrap();
    v.iter().filter(|&&b| b).count()
}

fn wait_for(cond: impl Fn() -> bool) -> bool {
    let t = Instant::now();
    while t.elapsed() < Duration::from_secs(10) {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

/// Wait for an asynchronous WILLNEED of pages [0, n) to land. Returns false
/// (test skipped) if it had no effect at all: the kernel ignores WILLNEED
/// when the backing device has readahead disabled (seen on CI runners).
fn willneed_works(m: &MappedFile, n: u64) -> bool {
    let ok = wait_for(|| count(m, 0, n) * 10 >= n as usize * 9);
    let got = count(m, 0, n);
    if !ok && got == 0 {
        eprintln!("POSIX_FADV_WILLNEED had no effect here (readahead disabled?); skipping");
        return false;
    }
    assert!(ok, "WILLNEED populated only {got}/{n} pages");
    true
}

#[test]
fn prefetch_populates_and_drop_evicts_unmapped_pages() {
    let Some(p) = scratch("prefetch.bin", 4 << 20) else {
        return;
    };
    let m = MappedFile::open(&p).unwrap();
    let pages = m.len() / m.page_size();
    assert!(
        count(&m, 0, pages) < pages as usize / 4,
        "file should start mostly cold"
    );
    let mut ops = LinuxOps::new(MappedFile::open(&p).unwrap());
    ops.prefetch(0, 1 << 20).unwrap();
    let first = (1u64 << 20) / m.page_size();
    if !willneed_works(&m, first) {
        return;
    }
    drop_file_cache(&p).unwrap();
    assert!(count(&m, 0, first) < first as usize / 4);
    // Out-of-range probes read as non-resident.
    let mut v = Vec::new();
    m.resident(pages + 10, 3, &mut v).unwrap();
    assert_eq!(v, vec![false; 3]);
}

#[test]
fn dontneed_cannot_evict_pages_mapped_by_another_mapping() {
    // The core reason moepager pins/demotes instead of using DONTNEED.
    let Some(p) = scratch("mapped.bin", 1 << 20) else {
        return;
    };
    let probe = MappedFile::open(&p).unwrap();
    let engine = MappedFile::open(&p).unwrap(); // stands in for llama.cpp's mapping
    let pages = probe.len() / probe.page_size();
    // Touch every page through a raw read of the "engine" mapping.
    let f = std::fs::File::open(&p).unwrap();
    let len = probe.len() as usize;
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            std::os::fd::AsRawFd::as_raw_fd(&f),
            0,
        )
    };
    assert_ne!(addr, libc::MAP_FAILED);
    let mut sum = 0u64;
    for i in (0..len).step_by(4096) {
        sum += unsafe { *(addr as *const u8).add(i) } as u64;
    }
    assert!(sum > 0);
    drop_file_cache(&p).unwrap();
    assert_eq!(
        count(&probe, 0, pages),
        pages as usize,
        "mapped pages must survive DONTNEED"
    );
    unsafe { libc::munmap(addr, len) };
    drop(engine);
    drop_file_cache(&p).unwrap();
    assert!(count(&probe, 0, pages) < pages as usize / 4);
}

#[test]
fn cachestat_agrees_with_mincore_on_own_file() {
    let Some(p) = scratch("cachestat.bin", 2 << 20) else {
        return;
    };
    let m = MappedFile::open(&p).unwrap();
    let mut ops = LinuxOps::new(MappedFile::open(&p).unwrap());
    ops.prefetch(0, 512 << 10).unwrap();
    let n = (512u64 << 10) / m.page_size();
    if !willneed_works(&m, n) {
        return;
    }
    match cachestat(m.fd(), 0, m.len()) {
        // Allow a few pages of drift between the two syscalls.
        Ok(cs) => {
            let mc = count(&m, 0, m.len() / m.page_size()) as i64;
            assert!(
                (cs.nr_cache as i64 - mc).abs() <= 4,
                "cachestat {} vs mincore {mc}",
                cs.nr_cache
            );
        }
        Err(e) if e.raw_os_error() == Some(libc::ENOSYS) => eprintln!("cachestat unsupported"),
        Err(e) => panic!("cachestat: {e}"),
    }
}

#[test]
fn small_mlock_round_trip() {
    let Some(p) = scratch("mlock.bin", 1 << 20) else {
        return;
    };
    let limit = memlock_limit().unwrap();
    if limit < 64 << 10 {
        eprintln!("RLIMIT_MEMLOCK {limit} too small; skipping");
        return;
    }
    let mut ops = LinuxOps::new(MappedFile::open(&p).unwrap());
    ops.pin(4096, 32 << 10).unwrap();
    let n = (32u64 << 10) / ops.file.page_size();
    assert_eq!(
        count(&ops.file, 1, n),
        n as usize,
        "mlock makes pages resident"
    );
    ops.unpin(4096, 32 << 10).unwrap();
    assert!(ops.demote(0, 4096).is_err(), "no target process configured");
}

#[test]
fn finds_own_mapping_in_proc_maps() {
    let Some(p) = scratch("maps.bin", 64 << 10) else {
        return;
    };
    let m = MappedFile::open(&p).unwrap();
    let (dev, ino) = m.identity().unwrap();
    let found = find_file_mapping(std::process::id(), dev, ino, Some(&p)).unwrap();
    assert!(!found.is_empty());
    assert!(found.iter().any(|e| e.addr_of(0).is_some()));
    // The kernel device (used by tracepoints) is learnable from our own mapping.
    let kdev = kernel_dev_of_mapping(ino, &p).unwrap();
    assert!(kdev.is_some());
}
