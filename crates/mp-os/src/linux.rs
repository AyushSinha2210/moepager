//! Linux implementations: mmap + mincore/mlock, fadvise, cachestat,
//! process_madvise.

use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use crate::{PageCacheOps, ResidencyProbe};

fn page_size() -> u64 {
    // SAFETY: sysconf has no preconditions.
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if p > 0 {
        p as u64
    } else {
        4096
    }
}

fn check(rc: libc::c_int) -> io::Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// `posix_fadvise` returns the error number directly instead of setting errno.
fn fadvise(fd: libc::c_int, off: u64, len: u64, advice: libc::c_int) -> io::Result<()> {
    // SAFETY: plain syscall on a valid fd.
    let rc = unsafe { libc::posix_fadvise(fd, off as libc::off_t, len as libc::off_t, advice) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(rc))
    }
}

/// A read-only `MAP_SHARED` mapping of the whole model file. Never read
/// through by moepager itself: it exists for `mincore` and `mlock`.
pub struct MappedFile {
    file: File,
    addr: *mut libc::c_void,
    len: u64,
    page: u64,
}

// The mapping is read-only and only used for syscalls taking its address.
unsafe impl Send for MappedFile {}

impl MappedFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty file"));
        }
        // SAFETY: fresh mapping of a valid fd; checked for MAP_FAILED.
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len as usize,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(MappedFile {
            file,
            addr,
            len,
            page: page_size(),
        })
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn fd(&self) -> libc::c_int {
        self.file.as_raw_fd()
    }

    pub fn file(&self) -> &File {
        &self.file
    }

    /// (device major, device minor, inode) of the file, for /proc/pid/maps matching.
    pub fn identity(&self) -> io::Result<((u32, u32), u64)> {
        let m = self.file.metadata()?;
        let dev = m.dev();
        Ok(((libc::major(dev), libc::minor(dev)), m.ino()))
    }

    /// Page-aligned (addr, len) inside our mapping covering [off, off+len).
    fn span(&self, off: u64, len: u64) -> io::Result<(*mut libc::c_void, usize)> {
        if off >= self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "offset beyond EOF",
            ));
        }
        let end = (off + len.max(1)).min(self.len);
        let a = off / self.page * self.page;
        let b = end.div_ceil(self.page) * self.page;
        // SAFETY: a < self.len, so the pointer stays inside the mapping.
        Ok((
            unsafe { (self.addr as *mut u8).add(a as usize) } as *mut libc::c_void,
            (b - a) as usize,
        ))
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        // SAFETY: unmapping exactly what we mapped.
        unsafe {
            libc::munmap(self.addr, self.len as usize);
        }
    }
}

impl ResidencyProbe for MappedFile {
    fn resident(&self, first: u64, n: u64, out: &mut Vec<bool>) -> io::Result<()> {
        out.clear();
        let total = self.len.div_ceil(self.page);
        let end = (first + n).min(total);
        if first < end {
            let mut vec = vec![0u8; (end - first) as usize];
            // SAFETY: range is inside the mapping; vec has one byte per page.
            let rc = unsafe {
                libc::mincore(
                    (self.addr as *mut u8).add((first * self.page) as usize) as *mut libc::c_void,
                    ((end - first) * self.page) as usize,
                    vec.as_mut_ptr(),
                )
            };
            check(rc)?;
            out.extend(vec.iter().map(|b| b & 1 == 1));
        }
        out.resize(n as usize, false);
        Ok(())
    }

    fn page_size(&self) -> u64 {
        self.page
    }
}

/// Result of `cachestat(2)` (Linux 6.5+).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStat {
    pub nr_cache: u64,
    pub nr_dirty: u64,
    pub nr_writeback: u64,
    pub nr_evicted: u64,
    pub nr_recently_evicted: u64,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SYS_CACHESTAT: libc::c_long = 451;

/// Page-cache statistics for a byte range. Fails with EPERM unless the
/// caller owns or can write the file; ENOSYS before Linux 6.5.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub fn cachestat(fd: libc::c_int, off: u64, len: u64) -> io::Result<CacheStat> {
    #[repr(C)]
    struct Range {
        off: u64,
        len: u64,
    }
    let r = Range { off, len };
    let mut cs = CacheStat::default();
    // SAFETY: pointers to valid, correctly laid-out structs.
    let rc = unsafe {
        libc::syscall(
            SYS_CACHESTAT,
            fd,
            &r as *const Range,
            &mut cs as *mut CacheStat,
            0u32,
        )
    };
    if rc == 0 {
        Ok(cs)
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Evict a file's unmapped pages from the page cache (cold-start helper).
/// Pages mapped by any process are skipped by the kernel.
pub fn drop_file_cache(path: &Path) -> io::Result<()> {
    let f = File::open(path)?;
    fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED)
}

/// Current soft RLIMIT_MEMLOCK in bytes (u64::MAX = unlimited).
pub fn memlock_limit() -> io::Result<u64> {
    let mut r = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: valid out-pointer.
    check(unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut r) })?;
    Ok(if r.rlim_cur == libc::RLIM_INFINITY {
        u64::MAX
    } else {
        r.rlim_cur
    })
}

/// Where the engine mapped the model file, for `process_madvise`.
pub struct TargetMapping {
    pidfd: OwnedFd,
    /// Engine virtual address of file offset `file_off`.
    base: u64,
    file_off: u64,
    len: u64,
}

impl TargetMapping {
    /// Open a pidfd for `pid` and remember one mapping of the model file.
    pub fn open(pid: u32, map_start: u64, map_file_off: u64, map_len: u64) -> io::Result<Self> {
        // SAFETY: pidfd_open(pid, 0) returns a new fd or -1.
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: fd is a fresh, owned descriptor.
        let pidfd = unsafe { OwnedFd::from_raw_fd(fd as libc::c_int) };
        Ok(TargetMapping {
            pidfd,
            base: map_start,
            file_off: map_file_off,
            len: map_len,
        })
    }

    fn madvise(&self, off: u64, len: u64, advice: libc::c_int, page: u64) -> io::Result<()> {
        if off < self.file_off || off + len > self.file_off + self.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "range not in target mapping",
            ));
        }
        let a = (self.base + (off - self.file_off)) / page * page;
        let b = (self.base + (off + len - self.file_off)).div_ceil(page) * page;
        let iov = libc::iovec {
            iov_base: a as *mut libc::c_void,
            iov_len: (b - a) as usize,
        };
        // SAFETY: the iovec describes the *target's* address space; the kernel validates it.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_process_madvise,
                self.pidfd.as_raw_fd(),
                &iov as *const libc::iovec,
                1usize,
                advice,
                0u32,
            )
        };
        if rc < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

/// Real page-cache operations on the model file.
pub struct LinuxOps {
    pub file: MappedFile,
    pub target: Option<TargetMapping>,
}

impl LinuxOps {
    pub fn new(file: MappedFile) -> Self {
        LinuxOps { file, target: None }
    }
}

impl PageCacheOps for LinuxOps {
    fn prefetch(&mut self, offset: u64, len: u64) -> io::Result<()> {
        fadvise(self.file.fd(), offset, len, libc::POSIX_FADV_WILLNEED)
    }
    fn pin(&mut self, offset: u64, len: u64) -> io::Result<()> {
        let (a, l) = self.file.span(offset, len)?;
        // SAFETY: range inside our mapping.
        check(unsafe { libc::mlock(a, l) })
    }
    fn unpin(&mut self, offset: u64, len: u64) -> io::Result<()> {
        let (a, l) = self.file.span(offset, len)?;
        // SAFETY: range inside our mapping.
        check(unsafe { libc::munlock(a, l) })
    }
    // UNTESTED-ON-HW: needs CAP_SYS_NICE and a running engine.
    fn demote(&mut self, offset: u64, len: u64) -> io::Result<()> {
        let page = self.file.page;
        match &self.target {
            Some(t) => t.madvise(offset, len, libc::MADV_COLD, page),
            None => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "no target process",
            )),
        }
    }
}
