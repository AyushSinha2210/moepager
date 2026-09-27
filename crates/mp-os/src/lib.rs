//! OS-facing page-cache operations and residency probes, behind traits so
//! policy code can be tested with [`MockOps`].
//!
//! Semantics verified on Linux 6.19 (IDEA_REVIEW §1.3–1.4):
//! * `posix_fadvise(WILLNEED)` populates the shared page cache, unprivileged.
//! * `posix_fadvise(DONTNEED)` does **not** drop pages mapped by any process.
//! * `mincore` on our own mapping reports page-cache residency, even for
//!   files we do not own; `cachestat` needs ownership or write access.
//! * `mlock` through our own `MAP_SHARED` mapping pins the shared folio.
//! * Demotion of another process's pages needs `process_madvise` with
//!   `CAP_SYS_NICE`.


pub mod maps;
pub mod mock;

pub use maps::{find_file_mapping, kernel_dev_of_mapping, parse_maps_line, MapEntry};
pub use mock::{MockOps, OpCall};

use std::io;

/// Page-cache actuators on byte ranges of the model file.
pub trait PageCacheOps {
    /// Start asynchronous readahead (no waiting).
    fn prefetch(&mut self, offset: u64, len: u64) -> io::Result<()>;
    /// Make the range resident and unevictable.
    fn pin(&mut self, offset: u64, len: u64) -> io::Result<()>;
    fn unpin(&mut self, offset: u64, len: u64) -> io::Result<()>;
    /// Hint that the range is cold (evict first). May be unsupported.
    fn demote(&mut self, offset: u64, len: u64) -> io::Result<()>;
}

/// Page-cache residency queries on the model file.
pub trait ResidencyProbe {
    /// Residency of pages `[first, first + n)`; pages beyond EOF read as false.
    fn resident(&self, first: u64, n: u64, out: &mut Vec<bool>) -> io::Result<()>;

    fn page_resident(&self, page: u64) -> io::Result<bool> {
        let mut v = Vec::with_capacity(1);
        self.resident(page, 1, &mut v)?;
        Ok(v[0])
    }

    fn page_size(&self) -> u64;
}
