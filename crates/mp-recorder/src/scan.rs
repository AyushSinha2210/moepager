//! Full-file residency scans diffed into page insert/evict events.
//! Good for offline recording; too slow for per-layer control on large
//! files (a scan costs O(file pages)).

use std::io;

use mp_os::ResidencyProbe;
use mp_trace::{PageEvent, PageKind};

pub struct ScanRecorder {
    prev: Vec<bool>,
    cur: Vec<bool>,
    n_pages: u64,
    chunk: u64,
}

impl ScanRecorder {
    pub fn new(file_len: u64, page_size: u64) -> Self {
        let n_pages = file_len.div_ceil(page_size);
        ScanRecorder {
            prev: vec![false; n_pages as usize],
            cur: Vec::new(),
            n_pages,
            chunk: 1 << 16,
        }
    }

    /// Take the initial snapshot without emitting events.
    pub fn baseline(&mut self, probe: &dyn ResidencyProbe) -> io::Result<()> {
        self.snapshot(probe)?;
        std::mem::swap(&mut self.prev, &mut self.cur);
        Ok(())
    }

    fn snapshot(&mut self, probe: &dyn ResidencyProbe) -> io::Result<()> {
        self.cur.clear();
        let mut buf = Vec::new();
        let mut p = 0;
        while p < self.n_pages {
            let n = self.chunk.min(self.n_pages - p);
            probe.resident(p, n, &mut buf)?;
            self.cur.extend_from_slice(&buf);
            p += n;
        }
        Ok(())
    }

    /// Scan once and return run-length-encoded changes since the last scan.
    pub fn tick(&mut self, probe: &dyn ResidencyProbe, t_ns: u64) -> io::Result<Vec<PageEvent>> {
        self.snapshot(probe)?;
        let ev = diff(&self.prev, &self.cur, t_ns);
        std::mem::swap(&mut self.prev, &mut self.cur);
        Ok(ev)
    }
}

/// Runs of pages that changed state between two snapshots.
pub fn diff(prev: &[bool], cur: &[bool], t_ns: u64) -> Vec<PageEvent> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < cur.len() {
        if cur[i] == prev[i] {
            i += 1;
            continue;
        }
        let state = cur[i];
        let start = i;
        while i < cur.len() && cur[i] != prev[i] && cur[i] == state && i - start < u32::MAX as usize
        {
            i += 1;
        }
        out.push(PageEvent {
            t_ns,
            page: start as u64,
            n_pages: (i - start) as u32,
            kind: if state {
                PageKind::Insert
            } else {
                PageKind::Evict
            },
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mp_os::{MockOps, PageCacheOps};

    #[test]
    fn diff_runs() {
        let a = [false, false, true, true, false, false];
        let b = [true, true, true, false, false, true];
        let d = diff(&a, &b, 5);
        assert_eq!(d.len(), 3);
        assert_eq!(
            (d[0].page, d[0].n_pages, d[0].kind),
            (0, 2, PageKind::Insert)
        );
        assert_eq!(
            (d[1].page, d[1].n_pages, d[1].kind),
            (3, 1, PageKind::Evict)
        );
        assert_eq!((d[2].page, d[2].kind), (5, PageKind::Insert));
    }

    #[test]
    fn records_mock_changes() {
        let mut m = MockOps::new(4096);
        let mut r = ScanRecorder::new(64 * 4096, 4096);
        m.prefetch(0, 4096).unwrap();
        r.baseline(&m).unwrap();
        m.prefetch(10 * 4096, 3 * 4096).unwrap();
        let ev = r.tick(&m, 1).unwrap();
        assert_eq!(ev.len(), 1);
        assert_eq!((ev[0].page, ev[0].n_pages), (10, 3));
        assert!(r.tick(&m, 2).unwrap().is_empty());
    }
}
