//! In-memory mock: records calls and simulates residency.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::io;

use crate::{PageCacheOps, ResidencyProbe};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpCall {
    Prefetch(u64, u64),
    Pin(u64, u64),
    Unpin(u64, u64),
    Demote(u64, u64),
}

/// Mock page cache. Prefetch and pin make pages resident; demote evicts
/// unpinned pages. `budget_pin_bytes` emulates RLIMIT_MEMLOCK.
pub struct MockOps {
    pub page_size: u64,
    pub calls: Vec<OpCall>,
    pub resident: RefCell<BTreeSet<u64>>,
    pub pinned: BTreeSet<u64>,
    pub budget_pin_bytes: u64,
    pub demote_supported: bool,
}

impl MockOps {
    pub fn new(page_size: u64) -> Self {
        MockOps {
            page_size,
            calls: Vec::new(),
            resident: RefCell::new(BTreeSet::new()),
            pinned: BTreeSet::new(),
            budget_pin_bytes: u64::MAX,
            demote_supported: true,
        }
    }

    fn pages(&self, off: u64, len: u64) -> std::ops::RangeInclusive<u64> {
        off / self.page_size..=(off + len.max(1) - 1) / self.page_size
    }

    pub fn pinned_bytes(&self) -> u64 {
        self.pinned.len() as u64 * self.page_size
    }
}

impl PageCacheOps for MockOps {
    fn prefetch(&mut self, offset: u64, len: u64) -> io::Result<()> {
        self.calls.push(OpCall::Prefetch(offset, len));
        let r = self.pages(offset, len);
        self.resident.borrow_mut().extend(r);
        Ok(())
    }
    fn pin(&mut self, offset: u64, len: u64) -> io::Result<()> {
        let r = self.pages(offset, len);
        let new = r.clone().filter(|p| !self.pinned.contains(p)).count() as u64;
        if self.pinned_bytes() + new * self.page_size > self.budget_pin_bytes {
            return Err(io::Error::from_raw_os_error(libc::ENOMEM));
        }
        self.calls.push(OpCall::Pin(offset, len));
        self.pinned.extend(r.clone());
        self.resident.borrow_mut().extend(r);
        Ok(())
    }
    fn unpin(&mut self, offset: u64, len: u64) -> io::Result<()> {
        self.calls.push(OpCall::Unpin(offset, len));
        for p in self.pages(offset, len) {
            self.pinned.remove(&p);
        }
        Ok(())
    }
    fn demote(&mut self, offset: u64, len: u64) -> io::Result<()> {
        if !self.demote_supported {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        }
        self.calls.push(OpCall::Demote(offset, len));
        for p in self.pages(offset, len) {
            if !self.pinned.contains(&p) {
                self.resident.borrow_mut().remove(&p);
            }
        }
        Ok(())
    }
}

impl ResidencyProbe for MockOps {
    fn resident(&self, first: u64, n: u64, out: &mut Vec<bool>) -> io::Result<()> {
        out.clear();
        let r = self.resident.borrow();
        out.extend((first..first + n).map(|p| r.contains(&p)));
        Ok(())
    }
    fn page_size(&self) -> u64 {
        self.page_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_semantics() {
        let mut m = MockOps::new(4096);
        m.prefetch(0, 8192).unwrap();
        assert!(m.page_resident(1).unwrap() && !m.page_resident(2).unwrap());
        m.pin(8192, 1).unwrap();
        m.demote(0, 3 * 4096).unwrap();
        assert!(!m.page_resident(0).unwrap());
        assert!(m.page_resident(2).unwrap(), "pinned pages survive demotion");
        m.budget_pin_bytes = 4096;
        assert!(m.pin(0, 4096).is_err());
        m.demote_supported = false;
        assert!(m.demote(0, 1).is_err());
        assert_eq!(m.calls.len(), 3);
    }
}
