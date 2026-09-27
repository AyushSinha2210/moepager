//! Eviction orders for the non-pinned part of the simulated page cache.
//!
//! `pos` is the global access index (position in the trace's access
//! sequence), which doubles as a logical clock.

use std::collections::BTreeSet;

pub trait Evictor {
    fn name(&self) -> &'static str;
    /// A unit became resident (and evictable). `accessed` = demand access
    /// rather than a prefetch or an unpin.
    fn insert(&mut self, u: u32, pos: usize, accessed: bool);
    /// A resident evictable unit was accessed.
    fn touch(&mut self, u: u32, pos: usize);
    /// Stop tracking a unit (evicted or pinned).
    fn remove(&mut self, u: u32);
    /// Choose and stop tracking the next victim.
    fn victim(&mut self) -> Option<u32>;
}

/// Least recently used (the kernel's default, approximately; MGLRU is
/// modelled as LRU).
pub struct Lru {
    stamp: Vec<usize>,
    set: BTreeSet<(usize, u32)>,
}

impl Lru {
    pub fn new(n: usize) -> Self {
        Lru {
            stamp: vec![0; n],
            set: BTreeSet::new(),
        }
    }
}

impl Evictor for Lru {
    fn name(&self) -> &'static str {
        "lru"
    }
    fn insert(&mut self, u: u32, pos: usize, _accessed: bool) {
        self.stamp[u as usize] = pos;
        self.set.insert((pos, u));
    }
    fn touch(&mut self, u: u32, pos: usize) {
        self.set.remove(&(self.stamp[u as usize], u));
        self.insert(u, pos, true);
    }
    fn remove(&mut self, u: u32) {
        self.set.remove(&(self.stamp[u as usize], u));
    }
    fn victim(&mut self) -> Option<u32> {
        self.set.pop_first().map(|(_, u)| u)
    }
}

/// Least frequently used with perfect (whole-history) counts; ties broken
/// by recency.
pub struct Lfu {
    count: Vec<u64>,
    stamp: Vec<usize>,
    set: BTreeSet<(u64, usize, u32)>,
}

impl Lfu {
    pub fn new(n: usize) -> Self {
        Lfu {
            count: vec![0; n],
            stamp: vec![0; n],
            set: BTreeSet::new(),
        }
    }
}

impl Evictor for Lfu {
    fn name(&self) -> &'static str {
        "lfu"
    }
    fn insert(&mut self, u: u32, pos: usize, accessed: bool) {
        let i = u as usize;
        if accessed {
            self.count[i] += 1;
        }
        self.stamp[i] = pos;
        self.set.insert((self.count[i], pos, u));
    }
    fn touch(&mut self, u: u32, pos: usize) {
        self.remove(u);
        self.insert(u, pos, true);
    }
    fn remove(&mut self, u: u32) {
        let i = u as usize;
        self.set.remove(&(self.count[i], self.stamp[i], u));
    }
    fn victim(&mut self) -> Option<u32> {
        self.set.pop_first().map(|(_, _, u)| u)
    }
}

/// Belady's MIN: evict the unit whose next use is furthest in the future.
/// Exact for uniform sizes; with variable sizes it is the standard
/// heuristic (exact variable-size MIN is NP-hard), reported as `belady*`.
pub struct Belady {
    /// For each unit, its access positions in increasing order.
    occ: Vec<Vec<usize>>,
    key: Vec<usize>,
    set: BTreeSet<(usize, u32)>,
}

impl Belady {
    pub fn new(n: usize, seq: &[u32]) -> Self {
        let mut occ = vec![Vec::new(); n];
        for (i, &u) in seq.iter().enumerate() {
            occ[u as usize].push(i);
        }
        Belady {
            occ,
            key: vec![0; n],
            set: BTreeSet::new(),
        }
    }

    fn next_use(&self, u: u32, pos: usize) -> usize {
        let o = &self.occ[u as usize];
        let i = o.partition_point(|&p| p <= pos);
        o.get(i).copied().unwrap_or(usize::MAX)
    }
}

impl Evictor for Belady {
    fn name(&self) -> &'static str {
        "belady*"
    }
    fn insert(&mut self, u: u32, pos: usize, accessed: bool) {
        // A prefetch at `pos` precedes the access at `pos` itself.
        let k = if accessed {
            self.next_use(u, pos)
        } else {
            self.next_use(u, pos.wrapping_sub(1))
        };
        self.key[u as usize] = k;
        self.set.insert((k, u));
    }
    fn touch(&mut self, u: u32, pos: usize) {
        self.remove(u);
        self.insert(u, pos, true);
    }
    fn remove(&mut self, u: u32) {
        self.set.remove(&(self.key[u as usize], u));
    }
    fn victim(&mut self) -> Option<u32> {
        self.set.pop_last().map(|(_, u)| u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_order() {
        let mut e = Lru::new(4);
        e.insert(0, 0, true);
        e.insert(1, 1, true);
        e.insert(2, 2, true);
        e.touch(0, 3);
        assert_eq!(e.victim(), Some(1));
        e.remove(2);
        assert_eq!(e.victim(), Some(0));
        assert_eq!(e.victim(), None);
    }

    #[test]
    fn lfu_order() {
        let mut e = Lfu::new(3);
        e.insert(0, 0, true);
        e.touch(0, 1);
        e.insert(1, 2, true);
        e.insert(2, 3, false); // prefetched: count 0
        assert_eq!(e.victim(), Some(2));
        assert_eq!(e.victim(), Some(1));
    }

    #[test]
    fn belady_picks_furthest_next_use() {
        let seq = [0, 1, 2, 0, 1, 0];
        let mut e = Belady::new(3, &seq);
        e.insert(0, 0, true);
        e.insert(1, 1, true);
        e.insert(2, 2, true); // never used again
        assert_eq!(e.victim(), Some(2));
        assert_eq!(e.victim(), Some(1)); // next at 4, vs 0 at 3
    }
}
