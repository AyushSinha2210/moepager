//! Page-level events → expert-level events using the GGUF map.

use std::collections::HashMap;

use mp_gguf::ExpertMap;
use mp_trace::{ExpertEvent, PageEvent, PageKind};

#[derive(Debug, Clone)]
pub struct ConvertConfig {
    /// Only pages at least this far inside a slice count. Reading one expert
    /// pulls neighbouring pages in: fault read-around, sequential readahead,
    /// and on btrfs with compression whole 128 KiB compressed extents
    /// (measured: reading a 544 KiB slice inserted 17 pages of its
    /// neighbour). Interior pages are immune to spill of up to `margin`.
    pub margin: u64,
    /// A unit counts as used when at least this many interior pages were
    /// inserted within `window_ns` (capped at the interior size).
    pub min_pages: u32,
    pub window_ns: u64,
}

impl Default for ConvertConfig {
    fn default() -> Self {
        ConvertConfig {
            margin: mp_gguf::map::SENTINEL_GUARD,
            min_pages: 8,
            window_ns: 50_000_000,
        }
    }
}

/// Interior page range of a slice: [first, last] at least `margin` from
/// both edges, or the middle page when the slice is too small.
fn interior(off: u64, len: u64, margin: u64, page: u64) -> (u64, u64) {
    if len > 2 * margin + 2 * page {
        (
            (off + margin).div_ceil(page),
            (off + len - margin) / page - 1,
        )
    } else {
        let mid = (off + len / 2) / page;
        (mid, mid)
    }
}

/// Convert insert events into miss-only expert events (token = 0; run
/// `mp_trace::infer_tokens` afterwards). Events must be time-ordered.
pub fn page_to_expert(
    events: &[PageEvent],
    map: &ExpertMap,
    cfg: &ConvertConfig,
) -> Vec<ExpertEvent> {
    let idx = map.page_index();
    let mut need = vec![0u32; map.units.len()];
    let ranges: Vec<Vec<(u64, u64)>> = map
        .units
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let r: Vec<(u64, u64)> = u
                .slices
                .iter()
                .map(|s| interior(s.offset, s.len, cfg.margin, map.page_size))
                .collect();
            let total: u64 = r.iter().map(|(a, b)| b - a + 1).sum();
            need[i] = (cfg.min_pages as u64).min(total).max(1) as u32;
            r
        })
        .collect();
    // unit -> (window start, interior pages counted, already emitted)
    let mut state: HashMap<u32, (u64, u32, bool)> = HashMap::new();
    let mut out = Vec::new();
    for e in events.iter().filter(|e| e.kind == PageKind::Insert) {
        for p in e.page..e.page + e.n_pages as u64 {
            for u in idx.units(p) {
                if !ranges[u as usize].iter().any(|&(a, b)| a <= p && p <= b) {
                    continue;
                }
                let s = state.entry(u).or_insert((e.t_ns, 0, false));
                if e.t_ns.saturating_sub(s.0) > cfg.window_ns {
                    *s = (e.t_ns, 0, false);
                }
                s.1 += 1;
                if !s.2 && s.1 >= need[u as usize] {
                    s.2 = true;
                    let unit = &map.units[u as usize];
                    out.push(ExpertEvent {
                        t_ns: e.t_ns,
                        token: 0,
                        layer: unit.layer as u16,
                        expert: unit.expert as u16,
                    });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interior_ranges() {
        // 1 MiB slice at a 32-byte offset: interior skips 128 KiB at both ends.
        let (a, b) = interior(32, 1 << 20, 128 << 10, 4096);
        assert!(a * 4096 >= 32 + (128 << 10));
        assert!((b + 1) * 4096 <= 32 + (1 << 20) - (128 << 10));
        // Tiny slice: middle page only.
        assert_eq!(interior(8192, 4096, 128 << 10, 4096), (2, 2));
    }
}
