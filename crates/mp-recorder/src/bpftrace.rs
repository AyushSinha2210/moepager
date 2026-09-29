//! Ingest the output of `bpf/moepager.bt`:
//! `A <t_ns> <ino> <index> <order>` (added) / `D <t_ns> <ino> <index> <order>` (deleted).

use mp_trace::{PageEvent, PageKind};

/// Parse bpftrace output for one inode. Timestamps are rebased to the
/// first event. Malformed lines (bpftrace banners etc.) are counted and skipped.
pub fn parse(text: &str, inode: u64) -> (Vec<PageEvent>, usize) {
    let mut out = Vec::new();
    let mut bad = 0;
    let mut t0: Option<u64> = None;
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let parsed = (|| {
            if f.len() != 5 {
                return None;
            }
            let kind = match f[0] {
                "A" => PageKind::Insert,
                "D" => PageKind::Evict,
                _ => return None,
            };
            let t: u64 = f[1].parse().ok()?;
            let ino: u64 = f[2].parse().ok()?;
            let index: u64 = f[3].parse().ok()?;
            let order: u32 = f[4].parse().ok()?;
            (order < 32).then_some((kind, t, ino, index, order))
        })();
        match parsed {
            Some((kind, t, ino, index, order)) if ino == inode => {
                let base = *t0.get_or_insert(t);
                out.push(PageEvent {
                    t_ns: t.saturating_sub(base),
                    page: index,
                    n_pages: 1 << order,
                    kind,
                });
            }
            Some(_) => {}
            None if line.trim().is_empty() || line.starts_with("Attaching") => {}
            None => bad += 1,
        }
    }
    out.sort_by_key(|e| e.t_ns);
    (out, bad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sample_output() {
        let txt = "Attaching 2 probes...\nA 1000 42 10 0\nA 1500 42 16 2\nA 1600 7 1 0\nD 2000 42 10 0\ngarbage line\n";
        let (ev, bad) = parse(txt, 42);
        assert_eq!(bad, 1);
        assert_eq!(ev.len(), 3);
        assert_eq!(
            ev[0],
            PageEvent {
                t_ns: 0,
                page: 10,
                n_pages: 1,
                kind: PageKind::Insert
            }
        );
        assert_eq!(ev[1].n_pages, 4);
        assert_eq!(ev[2].kind, PageKind::Evict);
    }
}
