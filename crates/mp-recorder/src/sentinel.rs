//! Sentinel polling: one probe page per expert slice, cheap enough to run
//! continuously without root. Emits a (miss-only) expert event when a
//! unit's sentinel enters the page cache.

use std::io;

use mp_gguf::ExpertMap;
use mp_os::ResidencyProbe;
use mp_trace::ExpertEvent;

struct Sentinel {
    page: u64,
    layer: u16,
    expert: u16,
}

pub struct SentinelRecorder {
    sentinels: Vec<Sentinel>,
    prev: Vec<bool>,
    /// Units with at least one unreliable (too small) sentinel.
    pub unreliable_units: usize,
}

impl SentinelRecorder {
    /// One sentinel per weight slice of every unit.
    pub fn new(map: &ExpertMap) -> Self {
        let mut sentinels = Vec::new();
        let mut unreliable = 0;
        for (id, u) in map.units.iter().enumerate() {
            let s = map.unit_sentinels(id as u32);
            if s.iter().any(|&(_, ok)| !ok) {
                unreliable += 1;
            }
            for (page, _) in s {
                sentinels.push(Sentinel {
                    page,
                    layer: u.layer as u16,
                    expert: u.expert as u16,
                });
            }
        }
        let n = sentinels.len();
        SentinelRecorder {
            sentinels,
            prev: vec![false; n],
            unreliable_units: unreliable,
        }
    }

    pub fn len(&self) -> usize {
        self.sentinels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sentinels.is_empty()
    }

    /// Record current state without emitting events.
    pub fn baseline(&mut self, probe: &dyn ResidencyProbe) -> io::Result<()> {
        for (i, s) in self.sentinels.iter().enumerate() {
            self.prev[i] = probe.page_resident(s.page)?;
        }
        Ok(())
    }

    /// Poll every sentinel; emit one event per unit whose sentinel(s)
    /// became resident. Token is left as 0 (see `mp_trace::infer_tokens`).
    pub fn tick(&mut self, probe: &dyn ResidencyProbe, t_ns: u64) -> io::Result<Vec<ExpertEvent>> {
        let mut out: Vec<ExpertEvent> = Vec::new();
        for (i, s) in self.sentinels.iter().enumerate() {
            let now = probe.page_resident(s.page)?;
            if now
                && !self.prev[i]
                && !out
                    .iter()
                    .any(|e| e.layer == s.layer && e.expert == s.expert)
            {
                out.push(ExpertEvent {
                    t_ns,
                    token: 0,
                    layer: s.layer,
                    expert: s.expert,
                });
            }
            self.prev[i] = now;
        }
        Ok(out)
    }
}
