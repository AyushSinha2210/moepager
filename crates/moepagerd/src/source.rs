//! Event sources for the daemon.

use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use mp_gguf::ExpertMap;
use mp_os::MappedFile;
use mp_recorder::SentinelRecorder;
use mp_trace::ExpertEvent;

pub trait EventSource {
    /// Next batch of observations; `None` when the source is exhausted.
    fn poll(&mut self) -> io::Result<Option<Vec<ExpertEvent>>>;
}

/// Replays a recorded trace (dry runs, tests).
pub struct TraceSource {
    events: std::vec::IntoIter<ExpertEvent>,
}

impl TraceSource {
    pub fn open(path: &Path) -> io::Result<Self> {
        let (_, mut ev) = mp_trace::read_expert_file(path)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        ev.sort_by_key(|e| (e.t_ns, e.token, e.layer));
        Ok(TraceSource {
            events: ev.into_iter(),
        })
    }

    pub fn from_events(ev: Vec<ExpertEvent>) -> Self {
        TraceSource {
            events: ev.into_iter(),
        }
    }
}

impl EventSource for TraceSource {
    fn poll(&mut self) -> io::Result<Option<Vec<ExpertEvent>>> {
        Ok(self.events.next().map(|e| vec![e]))
    }
}

/// Live, unprivileged: polls sentinel pages of the model file.
// UNTESTED-ON-HW: exercised against the replayer only, not llama.cpp.
pub struct SentinelSource {
    probe: MappedFile,
    rec: SentinelRecorder,
    start: Instant,
    interval: Duration,
    until: Option<Instant>,
}

impl SentinelSource {
    pub fn open(
        gguf: &Path,
        map: &ExpertMap,
        interval: Duration,
        duration: Option<Duration>,
    ) -> io::Result<Self> {
        let probe = MappedFile::open(gguf)?;
        let mut rec = SentinelRecorder::new(map);
        rec.baseline(&probe)?;
        let start = Instant::now();
        Ok(SentinelSource {
            probe,
            rec,
            start,
            interval,
            until: duration.map(|d| start + d),
        })
    }
}

impl EventSource for SentinelSource {
    fn poll(&mut self) -> io::Result<Option<Vec<ExpertEvent>>> {
        if self.until.is_some_and(|u| Instant::now() >= u) {
            return Ok(None);
        }
        std::thread::sleep(self.interval);
        let t = self.start.elapsed().as_nanos() as u64;
        self.rec.tick(&self.probe, t).map(Some)
    }
}
