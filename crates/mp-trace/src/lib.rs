//! moepager trace format (`.mpt`, version 1). Spec: docs/TRACE_FORMAT.md.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

pub mod steps;

pub use steps::{infer_tokens, Step, Steps};

pub const MAGIC: &[u8; 8] = b"MPTRACE\0";
pub const VERSION: u16 = 1;
pub const EXPERT_RECORD: usize = 16;
pub const PAGE_RECORD: usize = 24;
const MAX_HEADER: u32 = 16 << 20;

#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not an mpt trace (bad magic)")]
    BadMagic,
    #[error("unsupported trace version {0}")]
    Version(u16),
    #[error("expected {expected:?} records, file has {found:?}")]
    Kind {
        expected: RecordKind,
        found: RecordKind,
    },
    #[error("corrupt trace: {0}")]
    Corrupt(String),
    #[error("header json: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordKind {
    Expert = 1,
    Page = 2,
}

impl RecordKind {
    fn from_u16(v: u16) -> Result<Self, TraceError> {
        match v {
            1 => Ok(RecordKind::Expert),
            2 => Ok(RecordKind::Page),
            k => Err(TraceError::Corrupt(format!("record kind {k}"))),
        }
    }
    pub fn size(self) -> usize {
        match self {
            RecordKind::Expert => EXPERT_RECORD,
            RecordKind::Page => PAGE_RECORD,
        }
    }
}

/// Which accesses a trace contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Observation {
    /// Every use of an expert (synthetic or instrumented ground truth).
    #[default]
    Full,
    /// Only uses that missed the page cache (black-box recorders).
    MissOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceHeader {
    pub source: String,
    #[serde(default)]
    pub observation: Observation,
    pub n_layers: u32,
    pub n_experts: u32,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default = "default_page_size")]
    pub page_size: u64,
    #[serde(default)]
    pub model_file: Option<String>,
    #[serde(default)]
    pub model_size: Option<u64>,
    #[serde(default)]
    pub params: serde_json::Value,
}

fn default_page_size() -> u64 {
    4096
}

impl TraceHeader {
    pub fn new(source: &str, n_layers: u32, n_experts: u32) -> Self {
        TraceHeader {
            source: source.into(),
            observation: Observation::Full,
            n_layers,
            n_experts,
            top_k: None,
            page_size: 4096,
            model_file: None,
            model_size: None,
            params: serde_json::Value::Null,
        }
    }
}

/// One use (or miss) of expert unit (layer, expert).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertEvent {
    pub t_ns: u64,
    pub token: u32,
    pub layer: u16,
    pub expert: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageKind {
    Insert = 1,
    Evict = 2,
    Fault = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageEvent {
    pub t_ns: u64,
    pub page: u64,
    pub n_pages: u32,
    pub kind: PageKind,
}

impl ExpertEvent {
    fn encode(&self) -> [u8; EXPERT_RECORD] {
        let mut b = [0u8; EXPERT_RECORD];
        b[0..8].copy_from_slice(&self.t_ns.to_le_bytes());
        b[8..12].copy_from_slice(&self.token.to_le_bytes());
        b[12..14].copy_from_slice(&self.layer.to_le_bytes());
        b[14..16].copy_from_slice(&self.expert.to_le_bytes());
        b
    }
    fn decode(b: &[u8]) -> Self {
        ExpertEvent {
            t_ns: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            token: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            layer: u16::from_le_bytes(b[12..14].try_into().unwrap()),
            expert: u16::from_le_bytes(b[14..16].try_into().unwrap()),
        }
    }
}

impl PageEvent {
    fn encode(&self) -> [u8; PAGE_RECORD] {
        let mut b = [0u8; PAGE_RECORD];
        b[0..8].copy_from_slice(&self.t_ns.to_le_bytes());
        b[8..16].copy_from_slice(&self.page.to_le_bytes());
        b[16..20].copy_from_slice(&self.n_pages.to_le_bytes());
        b[20] = self.kind as u8;
        b
    }
    fn decode(b: &[u8]) -> Result<Self, TraceError> {
        let kind = match b[20] {
            1 => PageKind::Insert,
            2 => PageKind::Evict,
            3 => PageKind::Fault,
            k => return Err(TraceError::Corrupt(format!("page event kind {k}"))),
        };
        if b[21..24] != [0, 0, 0] {
            return Err(TraceError::Corrupt("reserved bytes not zero".into()));
        }
        Ok(PageEvent {
            t_ns: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            page: u64::from_le_bytes(b[8..16].try_into().unwrap()),
            n_pages: u32::from_le_bytes(b[16..20].try_into().unwrap()),
            kind,
        })
    }
}

/// Streaming writer.
pub struct TraceWriter<W: Write> {
    w: W,
    kind: RecordKind,
    count: u64,
}

impl<W: Write> TraceWriter<W> {
    pub fn new(mut w: W, kind: RecordKind, header: &TraceHeader) -> Result<Self, TraceError> {
        let json = serde_json::to_vec(header)?;
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&(kind as u16).to_le_bytes())?;
        w.write_all(&(json.len() as u32).to_le_bytes())?;
        w.write_all(&json)?;
        Ok(TraceWriter { w, kind, count: 0 })
    }

    pub fn expert(&mut self, e: &ExpertEvent) -> Result<(), TraceError> {
        debug_assert_eq!(self.kind, RecordKind::Expert);
        self.w.write_all(&e.encode())?;
        self.count += 1;
        Ok(())
    }

    pub fn page(&mut self, e: &PageEvent) -> Result<(), TraceError> {
        debug_assert_eq!(self.kind, RecordKind::Page);
        self.w.write_all(&e.encode())?;
        self.count += 1;
        Ok(())
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn finish(mut self) -> Result<W, TraceError> {
        self.w.flush()?;
        Ok(self.w)
    }
}

/// Create a buffered writer on a file.
pub fn create(
    path: &Path,
    kind: RecordKind,
    header: &TraceHeader,
) -> Result<TraceWriter<BufWriter<File>>, TraceError> {
    TraceWriter::new(BufWriter::new(File::create(path)?), kind, header)
}

/// Read the preamble + header, returning the record kind.
pub fn read_header<R: Read>(r: &mut R) -> Result<(TraceHeader, RecordKind), TraceError> {
    let mut pre = [0u8; 16];
    r.read_exact(&mut pre)?;
    if &pre[0..8] != MAGIC {
        return Err(TraceError::BadMagic);
    }
    let version = u16::from_le_bytes([pre[8], pre[9]]);
    if version != VERSION {
        return Err(TraceError::Version(version));
    }
    let kind = RecordKind::from_u16(u16::from_le_bytes([pre[10], pre[11]]))?;
    let hlen = u32::from_le_bytes(pre[12..16].try_into().unwrap());
    if hlen > MAX_HEADER {
        return Err(TraceError::Corrupt(format!("header length {hlen}")));
    }
    let mut json = vec![0u8; hlen as usize];
    r.read_exact(&mut json)?;
    Ok((serde_json::from_slice(&json)?, kind))
}

fn read_records<R: Read>(r: &mut R, size: usize) -> Result<Vec<u8>, TraceError> {
    let mut body = Vec::new();
    r.read_to_end(&mut body)?;
    if body.len() % size != 0 {
        return Err(TraceError::Corrupt(format!(
            "{} trailing bytes (partial record)",
            body.len() % size
        )));
    }
    Ok(body)
}

pub fn read_expert<R: Read>(mut r: R) -> Result<(TraceHeader, Vec<ExpertEvent>), TraceError> {
    let (h, kind) = read_header(&mut r)?;
    if kind != RecordKind::Expert {
        return Err(TraceError::Kind {
            expected: RecordKind::Expert,
            found: kind,
        });
    }
    let body = read_records(&mut r, EXPERT_RECORD)?;
    Ok((
        h,
        body.chunks_exact(EXPERT_RECORD)
            .map(ExpertEvent::decode)
            .collect(),
    ))
}

pub fn read_page<R: Read>(mut r: R) -> Result<(TraceHeader, Vec<PageEvent>), TraceError> {
    let (h, kind) = read_header(&mut r)?;
    if kind != RecordKind::Page {
        return Err(TraceError::Kind {
            expected: RecordKind::Page,
            found: kind,
        });
    }
    let body = read_records(&mut r, PAGE_RECORD)?;
    let ev = body
        .chunks_exact(PAGE_RECORD)
        .map(PageEvent::decode)
        .collect::<Result<_, _>>()?;
    Ok((h, ev))
}

pub fn read_expert_file(path: &Path) -> Result<(TraceHeader, Vec<ExpertEvent>), TraceError> {
    read_expert(BufReader::new(File::open(path)?))
}

pub fn read_page_file(path: &Path) -> Result<(TraceHeader, Vec<PageEvent>), TraceError> {
    read_page(BufReader::new(File::open(path)?))
}

pub fn write_expert_file(
    path: &Path,
    header: &TraceHeader,
    events: &[ExpertEvent],
) -> Result<(), TraceError> {
    let mut w = create(path, RecordKind::Expert, header)?;
    for e in events {
        w.expert(e)?;
    }
    w.finish()?;
    Ok(())
}

/// CSV export (for pandas): writes `t_ns,token,layer,expert` or `t_ns,page,n_pages,kind`.
pub fn to_csv<R: Read, W: Write>(mut r: R, mut w: W) -> Result<u64, TraceError> {
    let (_, kind) = read_header(&mut r)?;
    let body = read_records(&mut r, kind.size())?;
    let mut n = 0;
    match kind {
        RecordKind::Expert => {
            writeln!(w, "t_ns,token,layer,expert")?;
            for c in body.chunks_exact(EXPERT_RECORD) {
                let e = ExpertEvent::decode(c);
                writeln!(w, "{},{},{},{}", e.t_ns, e.token, e.layer, e.expert)?;
                n += 1;
            }
        }
        RecordKind::Page => {
            writeln!(w, "t_ns,page,n_pages,kind")?;
            for c in body.chunks_exact(PAGE_RECORD) {
                let e = PageEvent::decode(c)?;
                let k = match e.kind {
                    PageKind::Insert => "insert",
                    PageKind::Evict => "evict",
                    PageKind::Fault => "fault",
                };
                writeln!(w, "{},{},{},{}", e.t_ns, e.page, e.n_pages, k)?;
                n += 1;
            }
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> TraceHeader {
        let mut h = TraceHeader::new("test", 4, 8);
        h.top_k = Some(2);
        h.observation = Observation::MissOnly;
        h.params = serde_json::json!({"seed": 7});
        h
    }

    #[test]
    fn expert_round_trip() {
        let ev: Vec<ExpertEvent> = (0..100u32)
            .map(|i| ExpertEvent {
                t_ns: i as u64 * 1000,
                token: i / 8,
                layer: (i % 4) as u16,
                expert: (i % 8) as u16,
            })
            .collect();
        let mut buf = Vec::new();
        let mut w = TraceWriter::new(&mut buf, RecordKind::Expert, &header()).unwrap();
        for e in &ev {
            w.expert(e).unwrap();
        }
        assert_eq!(w.count(), 100);
        w.finish().unwrap();
        let (h, back) = read_expert(&buf[..]).unwrap();
        assert_eq!(h, header());
        assert_eq!(back, ev);
    }

    #[test]
    fn page_round_trip_and_csv() {
        let ev = [
            PageEvent {
                t_ns: 1,
                page: 10,
                n_pages: 4,
                kind: PageKind::Insert,
            },
            PageEvent {
                t_ns: 2,
                page: u64::MAX,
                n_pages: 1,
                kind: PageKind::Evict,
            },
        ];
        let mut buf = Vec::new();
        let mut w = TraceWriter::new(&mut buf, RecordKind::Page, &header()).unwrap();
        for e in &ev {
            w.page(e).unwrap();
        }
        w.finish().unwrap();
        let (_, back) = read_page(&buf[..]).unwrap();
        assert_eq!(back, ev);
        let mut csv = Vec::new();
        assert_eq!(to_csv(&buf[..], &mut csv).unwrap(), 2);
        let s = String::from_utf8(csv).unwrap();
        assert!(s.starts_with("t_ns,page,n_pages,kind\n1,10,4,insert\n"));
    }

    #[test]
    fn rejects_corruption() {
        let mut buf = Vec::new();
        let w = TraceWriter::new(&mut buf, RecordKind::Expert, &header()).unwrap();
        w.finish().unwrap();
        // wrong kind
        assert!(matches!(read_page(&buf[..]), Err(TraceError::Kind { .. })));
        // partial record
        let mut b2 = buf.clone();
        b2.extend([0u8; 5]);
        assert!(matches!(read_expert(&b2[..]), Err(TraceError::Corrupt(_))));
        // bad magic / version
        let mut b3 = buf.clone();
        b3[0] = b'X';
        assert!(matches!(read_expert(&b3[..]), Err(TraceError::BadMagic)));
        let mut b4 = buf.clone();
        b4[8] = 9;
        assert!(matches!(read_expert(&b4[..]), Err(TraceError::Version(9))));
    }

    #[test]
    fn unknown_header_fields_are_ignored() {
        let json = br#"{"source":"x","n_layers":1,"n_experts":2,"future_field":true}"#;
        let mut b = MAGIC.to_vec();
        b.extend(1u16.to_le_bytes());
        b.extend(1u16.to_le_bytes());
        b.extend((json.len() as u32).to_le_bytes());
        b.extend(json);
        let (h, ev) = read_expert(&b[..]).unwrap();
        assert_eq!(h.page_size, 4096);
        assert_eq!(h.observation, Observation::Full);
        assert!(ev.is_empty());
    }
}
