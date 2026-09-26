//! GGUF v2/v3 header parser (little-endian only).
//!
//! Layout: magic "GGUF", u32 version, u64 n_tensors, u64 n_kv, then n_kv
//! metadata entries, then n_tensors tensor infos, padding to
//! `general.alignment` (default 32), then the data section. Tensor offsets
//! are relative to the data section.

use std::collections::BTreeMap;
use std::io::Read;

use crate::types::GgmlType;

/// Upper bounds that reject corrupt headers before allocating.
const MAX_STRING: u64 = 1 << 26;
const MAX_TENSORS: u64 = 1 << 20;
const MAX_KV: u64 = 1 << 20;
const MAX_DIMS: u32 = 4;
/// Arrays longer than this are skipped (only their length is kept).
const KEEP_ARRAY: u64 = 64;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("unsupported GGUF version {0} (supported: 2, 3)")]
    Version(u32),
    #[error("corrupt header: {0}")]
    Corrupt(String),
}

/// A metadata value. Large arrays (e.g. tokenizer vocabularies) are not
/// materialised; see [`MetaValue::Array`].
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// `items` is empty when `len > 64`.
    Array {
        elem_type: u32,
        len: u64,
        items: Vec<MetaValue>,
    },
}

impl MetaValue {
    /// Integer view of any integer-typed value.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            MetaValue::U8(v) => Some(v as u64),
            MetaValue::U16(v) => Some(v as u64),
            MetaValue::U32(v) => Some(v as u64),
            MetaValue::U64(v) => Some(v),
            MetaValue::I8(v) => u64::try_from(v).ok(),
            MetaValue::I16(v) => u64::try_from(v).ok(),
            MetaValue::I32(v) => u64::try_from(v).ok(),
            MetaValue::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// One tensor descriptor from the header.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// ne[0] is the fastest-varying dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Offset relative to the start of the data section.
    pub offset: u64,
}

/// Parsed GGUF header.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub alignment: u64,
    pub metadata: BTreeMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset of the data section.
    pub data_offset: u64,
}

impl GgufHeader {
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    pub fn arch(&self) -> Option<&str> {
        self.get("general.architecture").and_then(MetaValue::as_str)
    }

    /// `<arch>.<suffix>` as an integer, e.g. `arch_u64("expert_count")`.
    pub fn arch_u64(&self, suffix: &str) -> Option<u64> {
        let arch = self.arch()?;
        self.get(&format!("{arch}.{suffix}"))
            .and_then(MetaValue::as_u64)
    }

    /// Byte size of a tensor: from the type table, or, for unknown types,
    /// from the distance to the next tensor in file order (an upper bound
    /// that may include alignment padding).
    pub fn tensor_nbytes(&self, idx: usize) -> Option<u64> {
        let t = &self.tensors[idx];
        if let Some(n) = t.ty.nbytes(&t.dims) {
            return Some(n);
        }
        let next = self
            .tensors
            .iter()
            .map(|o| o.offset)
            .filter(|&o| o > t.offset)
            .min()?;
        Some(next - t.offset)
    }

    /// Absolute file offset of a tensor.
    pub fn abs_offset(&self, idx: usize) -> u64 {
        self.data_offset + self.tensors[idx].offset
    }
}

struct Rd<R> {
    r: R,
    pos: u64,
}

impl<R: Read> Rd<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b)?;
        self.pos += N as u64;
        Ok(b)
    }
    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.bytes::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.bytes()?))
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String, GgufError> {
        let n = self.u64()?;
        if n > MAX_STRING {
            return Err(GgufError::Corrupt(format!("string length {n}")));
        }
        let mut v = vec![0u8; n as usize];
        self.r.read_exact(&mut v)?;
        self.pos += n;
        String::from_utf8(v).map_err(|_| GgufError::Corrupt("invalid utf-8".into()))
    }
    fn value(&mut self, ty: u32, depth: u32) -> Result<MetaValue, GgufError> {
        Ok(match ty {
            0 => MetaValue::U8(self.u8()?),
            1 => MetaValue::I8(self.u8()? as i8),
            2 => MetaValue::U16(self.u16()?),
            3 => MetaValue::I16(self.u16()? as i16),
            4 => MetaValue::U32(self.u32()?),
            5 => MetaValue::I32(self.u32()? as i32),
            6 => MetaValue::F32(f32::from_bits(self.u32()?)),
            7 => MetaValue::Bool(self.u8()? != 0),
            8 => MetaValue::Str(self.string()?),
            9 => {
                if depth > 2 {
                    return Err(GgufError::Corrupt("nested arrays too deep".into()));
                }
                let elem_type = self.u32()?;
                let len = self.u64()?;
                let keep = len <= KEEP_ARRAY;
                let mut items = Vec::new();
                for _ in 0..len {
                    let v = self.value(elem_type, depth + 1)?;
                    if keep {
                        items.push(v);
                    }
                }
                MetaValue::Array {
                    elem_type,
                    len,
                    items,
                }
            }
            10 => MetaValue::U64(self.u64()?),
            11 => MetaValue::I64(self.u64()? as i64),
            12 => MetaValue::F64(f64::from_bits(self.u64()?)),
            t => return Err(GgufError::Corrupt(format!("unknown metadata type {t}"))),
        })
    }
}

/// Parse a GGUF header from a reader positioned at the start of the file.
/// Reads only as far as the end of the tensor infos.
pub fn parse_header<R: Read>(r: R) -> Result<GgufHeader, GgufError> {
    let mut rd = Rd { r, pos: 0 };
    if &rd.bytes::<4>()? != b"GGUF" {
        return Err(GgufError::BadMagic);
    }
    let version = rd.u32()?;
    if !(2..=3).contains(&version) {
        return Err(GgufError::Version(version));
    }
    let n_tensors = rd.u64()?;
    let n_kv = rd.u64()?;
    if n_tensors > MAX_TENSORS || n_kv > MAX_KV {
        return Err(GgufError::Corrupt(format!(
            "{n_tensors} tensors / {n_kv} kv"
        )));
    }
    let mut metadata = BTreeMap::new();
    for _ in 0..n_kv {
        let key = rd.string()?;
        let ty = rd.u32()?;
        let v = rd.value(ty, 0)?;
        metadata.insert(key, v);
    }
    let alignment = metadata
        .get("general.alignment")
        .and_then(MetaValue::as_u64)
        .unwrap_or(32);
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::Corrupt(format!("alignment {alignment}")));
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = rd.string()?;
        let nd = rd.u32()?;
        if nd == 0 || nd > MAX_DIMS {
            return Err(GgufError::Corrupt(format!("tensor {name}: {nd} dims")));
        }
        let dims = (0..nd).map(|_| rd.u64()).collect::<Result<Vec<_>, _>>()?;
        let ty = GgmlType(rd.u32()?);
        let offset = rd.u64()?;
        if offset % alignment != 0 {
            return Err(GgufError::Corrupt(format!(
                "tensor {name}: unaligned offset {offset}"
            )));
        }
        tensors.push(TensorInfo {
            name,
            dims,
            ty,
            offset,
        });
    }
    let data_offset = rd.pos.div_ceil(alignment) * alignment;
    Ok(GgufHeader {
        version,
        alignment,
        metadata,
        tensors,
        data_offset,
    })
}

/// Parse the header of a GGUF file on disk.
pub fn parse_file(path: &std::path::Path) -> Result<GgufHeader, GgufError> {
    let f = std::fs::File::open(path)?;
    parse_header(std::io::BufReader::with_capacity(1 << 20, f))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(b: &mut Vec<u8>, x: &str) {
        b.extend((x.len() as u64).to_le_bytes());
        b.extend(x.as_bytes());
    }

    fn minimal(version: u32) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(version.to_le_bytes());
        b.extend(1u64.to_le_bytes()); // tensors
        b.extend(2u64.to_le_bytes()); // kv
        s(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        s(&mut b, "x");
        s(&mut b, "x.expert_count");
        b.extend(4u32.to_le_bytes());
        b.extend(8u32.to_le_bytes());
        s(&mut b, "t");
        b.extend(1u32.to_le_bytes());
        b.extend(64u64.to_le_bytes());
        b.extend(0u32.to_le_bytes()); // F32
        b.extend(0u64.to_le_bytes());
        b
    }

    #[test]
    fn parses_minimal_header() {
        let b = minimal(3);
        let h = parse_header(&b[..]).unwrap();
        assert_eq!(h.version, 3);
        assert_eq!(h.arch(), Some("x"));
        assert_eq!(h.arch_u64("expert_count"), Some(8));
        assert_eq!(h.tensors[0].dims, vec![64]);
        assert_eq!(h.data_offset % 32, 0);
        assert!(h.data_offset >= b.len() as u64);
        assert_eq!(h.tensor_nbytes(0), Some(256));
    }

    #[test]
    fn rejects_bad_magic_version_and_truncation() {
        assert!(matches!(
            parse_header(&b"GGUX\x03\0\0\0"[..]),
            Err(GgufError::BadMagic)
        ));
        assert!(matches!(
            parse_header(&minimal(1)[..]),
            Err(GgufError::Version(1))
        ));
        let b = minimal(3);
        for cut in [5, 20, b.len() - 1] {
            assert!(parse_header(&b[..cut]).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn rejects_absurd_string_length() {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(1u64.to_le_bytes());
        b.extend(u64::MAX.to_le_bytes());
        assert!(matches!(parse_header(&b[..]), Err(GgufError::Corrupt(_))));
    }
}
