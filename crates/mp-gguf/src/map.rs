//! Expert map: GGUF tensors → (layer, expert) units → byte slices → pages.
//!
//! A *unit* is everything one expert at one layer reads during decode: its
//! slice of each merged expert tensor (`ffn_up_exps`, `ffn_gate_exps`,
//! `ffn_down_exps`, or fused `ffn_gate_up_exps`) plus its row of any 2-D
//! per-expert tensor (`*_exps.bias`). Slices are 32-byte aligned, not page
//! aligned, so a boundary page can belong to two units.

use serde::{Deserialize, Serialize};

use crate::parse::GgufHeader;

/// Keep sentinel pages this far from slice edges so that a neighbour's
/// fault read-around (128 KiB default) does not bring them in.
pub const SENTINEL_GUARD: u64 = 128 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slice {
    /// e.g. "up", "gate", "down", "gate_up", "down_bias".
    pub kind: String,
    pub tensor: String,
    /// Absolute file offset.
    pub offset: u64,
    pub len: u64,
}

impl Slice {
    pub fn end(&self) -> u64 {
        self.offset + self.len
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unit {
    pub layer: u32,
    pub expert: u32,
    pub bytes: u64,
    pub slices: Vec<Slice>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenseRegion {
    pub name: String,
    pub layer: Option<u32>,
    pub offset: u64,
    pub len: u64,
}

/// What a byte range of the file belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Owner {
    Expert { unit: u32, slice: u16 },
    Dense { region: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertMap {
    pub arch: String,
    pub page_size: u64,
    pub data_offset: u64,
    pub file_size: Option<u64>,
    pub n_layers: u32,
    pub n_experts: u32,
    pub top_k: Option<u32>,
    /// Sorted by (layer, expert). Only layers with expert tensors appear.
    pub units: Vec<Unit>,
    pub dense: Vec<DenseRegion>,
    /// Non-fatal oddities found while mapping.
    pub warnings: Vec<String>,
    #[serde(skip)]
    layer_base: Vec<Option<u32>>,
}

/// `blk.<L>.<base>_exps.<suffix>` → (L, kind).
fn expert_name(name: &str) -> Option<(u32, String)> {
    let mut it = name.splitn(4, '.');
    if it.next()? != "blk" {
        return None;
    }
    let layer: u32 = it.next()?.parse().ok()?;
    let base = it.next()?.strip_suffix("_exps")?;
    let suffix = it.next()?;
    let base = base.strip_prefix("ffn_").unwrap_or(base);
    let kind = if suffix == "weight" {
        base.to_string()
    } else {
        format!("{base}_{suffix}")
    };
    Some((layer, kind))
}

fn layer_of(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

impl ExpertMap {
    /// Build the map from a parsed header. `file_size` (if known) is used
    /// only for validation.
    pub fn from_header(h: &GgufHeader, page_size: u64, file_size: Option<u64>) -> Self {
        assert!(page_size.is_power_of_two());
        let mut warnings = Vec::new();
        let n_layers = h.arch_u64("block_count").unwrap_or(0) as u32;
        let top_k = h.arch_u64("expert_used_count").map(|v| v as u32);
        let mut n_experts = h.arch_u64("expert_count").unwrap_or(0) as u32;
        if n_experts == 0 {
            // Fall back to the outer dimension of the first 3-D expert tensor.
            n_experts = h
                .tensors
                .iter()
                .find(|t| expert_name(&t.name).is_some() && t.dims.len() == 3)
                .map_or(0, |t| t.dims[2] as u32);
        }

        let mut slices: Vec<(u32, u32, Slice)> = Vec::new();
        let mut dense = Vec::new();
        for (i, t) in h.tensors.iter().enumerate() {
            let off = h.abs_offset(i);
            let Some(nb) = h.tensor_nbytes(i) else {
                warnings.push(format!(
                    "{}: cannot size tensor (type {})",
                    t.name,
                    t.ty.name()
                ));
                continue;
            };
            if t.ty.block().is_none() {
                warnings.push(format!(
                    "{}: unknown type {}, sized from offsets",
                    t.name,
                    t.ty.name()
                ));
            }
            let exp_dim = match t.dims.len() {
                3 => Some(t.dims[2]),
                2 => Some(t.dims[1]),
                _ => None,
            };
            match expert_name(&t.name) {
                Some((layer, kind)) if n_experts > 0 && exp_dim == Some(n_experts as u64) => {
                    if nb % n_experts as u64 != 0 {
                        warnings.push(format!("{}: {nb} bytes not divisible by experts", t.name));
                    }
                    let per = nb / n_experts as u64;
                    for e in 0..n_experts {
                        slices.push((
                            layer,
                            e,
                            Slice {
                                kind: kind.clone(),
                                tensor: t.name.clone(),
                                offset: off + e as u64 * per,
                                len: per,
                            },
                        ));
                    }
                }
                Some(_) => {
                    warnings.push(format!(
                        "{}: expert-named tensor with shape {:?}",
                        t.name, t.dims
                    ));
                    dense.push(DenseRegion {
                        name: t.name.clone(),
                        layer: layer_of(&t.name),
                        offset: off,
                        len: nb,
                    });
                }
                None => dense.push(DenseRegion {
                    name: t.name.clone(),
                    layer: layer_of(&t.name),
                    offset: off,
                    len: nb,
                }),
            }
        }

        slices.sort_by_key(|(l, e, s)| (*l, *e, s.offset));
        let mut units: Vec<Unit> = Vec::new();
        for (layer, expert, s) in slices {
            match units.last_mut() {
                Some(u) if u.layer == layer && u.expert == expert => {
                    u.bytes += s.len;
                    u.slices.push(s);
                }
                _ => units.push(Unit {
                    layer,
                    expert,
                    bytes: s.len,
                    slices: vec![s],
                }),
            }
        }
        dense.sort_by_key(|d| d.offset);

        if let Some(fs) = file_size {
            let end = units
                .iter()
                .flat_map(|u| u.slices.iter().map(Slice::end))
                .chain(dense.iter().map(|d| d.offset + d.len))
                .max()
                .unwrap_or(0);
            if end > fs {
                warnings.push(format!("tensor data ends at {end} beyond file size {fs}"));
            }
        }
        let n_layers = n_layers.max(units.last().map_or(0, |u| u.layer + 1));
        let mut m = ExpertMap {
            arch: h.arch().unwrap_or("unknown").to_string(),
            page_size,
            data_offset: h.data_offset,
            file_size,
            n_layers,
            n_experts,
            top_k,
            units,
            dense,
            warnings,
            layer_base: Vec::new(),
        };
        m.reindex();
        m
    }

    /// Rebuild the (layer, expert) → unit id index (needed after deserialising).
    pub fn reindex(&mut self) {
        self.layer_base = vec![None; self.n_layers as usize];
        for (i, u) in self.units.iter().enumerate() {
            if u.expert == 0 {
                if let Some(slot) = self.layer_base.get_mut(u.layer as usize) {
                    *slot = Some(i as u32);
                }
            }
        }
    }

    pub fn unit_id(&self, layer: u32, expert: u32) -> Option<u32> {
        if expert >= self.n_experts {
            return None;
        }
        let id = (*self.layer_base.get(layer as usize)?)? + expert;
        let u = self.units.get(id as usize)?;
        (u.layer == layer && u.expert == expert).then_some(id)
    }

    pub fn moe_layers(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.units.iter().map(|u| u.layer).collect();
        v.dedup();
        v
    }

    pub fn expert_bytes(&self) -> u64 {
        self.units.iter().map(|u| u.bytes).sum()
    }

    pub fn dense_bytes(&self) -> u64 {
        self.dense.iter().map(|d| d.len).sum()
    }

    /// Unit sizes indexed by unit id.
    pub fn unit_bytes(&self) -> Vec<u64> {
        self.units.iter().map(|u| u.bytes).collect()
    }

    /// Inclusive page range covered by a byte range.
    pub fn pages_of(&self, offset: u64, len: u64) -> (u64, u64) {
        let p = self.page_size;
        (offset / p, (offset + len.max(1) - 1) / p)
    }

    /// Inclusive page ranges of each slice of a unit.
    pub fn unit_pages(&self, id: u32) -> Vec<(u64, u64)> {
        self.units[id as usize]
            .slices
            .iter()
            .map(|s| self.pages_of(s.offset, s.len))
            .collect()
    }

    /// Number of distinct pages a unit touches (shared boundary pages count once per unit).
    pub fn unit_page_count(&self, id: u32) -> u64 {
        let mut r = self.unit_pages(id);
        r.sort();
        let mut n = 0;
        let mut last: Option<u64> = None;
        for (a, b) in r {
            let a = last.map_or(a, |l| a.max(l + 1));
            if b >= a {
                n += b - a + 1;
            }
            last = Some(last.map_or(b, |l| l.max(b)));
        }
        n
    }

    /// Sentinel page for a slice: the middle page, kept `SENTINEL_GUARD`
    /// away from both edges when the slice is large enough. Returns
    /// `(page, reliable)`.
    pub fn sentinel(&self, s: &Slice) -> (u64, bool) {
        let mid = s.offset + s.len / 2;
        let reliable = s.len >= 2 * SENTINEL_GUARD + 2 * self.page_size;
        (mid / self.page_size, reliable)
    }

    /// Sentinel pages of all slices of a unit (weights only; bias rows are
    /// too small to be reliable and live in other tensors' pages).
    pub fn unit_sentinels(&self, id: u32) -> Vec<(u64, bool)> {
        self.units[id as usize]
            .slices
            .iter()
            .filter(|s| !s.kind.ends_with("_bias"))
            .map(|s| self.sentinel(s))
            .collect()
    }

    /// Build a reverse page → owner index.
    pub fn page_index(&self) -> PageIndex {
        let mut iv = Vec::new();
        for (ui, u) in self.units.iter().enumerate() {
            for (si, s) in u.slices.iter().enumerate() {
                iv.push((
                    s.offset,
                    s.end(),
                    Owner::Expert {
                        unit: ui as u32,
                        slice: si as u16,
                    },
                ));
            }
        }
        for (di, d) in self.dense.iter().enumerate() {
            iv.push((
                d.offset,
                d.offset + d.len,
                Owner::Dense { region: di as u32 },
            ));
        }
        iv.sort_by_key(|x| (x.0, x.1));
        let mut max_end = Vec::with_capacity(iv.len());
        let mut m = 0;
        for x in &iv {
            m = m.max(x.1);
            max_end.push(m);
        }
        PageIndex {
            page_size: self.page_size,
            iv,
            max_end,
        }
    }

    /// Short human-readable summary.
    pub fn summary(&self) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        let n = self.units.len().max(1) as u64;
        let (mn, mx) = self
            .units
            .iter()
            .fold((u64::MAX, 0), |(a, b), u| (a.min(u.bytes), b.max(u.bytes)));
        format!(
            "arch={} layers={} moe_layers={} experts={} top_k={} units={} \
             expert_bytes={:.3}GB dense_bytes={:.3}GB unit_bytes[min/avg/max]={}/{}/{} \
             warnings={}",
            self.arch,
            self.n_layers,
            self.moe_layers().len(),
            self.n_experts,
            self.top_k.map_or("?".into(), |k| k.to_string()),
            self.units.len(),
            gb(self.expert_bytes()),
            gb(self.dense_bytes()),
            if self.units.is_empty() { 0 } else { mn },
            self.expert_bytes() / n,
            mx,
            self.warnings.len()
        )
    }
}

/// Reverse lookup from file pages to owners (expert slices or dense tensors).
pub struct PageIndex {
    page_size: u64,
    iv: Vec<(u64, u64, Owner)>,
    /// Prefix maximum of interval ends, for overlap queries.
    max_end: Vec<u64>,
}

impl PageIndex {
    /// All owners whose byte range intersects the page.
    pub fn owners(&self, page: u64) -> Vec<Owner> {
        let lo = page * self.page_size;
        let hi = lo + self.page_size;
        // Intervals starting before `hi`.
        let n = self.iv.partition_point(|x| x.0 < hi);
        // First index whose prefix max end exceeds `lo`; nothing before it can overlap.
        let first = self.max_end[..n].partition_point(|&e| e <= lo);
        self.iv[first..n]
            .iter()
            .filter(|x| x.1 > lo)
            .map(|x| x.2)
            .collect()
    }

    /// Expert units touching the page (deduplicated).
    pub fn units(&self, page: u64) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .owners(page)
            .into_iter()
            .filter_map(|o| match o {
                Owner::Expert { unit, .. } => Some(unit),
                Owner::Dense { .. } => None,
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_names() {
        assert_eq!(
            expert_name("blk.3.ffn_up_exps.weight"),
            Some((3, "up".into()))
        );
        assert_eq!(
            expert_name("blk.0.ffn_gate_up_exps.weight"),
            Some((0, "gate_up".into()))
        );
        assert_eq!(
            expert_name("blk.12.ffn_down_exps.bias"),
            Some((12, "down_bias".into()))
        );
        assert_eq!(expert_name("blk.1.ffn_up_shexp.weight"), None);
        assert_eq!(expert_name("blk.1.ffn_gate_inp.weight"), None);
        assert_eq!(expert_name("output.weight"), None);
        assert_eq!(layer_of("blk.7.attn_q.weight"), Some(7));
        assert_eq!(layer_of("token_embd.weight"), None);
    }
}
