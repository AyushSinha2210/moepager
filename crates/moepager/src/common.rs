//! Helpers shared by subcommands.

use std::path::Path;

use anyhow::{bail, Context, Result};
use mp_gguf::ExpertMap;

pub fn load_map(path: &Path) -> Result<ExpertMap> {
    let mut m: ExpertMap = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
    )
    .with_context(|| format!("parsing map {}", path.display()))?;
    m.reindex();
    Ok(m)
}

/// Dense unit sizes (`layer * n_experts + expert`) for a trace shape, taken
/// from a map, or uniform.
pub fn unit_sizes(
    map: Option<&ExpertMap>,
    uniform: u64,
    n_layers: u32,
    n_experts: u32,
) -> Result<Vec<u64>> {
    let n = (n_layers * n_experts) as usize;
    let Some(m) = map else {
        return Ok(vec![uniform; n]);
    };
    if m.n_experts != n_experts || m.n_layers < n_layers {
        bail!(
            "map shape ({} layers x {} experts) does not fit trace ({} x {})",
            m.n_layers,
            m.n_experts,
            n_layers,
            n_experts
        );
    }
    let mut v = vec![0u64; n];
    for l in 0..n_layers {
        for e in 0..n_experts {
            let id = m
                .unit_id(l, e)
                .with_context(|| format!("map has no unit for layer {l} expert {e}"))?;
            v[(l * n_experts + e) as usize] = m.units[id as usize].bytes;
        }
    }
    Ok(v)
}

pub fn gb(b: f64) -> String {
    format!("{:.3}", b / 1e9)
}
