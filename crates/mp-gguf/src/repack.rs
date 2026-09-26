//! Detect weights that llama.cpp's CPU backend will *repack* into anonymous
//! memory instead of serving from the mmap'd file.
//!
//! Source: ggml/src/ggml-cpu/repack.cpp, `ggml_repack_get_optimal_repack_type`
//! (read 2026-10). Repacked tensors include 3-D MoE expert tensors
//! (`supports_op` accepts `MUL_MAT_ID`). Once repacked, the file pages are
//! only touched at load time, so page-cache management cannot help them;
//! run llama.cpp with `--no-repack` (`-nr`). These rules track upstream and
//! may drift; treat the result as a warning, not ground truth.

use crate::map::ExpertMap;
use crate::parse::GgufHeader;
use crate::types::GgmlType;

/// CPU features relevant to ggml's repack selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    pub x86_avx2: bool,
    pub x86_avx512f: bool,
    pub arm_dotprod: bool,
    pub arm_i8mm: bool,
}

impl CpuFeatures {
    /// Parse the text of /proc/cpuinfo (x86 "flags" or arm64 "Features").
    pub fn from_cpuinfo(text: &str) -> Self {
        let mut f = CpuFeatures::default();
        for line in text.lines() {
            let Some((key, val)) = line.split_once(':') else {
                continue;
            };
            let key = key.trim();
            if key != "flags" && key != "Features" {
                continue;
            }
            for w in val.split_whitespace() {
                match w {
                    "avx2" => f.x86_avx2 = true,
                    "avx512f" => f.x86_avx512f = true,
                    "asimddp" => f.arm_dotprod = true,
                    "i8mm" => f.arm_i8mm = true,
                    _ => {}
                }
            }
            break;
        }
        f
    }

    /// Row-count divisor ggml requires to repack `ty` on this CPU, if any.
    pub fn repack_divisor(&self, ty: GgmlType) -> Option<u64> {
        use GgmlType as T;
        let x86_avx2 = [T::Q4_0, T::Q4_K, T::IQ4_NL, T::MXFP4];
        if self.x86_avx2 && x86_avx2.contains(&ty) {
            return Some(8);
        }
        if self.x86_avx512f && ty == T::Q2_K {
            return Some(8);
        }
        if self.arm_dotprod || self.arm_i8mm {
            if [T::Q4_0, T::IQ4_NL, T::Q8_0, T::MXFP4].contains(&ty) {
                return Some(4);
            }
            if [T::Q4_K, T::Q5_K, T::Q6_K].contains(&ty) {
                return Some(8);
            }
        }
        None
    }
}

/// Summary of repack exposure for a model on a CPU.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RepackReport {
    /// Expert tensors that would be repacked (name, type).
    pub expert_tensors: Vec<(String, String)>,
    pub expert_bytes_repacked: u64,
    pub expert_bytes_total: u64,
}

impl RepackReport {
    pub fn fraction(&self) -> f64 {
        if self.expert_bytes_total == 0 {
            0.0
        } else {
            self.expert_bytes_repacked as f64 / self.expert_bytes_total as f64
        }
    }
}

/// Which expert tensors default llama.cpp would repack on `cpu`.
pub fn repack_report(h: &GgufHeader, map: &ExpertMap, cpu: &CpuFeatures) -> RepackReport {
    let mut r = RepackReport {
        expert_bytes_total: map.expert_bytes(),
        ..Default::default()
    };
    let expert_tensors: std::collections::BTreeSet<&str> = map
        .units
        .iter()
        .flat_map(|u| u.slices.iter().map(|s| s.tensor.as_str()))
        .collect();
    for (i, t) in h.tensors.iter().enumerate() {
        if !expert_tensors.contains(t.name.as_str()) || t.dims.len() != 3 {
            continue;
        }
        if let Some(div) = cpu.repack_divisor(t.ty) {
            if t.dims[1] % div == 0 {
                r.expert_tensors.push((t.name.clone(), t.ty.name()));
                r.expert_bytes_repacked += h.tensor_nbytes(i).unwrap_or(0);
            }
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cpuinfo() {
        let x86 = "processor\t: 0\nflags\t\t: fpu sse2 avx avx2 fma\n";
        assert_eq!(
            CpuFeatures::from_cpuinfo(x86),
            CpuFeatures {
                x86_avx2: true,
                ..Default::default()
            }
        );
        let arm = "Features\t: fp asimd asimddp i8mm\n";
        let f = CpuFeatures::from_cpuinfo(arm);
        assert!(f.arm_dotprod && f.arm_i8mm && !f.x86_avx2);
    }

    #[test]
    fn avx2_repacks_q4k_and_mxfp4_not_q6k() {
        let f = CpuFeatures {
            x86_avx2: true,
            ..Default::default()
        };
        assert_eq!(f.repack_divisor(GgmlType::Q4_K), Some(8));
        assert_eq!(f.repack_divisor(GgmlType::MXFP4), Some(8));
        assert_eq!(f.repack_divisor(GgmlType::Q6_K), None);
        assert_eq!(f.repack_divisor(GgmlType::Q8_0), None);
        assert_eq!(CpuFeatures::default().repack_divisor(GgmlType::Q4_K), None);
    }
}
