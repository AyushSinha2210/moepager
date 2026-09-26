//! ggml tensor type table: block size and bytes per block.
//!
//! Ids follow `enum ggml_type` in ggml.h. Removed ids (4, 5, 31–33, 36–38)
//! are reported as unknown. Unknown types are not fatal for mapping: the
//! map falls back to sizes derived from tensor offsets.

/// A ggml tensor element type as stored in GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GgmlType(pub u32);

struct Info {
    id: u32,
    name: &'static str,
    block: u64,
    size: u64,
}

const TABLE: &[Info] = &[
    Info {
        id: 0,
        name: "F32",
        block: 1,
        size: 4,
    },
    Info {
        id: 1,
        name: "F16",
        block: 1,
        size: 2,
    },
    Info {
        id: 2,
        name: "Q4_0",
        block: 32,
        size: 18,
    },
    Info {
        id: 3,
        name: "Q4_1",
        block: 32,
        size: 20,
    },
    Info {
        id: 6,
        name: "Q5_0",
        block: 32,
        size: 22,
    },
    Info {
        id: 7,
        name: "Q5_1",
        block: 32,
        size: 24,
    },
    Info {
        id: 8,
        name: "Q8_0",
        block: 32,
        size: 34,
    },
    Info {
        id: 9,
        name: "Q8_1",
        block: 32,
        size: 36,
    },
    Info {
        id: 10,
        name: "Q2_K",
        block: 256,
        size: 84,
    },
    Info {
        id: 11,
        name: "Q3_K",
        block: 256,
        size: 110,
    },
    Info {
        id: 12,
        name: "Q4_K",
        block: 256,
        size: 144,
    },
    Info {
        id: 13,
        name: "Q5_K",
        block: 256,
        size: 176,
    },
    Info {
        id: 14,
        name: "Q6_K",
        block: 256,
        size: 210,
    },
    Info {
        id: 15,
        name: "Q8_K",
        block: 256,
        size: 292,
    },
    Info {
        id: 16,
        name: "IQ2_XXS",
        block: 256,
        size: 66,
    },
    Info {
        id: 17,
        name: "IQ2_XS",
        block: 256,
        size: 74,
    },
    Info {
        id: 18,
        name: "IQ3_XXS",
        block: 256,
        size: 98,
    },
    Info {
        id: 19,
        name: "IQ1_S",
        block: 256,
        size: 50,
    },
    Info {
        id: 20,
        name: "IQ4_NL",
        block: 32,
        size: 18,
    },
    Info {
        id: 21,
        name: "IQ3_S",
        block: 256,
        size: 110,
    },
    Info {
        id: 22,
        name: "IQ2_S",
        block: 256,
        size: 82,
    },
    Info {
        id: 23,
        name: "IQ4_XS",
        block: 256,
        size: 136,
    },
    Info {
        id: 24,
        name: "I8",
        block: 1,
        size: 1,
    },
    Info {
        id: 25,
        name: "I16",
        block: 1,
        size: 2,
    },
    Info {
        id: 26,
        name: "I32",
        block: 1,
        size: 4,
    },
    Info {
        id: 27,
        name: "I64",
        block: 1,
        size: 8,
    },
    Info {
        id: 28,
        name: "F64",
        block: 1,
        size: 8,
    },
    Info {
        id: 29,
        name: "IQ1_M",
        block: 256,
        size: 56,
    },
    Info {
        id: 30,
        name: "BF16",
        block: 1,
        size: 2,
    },
    Info {
        id: 34,
        name: "TQ1_0",
        block: 256,
        size: 54,
    },
    Info {
        id: 35,
        name: "TQ2_0",
        block: 256,
        size: 66,
    },
    Info {
        id: 39,
        name: "MXFP4",
        block: 32,
        size: 17,
    },
];

impl GgmlType {
    pub const F32: GgmlType = GgmlType(0);
    pub const F16: GgmlType = GgmlType(1);
    pub const Q4_0: GgmlType = GgmlType(2);
    pub const Q8_0: GgmlType = GgmlType(8);
    pub const Q2_K: GgmlType = GgmlType(10);
    pub const Q4_K: GgmlType = GgmlType(12);
    pub const Q5_K: GgmlType = GgmlType(13);
    pub const Q6_K: GgmlType = GgmlType(14);
    pub const IQ4_NL: GgmlType = GgmlType(20);
    pub const MXFP4: GgmlType = GgmlType(39);

    fn info(self) -> Option<&'static Info> {
        TABLE.iter().find(|i| i.id == self.0)
    }

    /// Human-readable name, or `TYPE<id>` for unknown ids.
    pub fn name(self) -> String {
        self.info()
            .map_or_else(|| format!("TYPE{}", self.0), |i| i.name.to_string())
    }

    /// (elements per block, bytes per block), if the type is known.
    pub fn block(self) -> Option<(u64, u64)> {
        self.info().map(|i| (i.block, i.size))
    }

    /// Bytes for a tensor of shape `dims` (ne[0] fastest). `None` if the type
    /// is unknown or `dims[0]` is not a multiple of the block size.
    pub fn nbytes(self, dims: &[u64]) -> Option<u64> {
        let (bs, ts) = self.block()?;
        let first = *dims.first()?;
        if first % bs != 0 {
            return None;
        }
        let n: u64 = dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d))?;
        (n / bs).checked_mul(ts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_sizes_match_real_qwen3_header() {
        // From the Qwen3-30B-A3B Q4_K_M header: ffn_gate_exps [2048, 768, 128] Q4_K.
        assert_eq!(GgmlType::Q4_K.nbytes(&[2048, 768, 128]), Some(113_246_208));
        // ffn_down_exps [768, 2048, 128] Q6_K.
        assert_eq!(GgmlType::Q6_K.nbytes(&[768, 2048, 128]), Some(165_150_720));
        // gpt-oss-20b: ffn_up_exps [2880, 2880, 32] MXFP4.
        assert_eq!(GgmlType::MXFP4.nbytes(&[2880, 2880, 32]), Some(141_004_800));
    }

    #[test]
    fn unknown_and_misaligned() {
        assert_eq!(GgmlType(4).block(), None);
        assert_eq!(GgmlType(4).name(), "TYPE4");
        assert_eq!(GgmlType::Q4_K.nbytes(&[100, 2]), None);
        assert_eq!(GgmlType::F32.nbytes(&[]), None);
    }
}
