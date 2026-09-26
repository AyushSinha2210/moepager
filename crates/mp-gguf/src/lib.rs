//! GGUF header parser and MoE expert map.
//!
//! Only the header is read; tensor data is never touched.

pub mod parse;
pub mod types;

pub use parse::{parse_file, parse_header, GgufError, GgufHeader, MetaValue, TensorInfo};
pub use types::GgmlType;
