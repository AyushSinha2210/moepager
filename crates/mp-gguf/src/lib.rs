//! GGUF header parser and MoE expert map.
//!
//! Only the header is read; tensor data is never touched.

pub mod types;

pub use types::GgmlType;
