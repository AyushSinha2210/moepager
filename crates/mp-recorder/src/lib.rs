//! Black-box recorders (page-cache observation without touching the
//! engine), converters, and a trace replayer for real-kernel experiments.

pub mod bpftrace;
pub mod convert;
pub mod replay;
pub mod scan;
pub mod sentinel;

pub use convert::{page_to_expert, ConvertConfig};
pub use replay::{replay, ReplayConfig, ReplayStats, TouchMode};
pub use scan::ScanRecorder;
pub use sentinel::SentinelRecorder;
