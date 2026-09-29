//! Black-box recorders (page-cache observation without touching the
//! engine), converters, and a trace replayer for real-kernel experiments.

pub mod bpftrace;
pub mod convert;
pub mod scan;
pub mod sentinel;
