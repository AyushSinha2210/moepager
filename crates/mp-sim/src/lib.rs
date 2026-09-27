//! Trace-driven simulator of the page cache at expert-unit granularity.

pub mod control;
pub mod engine;
pub mod evict;
pub mod policy;

pub use engine::{simulate, Controller, Metrics, NoControl, SimConfig, View};
pub use evict::{Belady, Evictor, Lfu, Lru};
pub use policy::{csv, markdown, run_policy, sweep, PolicyOptions, RunResult, POLICIES};
