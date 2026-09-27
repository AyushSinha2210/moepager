//! Trace-driven simulator of the page cache at expert-unit granularity.

pub mod control;
pub mod engine;
pub mod evict;

pub use engine::{simulate, Controller, Metrics, NoControl, SimConfig, View};
pub use evict::{Belady, Evictor, Lfu, Lru};
