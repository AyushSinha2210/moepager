//! Trace-driven simulator of the page cache at expert-unit granularity.

pub mod evict;

pub use evict::{Belady, Evictor, Lfu, Lru};
