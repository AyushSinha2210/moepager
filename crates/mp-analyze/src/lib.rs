//! Offline trace analysis: reuse distance, exact LRU miss-ratio curve,
//! routing statistics and timing.

pub mod reuse;

pub use reuse::{access_sequence, log2_histogram, lru_mrc, reuse_distances, MrcPoint, Reuse};
