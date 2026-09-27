//! Pure, deterministic policy core shared by the simulator and the daemon.
//!
//! Nothing in this crate performs I/O or reads clocks; time is passed in.

pub mod cost;
pub mod rng;

pub use cost::CostModel;
pub use rng::{Rng, Zipf};
