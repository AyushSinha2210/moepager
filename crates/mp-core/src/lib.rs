//! Pure, deterministic policy core shared by the simulator and the daemon.
//!
//! Nothing in this crate performs I/O or reads clocks; time is passed in.

pub mod rng;

pub use rng::{Rng, Zipf};
