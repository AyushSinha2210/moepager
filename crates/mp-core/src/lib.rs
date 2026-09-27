//! Pure, deterministic policy core shared by the simulator and the daemon.
//!
//! Nothing in this crate performs I/O or reads clocks; time is passed in.

pub mod cost;
pub mod prefetch;
pub mod residency;
pub mod rng;
pub mod stats;

pub use cost::CostModel;
pub use prefetch::{PrefetchConfig, PrefetchPlanner};
pub use residency::{Action, ResidencyConfig, ResidencyEngine};
pub use rng::{Rng, Zipf};
pub use stats::{OnlineStats, Seen};
