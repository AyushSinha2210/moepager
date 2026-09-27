//! moepager command-line tools.

use anyhow::Result;
use clap::{Parser, Subcommand};

mod cmd_gguf;
mod cmd_sim;
mod cmd_trace;
mod common;

#[derive(Parser)]
#[command(
    name = "moepager",
    version,
    about = "Expert-aware page-cache tooling for MoE GGUF models"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse a GGUF header and write the (layer, expert) → byte/page map.
    GgufMap(cmd_gguf::Args),
    /// Generate a synthetic MoE expert-access trace.
    Synth(cmd_trace::SynthArgs),
    /// Analyze an expert trace: reuse distance, LRU curve, routing stats, timing.
    Analyze(cmd_trace::AnalyzeArgs),
    /// Simulate page-cache policies (incl. Belady oracle) over an expert trace.
    Sim(cmd_sim::Args),
    /// Print a trace as CSV.
    TraceCsv(cmd_trace::CsvArgs),
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::GgufMap(a) => cmd_gguf::run(a),
        Cmd::Synth(a) => cmd_trace::synth(a),
        Cmd::Analyze(a) => cmd_trace::analyze(a),
        Cmd::TraceCsv(a) => cmd_trace::csv(a),
        Cmd::Sim(a) => cmd_sim::run(a),
    }
}
