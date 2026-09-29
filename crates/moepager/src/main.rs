//! moepager command-line tools.

use anyhow::Result;
use clap::{Parser, Subcommand};

mod cmd_faultio;
mod cmd_gguf;
mod cmd_record;
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
    /// Record page-cache activity on a model file (black box, no root).
    Record(cmd_record::RecordArgs),
    /// Convert `bpf/moepager.bt` output into a page trace.
    IngestBpftrace(cmd_record::IngestArgs),
    /// Convert a page trace into a (miss-only) expert trace.
    Page2expert(cmd_record::Page2ExpertArgs),
    /// Replay an expert trace against the real model file (Experiment B).
    Replay(cmd_record::ReplayArgs),
    /// Microbenchmark: fault-driven vs bulk read bandwidth for expert units (Experiment C).
    FaultIo(cmd_faultio::Args),
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
        Cmd::Record(a) => cmd_record::record(a),
        Cmd::IngestBpftrace(a) => cmd_record::ingest(a),
        Cmd::Page2expert(a) => cmd_record::page2expert(a),
        Cmd::Replay(a) => cmd_record::replay(a),
        Cmd::FaultIo(a) => cmd_faultio::run(a),
    }
}
