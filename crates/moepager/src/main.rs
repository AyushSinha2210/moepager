//! moepager command-line tools.

use anyhow::Result;
use clap::{Parser, Subcommand};

mod cmd_gguf;

#[derive(Parser)]
#[command(name = "moepager", version, about = "Expert-aware page-cache tooling for MoE GGUF models")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse a GGUF header and write the (layer, expert) → byte/page map.
    GgufMap(cmd_gguf::Args),
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::GgufMap(a) => cmd_gguf::run(a),
    }
}
