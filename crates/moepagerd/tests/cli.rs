//! Daemon dry run through the binary: trace source + mock OS layer.

use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn dry_run_on_synthetic_trace() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let gguf = root.join("fixtures/generated/tiny_moe.gguf");
    if !gguf.exists() {
        assert!(Command::new("python3")
            .arg(root.join("python/moepager_tools/gguf_fixtures.py"))
            .arg(root.join("fixtures/generated"))
            .status()
            .unwrap()
            .success());
    }
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("daemon");
    std::fs::create_dir_all(&dir).unwrap();
    // Build a trace for the fixture's shape (2 layers × 4 experts, top-2).
    let p = mp_synth::SynthParams {
        n_layers: 2,
        n_experts: 4,
        top_k: 2,
        tokens: 40,
        ..Default::default()
    };
    let (h, ev) = mp_synth::generate(&p);
    let trace = dir.join("t.mpt");
    mp_trace::write_expert_file(&trace, &h, &ev).unwrap();
    let stats = dir.join("stats.json");
    let out = Command::new(env!("CARGO_BIN_EXE_moepagerd"))
        .arg(&gguf)
        .args(["--ops", "mock", "--policy", "v", "--budget-mb", "1"])
        .arg(format!("--source=trace:{}", trace.display()))
        .arg("--stats-json")
        .arg(&stats)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let c: serde_json::Value = serde_json::from_slice(&std::fs::read(&stats).unwrap()).unwrap();
    assert_eq!(c["events"], 40 * 2 * 2);
    assert_eq!(c["tokens"], 39);
    assert_eq!(
        c["prefetch_ops"], 160,
        "completion readahead on every observed miss"
    );
    assert_eq!(c["failed_ops"], 0);
}
