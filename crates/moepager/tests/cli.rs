//! Smoke test of the CLI pipeline: gguf-map → synth → analyze → sim → trace-csv.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_moepager")
}

fn tmp(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cli");
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

fn run(args: &[&str]) -> String {
    let out = Command::new(bin())
        .args(args)
        .output()
        .expect("run moepager");
    assert!(
        out.status.success(),
        "moepager {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn fixture() -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let p = root.join("fixtures/generated/tiny_moe.gguf");
    if !p.exists() {
        let ok = Command::new("python3")
            .arg(root.join("python/moepager_tools/gguf_fixtures.py"))
            .arg(root.join("fixtures/generated"))
            .status()
            .unwrap()
            .success();
        assert!(ok);
    }
    p
}

#[test]
fn pipeline_runs_end_to_end() {
    let map = tmp("tiny.map.json");
    let trace = tmp("tiny.mpt");
    let json = tmp("tiny.analyze.json");
    let csv = tmp("tiny.sim.csv");
    let out = run(&[
        "gguf-map",
        fixture().to_str().unwrap(),
        "-o",
        map.to_str().unwrap(),
    ]);
    assert!(
        out.contains("experts=4") && out.contains("units=8"),
        "{out}"
    );
    run(&[
        "synth",
        "--like",
        map.to_str().unwrap(),
        "--tokens",
        "50",
        "-o",
        trace.to_str().unwrap(),
    ]);
    let out = run(&[
        "analyze",
        trace.to_str().unwrap(),
        "--map",
        map.to_str().unwrap(),
        "--json",
        json.to_str().unwrap(),
    ]);
    assert!(out.contains("tokens=50"), "{out}");
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
    assert_eq!(report["accesses"], 50 * 2 * 2);
    let out = run(&[
        "sim",
        trace.to_str().unwrap(),
        "--map",
        map.to_str().unwrap(),
        "--caps",
        "0.5",
        "--policies",
        "lru,belady,v-miss",
        "--warmup",
        "5",
        "--csv",
        csv.to_str().unwrap(),
    ]);
    assert!(
        out.contains("| lru |") && out.contains("belady*") && out.contains("v-miss"),
        "{out}"
    );
    assert_eq!(
        std::fs::read_to_string(&csv).unwrap().lines().count(),
        1 + 2 * 3
    );
    let out = run(&["trace-csv", trace.to_str().unwrap()]);
    assert_eq!(out.lines().count(), 1 + 200);
}

#[test]
fn bad_inputs_fail_cleanly() {
    let out = Command::new(bin())
        .args(["sim", "/nonexistent.mpt"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let out = Command::new(bin())
        .args(["synth", "-o", "/dev/null", "--top-k", "0"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("top_k"));
}
