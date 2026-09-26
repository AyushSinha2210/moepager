//! Parser + map tests against fixtures written by the independent Python
//! writer (python/moepager_tools/gguf_fixtures.py).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

use mp_gguf::map::{ExpertMap, Owner};
use mp_gguf::parse_file;
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixtures() -> PathBuf {
    static GEN: Once = Once::new();
    let dir = root().join("fixtures/generated");
    GEN.call_once(|| {
        if !dir.join("big_moe.expected.json").exists() {
            let ok = Command::new("python3")
                .arg(root().join("python/moepager_tools/gguf_fixtures.py"))
                .arg(&dir)
                .arg("--big")
                .status()
                .expect("python3 needed to generate fixtures")
                .success();
            assert!(ok, "fixture generation failed");
        }
    });
    dir
}

fn load(name: &str) -> (ExpertMap, Value) {
    let p = fixtures().join(name);
    let h = parse_file(&p).unwrap();
    let size = std::fs::metadata(&p).unwrap().len();
    let exp: Value =
        serde_json::from_str(&std::fs::read_to_string(p.with_extension("expected.json")).unwrap())
            .unwrap();
    assert_eq!(
        h.data_offset,
        exp["data_offset"].as_u64().unwrap(),
        "{name} data offset"
    );
    assert_eq!(size, exp["file_size"].as_u64().unwrap());
    (ExpertMap::from_header(&h, 4096, Some(size)), exp)
}

fn check_against_expected(name: &str) -> ExpertMap {
    let (m, exp) = load(name);
    assert!(m.warnings.is_empty(), "{name}: {:?}", m.warnings);
    assert_eq!(m.n_layers as u64, exp["n_layers"].as_u64().unwrap());
    assert_eq!(m.n_experts as u64, exp["n_experts"].as_u64().unwrap());
    assert_eq!(m.top_k.map(|k| k as u64), exp["top_k"].as_u64());
    let units = exp["units"].as_array().unwrap();
    assert_eq!(m.units.len(), units.len());
    for (u, e) in m.units.iter().zip(units) {
        assert_eq!(u.layer as u64, e["layer"].as_u64().unwrap());
        assert_eq!(u.expert as u64, e["expert"].as_u64().unwrap());
        let es = e["slices"].as_array().unwrap();
        assert_eq!(
            u.slices.len(),
            es.len(),
            "{name} unit {}/{}",
            u.layer,
            u.expert
        );
        for (s, x) in u.slices.iter().zip(es) {
            assert_eq!(s.kind, x["kind"].as_str().unwrap());
            assert_eq!(s.offset, x["offset"].as_u64().unwrap());
            assert_eq!(s.len, x["len"].as_u64().unwrap());
        }
        assert_eq!(u.bytes, u.slices.iter().map(|s| s.len).sum::<u64>());
        assert_eq!(
            m.unit_id(u.layer, u.expert).map(|i| &m.units[i as usize]),
            Some(u)
        );
    }
    let dense = exp["dense"].as_array().unwrap();
    assert_eq!(m.dense.len(), dense.len());
    m
}

#[test]
fn tiny_moe_matches_expected() {
    let m = check_against_expected("tiny_moe.gguf");
    assert_eq!(m.arch, "testmoe");
    // Down projection alternates Q6_K / Q4_K across layers → unequal unit sizes.
    assert_ne!(m.units[0].bytes, m.units[m.n_experts as usize].bytes);
}

#[test]
fn bias_fixture_groups_bias_rows_into_units() {
    let m = check_against_expected("moe_bias_mxfp4.gguf");
    let kinds: Vec<&str> = m.units[0].slices.iter().map(|s| s.kind.as_str()).collect();
    assert_eq!(kinds.len(), 6);
    assert!(kinds.contains(&"down_bias") && !kinds.contains(&"gate_up"));
    // Sentinels skip bias rows.
    assert_eq!(m.unit_sentinels(0).len(), 3);
}

#[test]
fn v2_and_big_fixtures() {
    check_against_expected("v2_moe.gguf");
    let m = check_against_expected("big_moe.gguf");
    assert!(m.unit_sentinels(0).iter().all(|&(_, reliable)| reliable));
}

#[test]
fn dense_model_has_no_units() {
    let (m, _) = load("dense.gguf");
    assert!(m.units.is_empty());
    assert_eq!(m.n_experts, 0);
    assert_eq!(m.dense.len(), 2);
}

#[test]
fn page_index_is_consistent_with_slices() {
    let (m, _) = load("tiny_moe.gguf");
    let idx = m.page_index();
    let mut shared = 0;
    for id in 0..m.units.len() as u32 {
        for (a, b) in m.unit_pages(id) {
            for p in a..=b {
                let us = idx.units(p);
                assert!(us.contains(&id), "page {p} missing unit {id}");
                if us.len() > 1 {
                    shared += 1;
                }
            }
        }
        for (p, _) in m.unit_sentinels(id) {
            assert!(idx.units(p).contains(&id));
        }
    }
    // Offsets are 32-B aligned, not page aligned: some pages are shared.
    assert!(shared > 0);
    // A page beyond the file has no owners; the header page has none either.
    assert!(idx.owners(u64::MAX / 8192).is_empty());
    assert!(idx.owners(0).is_empty());
    // Dense owners exist (attention tensors).
    let dense_page = m.dense[1].offset / 4096;
    assert!(idx
        .owners(dense_page)
        .iter()
        .any(|o| matches!(o, Owner::Dense { .. })));
}

#[test]
fn page_count_handles_shared_pages() {
    let (m, _) = load("tiny_moe.gguf");
    for id in 0..m.units.len() as u32 {
        let n = m.unit_page_count(id);
        let lower = m.units[id as usize].bytes.div_ceil(4096);
        let upper: u64 = m.unit_pages(id).iter().map(|(a, b)| b - a + 1).sum();
        assert!(lower <= n && n <= upper);
    }
}

#[test]
fn map_json_round_trip() {
    let (m, _) = load("tiny_moe.gguf");
    let s = serde_json::to_string(&m).unwrap();
    let mut back: ExpertMap = serde_json::from_str(&s).unwrap();
    back.reindex();
    assert_eq!(back.units, m.units);
    assert_eq!(back.unit_id(1, 3), m.unit_id(1, 3));
}

/// Optional: real model headers (first MBs of a GGUF) in $MOEPAGER_REAL_HEADERS.
#[test]
fn real_headers_if_available() {
    let Ok(dir) = std::env::var("MOEPAGER_REAL_HEADERS") else {
        eprintln!("MOEPAGER_REAL_HEADERS not set; skipping");
        return;
    };
    let p = Path::new(&dir).join("qwen3-30b-a3b-q4km.head");
    if !p.exists() {
        return;
    }
    let h = parse_file(&p).unwrap();
    let m = ExpertMap::from_header(&h, 4096, Some(18_556_686_912));
    assert_eq!((m.n_layers, m.n_experts, m.top_k), (48, 128, Some(8)));
    assert_eq!(m.units.len(), 48 * 128);
    assert_eq!(m.units[0].slices.len(), 3);
    assert_eq!(m.units[0].bytes, 884_736 * 2 + 1_290_240);
    assert!(m.warnings.is_empty(), "{:?}", m.warnings);
}

#[test]
fn repack_report_on_fixture() {
    use mp_gguf::repack::{repack_report, CpuFeatures};
    let p = fixtures().join("tiny_moe.gguf");
    let h = parse_file(&p).unwrap();
    let m = ExpertMap::from_header(&h, 4096, None);
    let avx2 = CpuFeatures {
        x86_avx2: true,
        ..Default::default()
    };
    let r = repack_report(&h, &m, &avx2);
    // up/gate are Q4_K in every layer; down is Q4_K in layer 1 only.
    let names: Vec<&str> = r.expert_tensors.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"blk.0.ffn_up_exps.weight"));
    assert!(names.contains(&"blk.1.ffn_down_exps.weight"));
    assert!(!names.contains(&"blk.0.ffn_down_exps.weight"));
    assert!(r.fraction() > 0.5 && r.fraction() < 1.0);
    assert_eq!(
        repack_report(&h, &m, &CpuFeatures::default()).fraction(),
        0.0
    );
}
