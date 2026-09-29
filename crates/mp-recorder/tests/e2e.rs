//! End-to-end on a real file: replay a synthetic trace, recover it with the
//! black-box recorders, compare with ground truth.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use mp_gguf::{parse_file, ExpertMap};
use mp_os::{drop_file_cache, MappedFile};
use mp_recorder::{
    page_to_expert, replay, ConvertConfig, ReplayConfig, ScanRecorder, SentinelRecorder, TouchMode,
};
use mp_synth::SynthParams;
use mp_trace::{infer_tokens, ExpertEvent, Steps};

/// Tests share one file and observe the global page cache: run them one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

fn fixture() -> Option<PathBuf> {
    static GEN: Once = Once::new();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = root.join("fixtures/generated");
    GEN.call_once(|| {
        if !dir.join("big_moe.gguf").exists() {
            let ok = Command::new("python3")
                .arg(root.join("python/moepager_tools/gguf_fixtures.py"))
                .arg(&dir)
                .arg("--big")
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            assert!(ok, "fixture generation failed");
        }
    });
    let p = dir.join("big_moe.gguf");
    // tmpfs pages cannot be dropped; skip there.
    drop_file_cache(&p).ok()?;
    let m = MappedFile::open(&p).ok()?;
    let mut v = Vec::new();
    mp_os::ResidencyProbe::resident(&m, 0, m.len() / 4096, &mut v).ok()?;
    if v.iter().filter(|&&b| b).count() > v.len() / 2 {
        eprintln!("cannot drop page cache here (tmpfs?); skipping");
        return None;
    }
    Some(p)
}

fn setup(p: &Path, tokens: u32) -> (ExpertMap, Steps) {
    let h = parse_file(p).unwrap();
    let map = ExpertMap::from_header(&h, 4096, None);
    let sp = SynthParams {
        n_layers: map.n_layers,
        n_experts: map.n_experts,
        top_k: 2,
        tokens,
        seed: 11,
        ..Default::default()
    };
    let (th, ev) = mp_synth::generate(&sp);
    (map, Steps::from_events(&th, &ev))
}

fn sets(s: &Steps) -> Vec<BTreeSet<(u16, u16)>> {
    s.steps
        .iter()
        .map(|st| {
            st.layers
                .iter()
                .flat_map(|l| l.experts.iter().map(move |&e| (l.layer, e)))
                .collect()
        })
        .collect()
}

#[test]
fn scan_recorder_recovers_every_unit_exactly() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(p) = fixture() else { return };
    let (map, truth) = setup(&p, 6);
    let probe = MappedFile::open(&p).unwrap();
    let mut rec = ScanRecorder::new(probe.len(), 4096);
    drop_file_cache(&p).unwrap();
    rec.baseline(&probe).unwrap();
    // Readahead off: with btrfs's 4 MiB window, the first sequential read
    // spills into neighbouring experts' interiors (see PHASES known issues).
    let cfg = ReplayConfig {
        mode: TouchMode::Pread,
        no_readahead: true,
        ..Default::default()
    };
    let mut per_token = Vec::new();
    replay(&p, &map, &truth, &cfg, |_| {
        let ev = rec.tick(&probe, 0).unwrap();
        let ex = page_to_expert(&ev, &map, &ConvertConfig::default());
        per_token.push(
            ex.iter()
                .map(|e| (e.layer, e.expert))
                .collect::<BTreeSet<_>>(),
        );
        // Start every token cold, so each use is a miss, and re-baseline.
        drop_file_cache(&p).unwrap();
        rec.baseline(&probe).unwrap();
    })
    .unwrap();
    let want = sets(&truth);
    // Each token after the first starts cold, so insertions == uses.
    for (i, (got, want)) in per_token.iter().zip(&want).enumerate() {
        if got != want {
            eprintln!(
                "token {i}: extra {:?} missing {:?}",
                got.difference(want).collect::<Vec<_>>(),
                want.difference(got).collect::<Vec<_>>()
            );
        }
        assert_eq!(got, want, "token {i}");
    }
}

#[test]
fn sentinel_recorder_recovers_trace_with_inferred_tokens() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(p) = fixture() else { return };
    let (map, truth) = setup(&p, 12);
    drop_file_cache(&p).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let events: Arc<Mutex<Vec<ExpertEvent>>> = Arc::default();
    let start = Instant::now();
    let rec_thread = {
        let (stop, events, map, p) = (stop.clone(), events.clone(), map.clone(), p.clone());
        std::thread::spawn(move || {
            let probe = MappedFile::open(&p).unwrap();
            let mut rec = SentinelRecorder::new(&map);
            assert_eq!(rec.unreliable_units, 0);
            rec.baseline(&probe).unwrap();
            while !stop.load(Ordering::Relaxed) {
                let t = start.elapsed().as_nanos() as u64;
                let ev = rec.tick(&probe, t).unwrap();
                events.lock().unwrap().extend(ev);
                std::thread::sleep(Duration::from_micros(200));
            }
        })
    };
    std::thread::sleep(Duration::from_millis(20));
    let cfg = ReplayConfig {
        mode: TouchMode::Pread,
        drop_each_token: true,
        t_layer: Duration::from_millis(4),
        t_token_gap: Duration::from_millis(4),
        no_readahead: true,
        ..Default::default()
    };
    replay(&p, &map, &truth, &cfg, |_| {
        std::thread::sleep(Duration::from_millis(4))
    })
    .unwrap();
    std::thread::sleep(Duration::from_millis(20));
    stop.store(true, Ordering::Relaxed);
    rec_thread.join().unwrap();
    let mut ev = events.lock().unwrap().clone();
    let n_tok = infer_tokens(&mut ev, 0);
    let got = sets(&Steps::from_events(
        &mp_trace::TraceHeader::new("t", map.n_layers, map.n_experts),
        &ev,
    ));
    let want = sets(&truth);
    let (mut hit, mut total, mut extra) = (0, 0, 0);
    for (g, w) in got.iter().zip(&want) {
        hit += g.intersection(w).count();
        total += w.len();
        extra += g.difference(w).count();
    }
    let recall = hit as f64 / total as f64;
    eprintln!(
        "tokens inferred {n_tok}/{} recall {recall:.3} extra {extra}",
        want.len()
    );
    assert_eq!(n_tok as usize, want.len());
    assert!(recall >= 0.9, "recall {recall}");
    assert!(extra <= total / 10, "extra {extra}");
}
