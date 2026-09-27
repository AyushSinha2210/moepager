use mp_core::CostModel;
use mp_sim::{run_policy, simulate, Lru, NoControl, PolicyOptions, SimConfig, POLICIES};
use mp_synth::SynthParams;
use mp_trace::Steps;

fn trace(p: SynthParams) -> Steps {
    let (h, ev) = mp_synth::generate(&p);
    Steps::from_events(&h, &ev)
}

fn small() -> SynthParams {
    SynthParams {
        n_layers: 6,
        n_experts: 32,
        top_k: 4,
        tokens: 300,
        ..Default::default()
    }
}

/// Unit sizes that vary by layer, like Q6_K/Q4_K alternation in real files.
fn sizes(s: &Steps) -> Vec<u64> {
    (0..s.n_units())
        .map(|u| {
            if (u / s.n_experts) % 2 == 0 {
                3_059_712
            } else {
                2_654_208
            }
        })
        .collect()
}

fn footprint(sz: &[u64]) -> u64 {
    sz.iter().sum()
}

#[test]
fn lru_matches_analyzer_miss_ratio_curve() {
    // Two independent implementations must agree exactly.
    let s = trace(small());
    let sz = sizes(&s);
    let fp = footprint(&sz);
    let caps: Vec<u64> = [0.05, 0.1, 0.25, 0.5, 0.8, 1.0]
        .iter()
        .map(|f| (fp as f64 * f) as u64)
        .collect();
    let seq = mp_analyze::access_sequence(&s);
    let r = mp_analyze::reuse_distances(&seq, s.n_units(), &|u| sz[u as usize]);
    let mrc = mp_analyze::lru_mrc(&r, &caps);
    for (c, m) in caps.iter().zip(&mrc) {
        let cfg = SimConfig {
            capacity_bytes: *c,
            ..Default::default()
        };
        let got = simulate(&s, &sz, &cfg, &mut Lru::new(sz.len()), &mut NoControl);
        assert_eq!(got.misses, m.misses, "capacity {c}");
        assert_eq!(got.demand_bytes, m.miss_bytes, "capacity {c}");
    }
}

#[test]
fn belady_is_a_lower_bound_with_uniform_sizes() {
    let s = trace(SynthParams {
        skew: 1.1,
        ..small()
    });
    let sz = vec![1_000_000u64; s.n_units() as usize];
    let opt = PolicyOptions::default();
    for frac in [0.1, 0.3, 0.6] {
        let cfg = SimConfig {
            capacity_bytes: (footprint(&sz) as f64 * frac) as u64,
            ..Default::default()
        };
        let bel = run_policy("belady", &s, &sz, &cfg, &opt)
            .unwrap()
            .metrics
            .misses;
        for p in [
            "lru",
            "lfu",
            "prefix-pin",
            "static-freq",
            "v-full",
            "v-miss",
        ] {
            let m = run_policy(p, &s, &sz, &cfg, &opt).unwrap().metrics.misses;
            assert!(bel <= m, "{p} beat belady at {frac}: {m} < {bel}");
        }
    }
}

#[test]
fn lru_misses_are_monotone_in_capacity() {
    let s = trace(small());
    let sz = sizes(&s);
    let mut last = u64::MAX;
    for i in 1..=10 {
        let cfg = SimConfig {
            capacity_bytes: footprint(&sz) * i / 10,
            ..Default::default()
        };
        let m = simulate(&s, &sz, &cfg, &mut Lru::new(sz.len()), &mut NoControl).misses;
        assert!(m <= last);
        last = m;
    }
}

#[test]
fn completion_cuts_stall_not_misses() {
    let s = trace(small());
    let sz = sizes(&s);
    let base = SimConfig {
        capacity_bytes: footprint(&sz) / 4,
        cost: CostModel {
            demand_bw: 1e9,
            bulk_bw: 3e9,
            ..Default::default()
        },
        ..Default::default()
    };
    let opt = PolicyOptions::default();
    let off = run_policy("lru", &s, &sz, &base, &opt).unwrap().metrics;
    let on = run_policy(
        "lru",
        &s,
        &sz,
        &SimConfig {
            completion: true,
            ..base
        },
        &opt,
    )
    .unwrap()
    .metrics;
    assert_eq!(off.misses, on.misses);
    assert!(on.stall_ns < 0.6 * off.stall_ns);
    assert!(on.tokens_per_s() > off.tokens_per_s());
}

#[test]
fn pinning_respects_budget_and_never_bypasses() {
    let s = trace(small());
    let sz = sizes(&s);
    let cfg = SimConfig {
        capacity_bytes: footprint(&sz) / 5,
        ..Default::default()
    };
    for p in [
        "prefix-pin",
        "static-freq",
        "v-full",
        "v-miss",
        "v-miss+predict",
    ] {
        let r = run_policy(p, &s, &sz, &cfg, &PolicyOptions::default()).unwrap();
        assert_eq!(r.metrics.bypassed, 0, "{p}");
        assert_eq!(r.metrics.accesses as usize, s.n_accesses());
    }
}

#[test]
fn value_residency_beats_lru_when_popularity_dominates() {
    // Strong skew, no token reuse: LRU churns the tail, frequency pinning wins.
    let s = trace(SynthParams {
        skew: 1.3,
        reuse: 0.0,
        affinity: 0.0,
        tokens: 600,
        ..small()
    });
    let sz = sizes(&s);
    let cfg = SimConfig {
        capacity_bytes: footprint(&sz) / 5,
        warmup_tokens: 100,
        ..Default::default()
    };
    let opt = PolicyOptions::default();
    let lru = run_policy("lru", &s, &sz, &cfg, &opt)
        .unwrap()
        .metrics
        .misses;
    let v = run_policy("v-full", &s, &sz, &cfg, &opt)
        .unwrap()
        .metrics
        .misses;
    let vm = run_policy("v-miss", &s, &sz, &cfg, &opt)
        .unwrap()
        .metrics
        .misses;
    assert!(v < lru, "v-full {v} vs lru {lru}");
    assert!(vm < lru, "v-miss {vm} vs lru {lru}");
}

#[test]
fn warmup_excludes_tokens_and_all_policies_run() {
    let s = trace(small());
    let sz = sizes(&s);
    let cfg = SimConfig {
        capacity_bytes: footprint(&sz) / 3,
        warmup_tokens: 50,
        ..Default::default()
    };
    for p in POLICIES {
        let r = run_policy(p, &s, &sz, &cfg, &PolicyOptions::default()).unwrap();
        assert_eq!(r.metrics.tokens, 250, "{p}");
        assert!(r.metrics.hits + r.metrics.misses == r.metrics.accesses);
    }
    assert!(run_policy("nope", &s, &sz, &cfg, &PolicyOptions::default()).is_err());
}
