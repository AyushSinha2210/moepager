//! Routing statistics: token-to-token reuse, popularity, cross-layer
//! transitions (with online predictability), and per-layer timing.

use std::collections::HashMap;

use mp_trace::Steps;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ReuseStats {
    /// Mean fraction of a layer's experts also used by that layer on the previous token.
    pub mean: f64,
    pub per_layer: Vec<f64>,
}

pub fn token_reuse(s: &Steps) -> ReuseStats {
    let nl = s.n_layers as usize;
    let mut hit = vec![0u64; nl];
    let mut tot = vec![0u64; nl];
    for w in s.steps.windows(2) {
        let prev: HashMap<u16, &Vec<u16>> =
            w[0].layers.iter().map(|l| (l.layer, &l.experts)).collect();
        for l in &w[1].layers {
            let Some(p) = prev.get(&l.layer) else {
                continue;
            };
            let li = l.layer as usize;
            if li >= nl {
                continue;
            }
            hit[li] += l
                .experts
                .iter()
                .filter(|e| p.binary_search(e).is_ok())
                .count() as u64;
            tot[li] += l.experts.len() as u64;
        }
    }
    let per_layer: Vec<f64> = hit
        .iter()
        .zip(&tot)
        .map(|(&h, &t)| if t == 0 { 0.0 } else { h as f64 / t as f64 })
        .collect();
    let mean = ratio(hit.iter().sum(), tot.iter().sum());
    ReuseStats { mean, per_layer }
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Popularity {
    /// Share of accesses going to the top 10 % of experts, averaged over layers.
    pub top10_share: f64,
    /// Normalised entropy of the per-layer expert distribution (1 = uniform), averaged.
    pub norm_entropy: f64,
    /// Fraction of all units ever touched.
    pub units_touched: f64,
}

pub fn popularity(s: &Steps) -> Popularity {
    let ne = s.n_experts as usize;
    let mut counts = vec![vec![0u64; ne]; s.n_layers as usize];
    for st in &s.steps {
        for l in &st.layers {
            for &e in &l.experts {
                if let Some(c) = counts
                    .get_mut(l.layer as usize)
                    .and_then(|c| c.get_mut(e as usize))
                {
                    *c += 1;
                }
            }
        }
    }
    let top = (ne / 10).max(1);
    let (mut share, mut ent, mut n, mut touched) = (0.0, 0.0, 0usize, 0usize);
    for mut c in counts {
        let tot: u64 = c.iter().sum();
        touched += c.iter().filter(|&&x| x > 0).count();
        if tot == 0 {
            continue;
        }
        c.sort_unstable_by(|a, b| b.cmp(a));
        share += c[..top].iter().sum::<u64>() as f64 / tot as f64;
        let h: f64 = c
            .iter()
            .filter(|&&x| x > 0)
            .map(|&x| {
                let p = x as f64 / tot as f64;
                -p * p.ln()
            })
            .sum();
        ent += if ne > 1 { h / (ne as f64).ln() } else { 0.0 };
        n += 1;
    }
    let n = n.max(1) as f64;
    Popularity {
        top10_share: share / n,
        norm_entropy: ent / n,
        units_touched: touched as f64 / (s.n_units().max(1)) as f64,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Predictability {
    /// Online recall@k of next-layer experts predicted from decayed
    /// cross-layer transition counts (learned only from the past).
    pub transition_recall: f64,
    /// Online recall@k of predicting the next layer's k most frequent experts so far.
    pub popularity_recall: f64,
    /// Online recall@k of predicting the same experts as the previous token at that layer.
    pub reuse_recall: f64,
    /// Fraction of (e → e') pair mass covered by each e's top-4 successors (in-sample).
    pub transition_concentration: f64,
}

/// Online predictability of layer l+1 given layer l (and the past).
pub fn predictability(s: &Steps) -> Predictability {
    let ne = s.n_experts as usize;
    let nl = s.n_layers as usize;
    let mut trans: HashMap<(u16, u16), Vec<f64>> = HashMap::new(); // (layer, e) -> counts over e'
    let mut freq = vec![vec![0f64; ne]; nl];
    let mut prev_tok: HashMap<u16, Vec<u16>> = HashMap::new();
    let (mut tr_hit, mut pop_hit, mut re_hit, mut tot) = (0u64, 0u64, 0u64, 0u64);
    let top = |scores: &[f64], k: usize| -> Vec<u16> {
        let mut idx: Vec<u16> = (0..scores.len() as u16).collect();
        idx.sort_by(|&a, &b| {
            scores[b as usize]
                .total_cmp(&scores[a as usize])
                .then(a.cmp(&b))
        });
        idx.truncate(k);
        idx
    };
    for st in &s.steps {
        for w in st.layers.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if b.layer != a.layer + 1 || b.layer as usize >= nl {
                continue;
            }
            let k = b.experts.len();
            let mut score = vec![0f64; ne];
            for &e in &a.experts {
                if let Some(c) = trans.get(&(a.layer, e)) {
                    for (x, v) in score.iter_mut().zip(c) {
                        *x += v;
                    }
                }
            }
            let hits = |pred: &[u16]| b.experts.iter().filter(|e| pred.contains(e)).count() as u64;
            if score.iter().any(|&x| x > 0.0) {
                tr_hit += hits(&top(&score, k));
            }
            pop_hit += hits(&top(&freq[b.layer as usize], k));
            if let Some(p) = prev_tok.get(&b.layer) {
                re_hit += hits(p);
            }
            tot += k as u64;
            for &e in &a.experts {
                let c = trans.entry((a.layer, e)).or_insert_with(|| vec![0.0; ne]);
                for &e2 in &b.experts {
                    if let Some(x) = c.get_mut(e2 as usize) {
                        *x += 1.0;
                    }
                }
            }
        }
        for l in &st.layers {
            if let Some(f) = freq.get_mut(l.layer as usize) {
                for &e in &l.experts {
                    if let Some(x) = f.get_mut(e as usize) {
                        *x += 1.0;
                    }
                }
            }
            prev_tok.insert(l.layer, l.experts.clone());
        }
    }
    let (mut covered, mut mass) = (0.0, 0.0);
    for c in trans.values() {
        let mut v = c.clone();
        v.sort_by(|a, b| b.total_cmp(a));
        covered += v.iter().take(4).sum::<f64>();
        mass += v.iter().sum::<f64>();
    }
    Predictability {
        transition_recall: ratio(tr_hit, tot),
        popularity_recall: ratio(pop_hit, tot),
        reuse_recall: ratio(re_hit, tot),
        transition_concentration: if mass > 0.0 { covered / mass } else { 0.0 },
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Timing {
    /// Gap between the first events of consecutive layers within a token (ns).
    pub layer_gap_p10: u64,
    pub layer_gap_p50: u64,
    pub layer_gap_p90: u64,
    /// Median layer gap per layer (index = layer; 0 when unknown).
    pub layer_gap_p50_per_layer: Vec<u64>,
    /// Gap between the first events of consecutive tokens (ns).
    pub token_p50: u64,
    pub token_p90: u64,
}

fn pct(v: &mut [u64], p: f64) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[((v.len() - 1) as f64 * p).round() as usize]
}

pub fn timing(s: &Steps) -> Timing {
    let mut gaps = Vec::new();
    let mut per: Vec<Vec<u64>> = vec![Vec::new(); s.n_layers as usize];
    for st in &s.steps {
        for w in st.layers.windows(2) {
            if w[1].layer == w[0].layer + 1 {
                let g = w[1].t_ns.saturating_sub(w[0].t_ns);
                gaps.push(g);
                if let Some(p) = per.get_mut(w[0].layer as usize) {
                    p.push(g);
                }
            }
        }
    }
    let mut tok: Vec<u64> = s
        .steps
        .windows(2)
        .map(|w| w[1].t_ns().saturating_sub(w[0].t_ns()))
        .collect();
    Timing {
        layer_gap_p10: pct(&mut gaps, 0.1),
        layer_gap_p50: pct(&mut gaps, 0.5),
        layer_gap_p90: pct(&mut gaps, 0.9),
        layer_gap_p50_per_layer: per.iter_mut().map(|v| pct(v, 0.5)).collect(),
        token_p50: pct(&mut tok, 0.5),
        token_p90: pct(&mut tok, 0.9),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mp_trace::{ExpertEvent, TraceHeader};

    fn steps(rows: &[(u32, u16, &[u16], u64)]) -> Steps {
        let h = TraceHeader::new("t", 3, 4);
        let ev: Vec<ExpertEvent> = rows
            .iter()
            .flat_map(|&(token, layer, ex, t)| {
                ex.iter().map(move |&expert| ExpertEvent {
                    t_ns: t,
                    token,
                    layer,
                    expert,
                })
            })
            .collect();
        Steps::from_events(&h, &ev)
    }

    #[test]
    fn reuse_and_popularity_by_hand() {
        let s = steps(&[(0, 0, &[0, 1], 0), (1, 0, &[1, 2], 10), (2, 0, &[1, 2], 20)]);
        let r = token_reuse(&s);
        // token1: 1 of 2 reused; token2: 2 of 2 → 3/4.
        assert!((r.mean - 0.75).abs() < 1e-12);
        let p = popularity(&s);
        assert!(p.top10_share > 0.0 && p.norm_entropy > 0.0 && p.norm_entropy < 1.0);
    }

    #[test]
    fn deterministic_transitions_become_predictable() {
        // Layer0 expert e always followed by layer1 expert (e+1)%4.
        let mut rows = Vec::new();
        for t in 0..50u32 {
            let e = (t % 4) as u16;
            rows.push((t, 0u16, vec![e], t as u64 * 100));
            rows.push((t, 1u16, vec![(e + 1) % 4], t as u64 * 100 + 10));
        }
        let rows: Vec<(u32, u16, &[u16], u64)> = rows
            .iter()
            .map(|(a, b, c, d)| (*a, *b, c.as_slice(), *d))
            .collect();
        let s = steps(&rows);
        let p = predictability(&s);
        assert!(p.transition_recall > 0.9, "{p:?}");
        assert!(p.popularity_recall < 0.5);
        assert!((p.transition_concentration - 1.0).abs() < 1e-12);
        let t = timing(&s);
        assert_eq!(t.layer_gap_p50, 10);
        assert_eq!(t.token_p50, 100);
    }
}
