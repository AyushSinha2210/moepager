//! Synthetic MoE expert-access traces with controllable structure.
//!
//! Per (token, layer), `top_k` distinct experts are drawn slot by slot from
//! a mixture of three mechanisms:
//!
//! * **token-to-token reuse** (prob `reuse`): an expert this layer used on
//!   the previous token;
//! * **cross-layer affinity** (prob `affinity`): a preferred successor of
//!   one of the experts chosen at layer `l-1` for this token;
//! * **popularity** (otherwise): a Zipf(`skew`) draw over a per-layer random
//!   permutation of experts.
//!
//! Optional drift re-shuffles part of every layer's popularity order every
//! `drift_every` tokens (domain shift). Timing follows a per-layer compute
//! model with jitter. The output is ground truth (`observation = full`).
//! These traces exercise the tooling; they are not evidence about real
//! models.

use mp_core::{Rng, Zipf};
use mp_trace::{ExpertEvent, Observation, TraceHeader};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SynthParams {
    pub n_layers: u32,
    pub n_experts: u32,
    pub top_k: u32,
    pub tokens: u32,
    pub seed: u64,
    /// Zipf exponent of per-layer expert popularity (0 = uniform).
    pub skew: f64,
    /// Per-slot probability of reusing an expert from the previous token.
    pub reuse: f64,
    /// Per-slot probability of following cross-layer affinity.
    pub affinity: f64,
    /// Preferred successors per expert at the next layer.
    pub affinity_fanout: u32,
    /// Zipf exponent over an expert's successors.
    pub affinity_skew: f64,
    /// Re-shuffle popularity every this many tokens (0 = never).
    pub drift_every: u32,
    /// Fraction of popularity ranks re-shuffled at each drift.
    pub drift_frac: f64,
    /// Mean compute time per layer (ns).
    pub t_layer_ns: u64,
    /// Relative standard deviation of per-layer time.
    pub jitter: f64,
    /// Extra time per token outside the layers (sampling, attention setup).
    pub t_token_overhead_ns: u64,
}

impl Default for SynthParams {
    fn default() -> Self {
        SynthParams {
            n_layers: 24,
            n_experts: 64,
            top_k: 8,
            tokens: 256,
            seed: 1,
            skew: 0.9,
            reuse: 0.3,
            affinity: 0.3,
            affinity_fanout: 4,
            affinity_skew: 1.0,
            drift_every: 0,
            drift_frac: 0.2,
            t_layer_ns: 800_000,
            jitter: 0.1,
            t_token_overhead_ns: 2_000_000,
        }
    }
}

impl SynthParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.n_layers == 0 || self.n_experts == 0 || self.tokens == 0 {
            return Err("n_layers, n_experts and tokens must be > 0".into());
        }
        if self.top_k == 0 || self.top_k > self.n_experts {
            return Err(format!("top_k must be in 1..={}", self.n_experts));
        }
        if self.n_layers > u16::MAX as u32 + 1 || self.n_experts > u16::MAX as u32 + 1 {
            return Err("layers/experts must fit in u16".into());
        }
        for (n, p) in [
            ("reuse", self.reuse),
            ("affinity", self.affinity),
            ("drift_frac", self.drift_frac),
        ] {
            if !(0.0..=1.0).contains(&p) {
                return Err(format!("{n} must be in [0, 1]"));
            }
        }
        if self.reuse + self.affinity > 1.0 {
            return Err("reuse + affinity must be <= 1".into());
        }
        if self.affinity_fanout == 0 || self.affinity_fanout > self.n_experts {
            return Err("affinity_fanout must be in 1..=n_experts".into());
        }
        Ok(())
    }

    pub fn header(&self) -> TraceHeader {
        let mut h = TraceHeader::new("synth", self.n_layers, self.n_experts);
        h.top_k = Some(self.top_k);
        h.observation = Observation::Full;
        h.params = serde_json::to_value(self).unwrap_or_default();
        h
    }
}

/// Generate a trace. Panics if `p` is invalid (call `validate` first).
pub fn generate(p: &SynthParams) -> (TraceHeader, Vec<ExpertEvent>) {
    p.validate().expect("invalid synth params");
    let (nl, ne, k) = (p.n_layers as usize, p.n_experts as usize, p.top_k as usize);
    let mut root = Rng::new(p.seed);
    let mut r_struct = root.fork(1);
    let mut r_route = root.fork(2);
    let mut r_time = root.fork(3);

    // Per-layer popularity order: perm[l][rank] = expert.
    let mut perm: Vec<Vec<u16>> = (0..nl)
        .map(|_| {
            let mut v: Vec<u16> = (0..ne as u16).collect();
            r_struct.shuffle(&mut v);
            v
        })
        .collect();
    // affinity[l][e] = preferred successors at layer l+1.
    let affinity: Vec<Vec<Vec<u16>>> = (0..nl.saturating_sub(1))
        .map(|_| {
            (0..ne)
                .map(|_| {
                    let mut v: Vec<u16> = (0..ne as u16).collect();
                    r_struct.shuffle(&mut v);
                    v.truncate(p.affinity_fanout as usize);
                    v
                })
                .collect()
        })
        .collect();
    let pop = Zipf::new(ne, p.skew);
    let aff = Zipf::new(p.affinity_fanout as usize, p.affinity_skew);

    let mut events = Vec::with_capacity(p.tokens as usize * nl * k);
    let mut prev: Vec<Vec<u16>> = vec![Vec::new(); nl];
    let mut t: u64 = 0;
    for tok in 0..p.tokens {
        if p.drift_every > 0 && tok > 0 && tok % p.drift_every == 0 {
            let m = ((ne as f64 * p.drift_frac).round() as usize).min(ne);
            for order in perm.iter_mut() {
                let mut ranks: Vec<usize> = (0..ne).collect();
                r_struct.shuffle(&mut ranks);
                ranks.truncate(m);
                let mut ids: Vec<u16> = ranks.iter().map(|&r| order[r]).collect();
                r_struct.shuffle(&mut ids);
                for (&r, id) in ranks.iter().zip(ids) {
                    order[r] = id;
                }
            }
        }
        let mut cur: Vec<Vec<u16>> = vec![Vec::new(); nl];
        for l in 0..nl {
            let mut chosen: Vec<u16> = Vec::with_capacity(k);
            while chosen.len() < k {
                let mut picked = None;
                for _ in 0..64 {
                    let u = r_route.f64();
                    let cand = if u < p.reuse && !prev[l].is_empty() {
                        prev[l][r_route.below(prev[l].len() as u64) as usize]
                    } else if u < p.reuse + p.affinity && l > 0 && !cur[l - 1].is_empty() {
                        let src = cur[l - 1][r_route.below(cur[l - 1].len() as u64) as usize];
                        affinity[l - 1][src as usize][aff.sample(&mut r_route)]
                    } else {
                        perm[l][pop.sample(&mut r_route)]
                    };
                    if !chosen.contains(&cand) {
                        picked = Some(cand);
                        break;
                    }
                }
                let e = picked.unwrap_or_else(|| {
                    *perm[l]
                        .iter()
                        .find(|e| !chosen.contains(e))
                        .expect("top_k <= n_experts")
                });
                chosen.push(e);
            }
            chosen.sort_unstable();
            // ggml's mul_mat_id walks selected experts in index order.
            for (i, &e) in chosen.iter().enumerate() {
                events.push(ExpertEvent {
                    t_ns: t + i as u64 * 1_000,
                    token: tok,
                    layer: l as u16,
                    expert: e,
                });
            }
            let dt = p.t_layer_ns as f64 * (1.0 + p.jitter * r_time.normal());
            t += dt.max(0.1 * p.t_layer_ns as f64) as u64;
            cur[l] = chosen;
        }
        t += p.t_token_overhead_ns;
        prev = cur;
    }
    (p.header(), events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mp_trace::Steps;

    fn small() -> SynthParams {
        SynthParams {
            n_layers: 8,
            n_experts: 32,
            top_k: 4,
            tokens: 400,
            ..Default::default()
        }
    }

    fn steps(p: &SynthParams) -> Steps {
        let (h, ev) = generate(p);
        Steps::from_events(&h, &ev)
    }

    /// Mean fraction of a layer's experts that were also used on the previous token.
    fn measured_reuse(s: &Steps) -> f64 {
        let mut hit = 0usize;
        let mut tot = 0usize;
        for w in s.steps.windows(2) {
            for (a, b) in w[0].layers.iter().zip(&w[1].layers) {
                hit += b.experts.iter().filter(|e| a.experts.contains(e)).count();
                tot += b.experts.len();
            }
        }
        hit as f64 / tot as f64
    }

    /// Share of accesses that go to the top 10% of experts per layer.
    fn top_share(s: &Steps) -> f64 {
        let mut counts = vec![vec![0u32; s.n_experts as usize]; s.n_layers as usize];
        for st in &s.steps {
            for l in &st.layers {
                for &e in &l.experts {
                    counts[l.layer as usize][e as usize] += 1;
                }
            }
        }
        let top = (s.n_experts as usize / 10).max(1);
        let mut hot = 0u32;
        let mut all = 0u32;
        for mut c in counts {
            c.sort_unstable_by(|a, b| b.cmp(a));
            hot += c[..top].iter().sum::<u32>();
            all += c.iter().sum::<u32>();
        }
        hot as f64 / all as f64
    }

    #[test]
    fn shape_and_determinism() {
        let p = small();
        let (h, a) = generate(&p);
        let (_, b) = generate(&p);
        assert_eq!(a, b);
        assert_eq!(a.len(), 400 * 8 * 4);
        assert_eq!(h.top_k, Some(4));
        let s = Steps::from_events(&h, &a);
        assert!(s
            .steps
            .iter()
            .all(|st| st.layers.len() == 8 && st.layers.iter().all(|l| l.experts.len() == 4)));
        assert!(a.windows(2).all(|w| w[0].t_ns <= w[1].t_ns));
        let (_, c) = generate(&SynthParams { seed: 2, ..p });
        assert_ne!(a, c);
    }

    #[test]
    fn reuse_knob_is_monotone() {
        let lo = measured_reuse(&steps(&SynthParams {
            reuse: 0.0,
            ..small()
        }));
        let hi = measured_reuse(&steps(&SynthParams {
            reuse: 0.6,
            ..small()
        }));
        assert!(hi > lo + 0.3, "lo={lo} hi={hi}");
        assert!(
            hi >= 0.6 * 0.9,
            "reuse slots alone should give ~0.6, got {hi}"
        );
    }

    #[test]
    fn skew_knob_is_monotone() {
        let base = SynthParams {
            reuse: 0.0,
            affinity: 0.0,
            ..small()
        };
        let flat = top_share(&steps(&SynthParams {
            skew: 0.0,
            ..base.clone()
        }));
        let steep = top_share(&steps(&SynthParams { skew: 1.5, ..base }));
        assert!(steep > flat + 0.15, "flat={flat} steep={steep}");
    }

    #[test]
    fn drift_changes_popularity() {
        let p = SynthParams {
            reuse: 0.0,
            affinity: 0.0,
            skew: 1.5,
            tokens: 200,
            ..small()
        };
        let still = steps(&p);
        let moving = steps(&SynthParams {
            drift_every: 50,
            drift_frac: 1.0,
            ..p
        });
        // Same seed and same first 50 tokens; after the first drift they diverge.
        assert_eq!(still.steps[..50], moving.steps[..50]);
        assert_ne!(still.steps[50..], moving.steps[50..]);
    }

    #[test]
    fn validation() {
        assert!(SynthParams {
            top_k: 0,
            ..small()
        }
        .validate()
        .is_err());
        assert!(SynthParams {
            top_k: 33,
            ..small()
        }
        .validate()
        .is_err());
        assert!(SynthParams {
            reuse: 0.7,
            affinity: 0.5,
            ..small()
        }
        .validate()
        .is_err());
        assert!(small().validate().is_ok());
    }
}
