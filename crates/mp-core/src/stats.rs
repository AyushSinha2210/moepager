//! Online routing statistics learned from observations.
//!
//! Decay uses the scaled-weight trick: an event at token `t` adds `g^t`
//! (with `g = 1/decay > 1`) instead of decaying every counter each token.
//! Rates are ratios of two such sums, so the common factor cancels, and the
//! counters are renormalised before they overflow.
//!
//! Two regimes (see ADR 0002):
//! * `Full`: every access is observed; `rate(u)` = decayed uses per token.
//! * `MissOnly`: only page-cache misses are observed; `rate(u)` = decayed
//!   misses per token *of exposure while unpinned*. That is the
//!   counterfactual a pin decision needs. A pinned unit's estimate is
//!   frozen because it accrues neither misses nor exposure.

use mp_trace::Observation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seen {
    Hit,
    Miss,
}

#[derive(Debug, Clone)]
pub struct OnlineStats {
    pub n_layers: u32,
    pub n_experts: u32,
    pub regime: Observation,
    g: f64,
    w: f64,
    count: Vec<f64>,
    exposure: Vec<f64>,
    /// Exposure shared by all units in the `Full` regime.
    exposure_all: f64,
    prior: f64,
    /// trans[layer * n_experts + e] = sparse decayed counts of next-layer experts.
    trans: Vec<Vec<(u16, f64)>>,
    /// Experts observed per layer in the current token.
    cur: Vec<Vec<u16>>,
    /// Time of the first observation per layer in the current token.
    cur_t: Vec<Option<u64>>,
    t_layer_ns: f64,
    tokens: u64,
}

impl OnlineStats {
    /// `half_life_tokens`: tokens after which an observation's weight halves.
    /// `prior_rate`: rate assumed for units never exposed (e.g. top_k / n_experts).
    pub fn new(
        n_layers: u32,
        n_experts: u32,
        regime: Observation,
        half_life_tokens: f64,
        prior_rate: f64,
    ) -> Self {
        assert!(half_life_tokens > 0.0);
        let n = (n_layers * n_experts) as usize;
        OnlineStats {
            n_layers,
            n_experts,
            regime,
            g: 2f64.powf(1.0 / half_life_tokens),
            w: 1.0,
            count: vec![0.0; n],
            exposure: vec![0.0; n],
            exposure_all: 0.0,
            prior: prior_rate,
            trans: vec![Vec::new(); n],
            cur: vec![Vec::new(); n_layers as usize],
            cur_t: vec![None; n_layers as usize],
            t_layer_ns: 0.0,
            tokens: 0,
        }
    }

    pub fn unit(&self, layer: u16, expert: u16) -> u32 {
        layer as u32 * self.n_experts + expert as u32
    }

    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    /// Record an access. In the `MissOnly` regime hits are dropped here, so
    /// callers can feed ground truth and get black-box behaviour.
    pub fn observe(&mut self, layer: u16, expert: u16, seen: Seen, t_ns: u64) {
        if self.regime == Observation::MissOnly && seen == Seen::Hit {
            return;
        }
        let l = layer as usize;
        if l >= self.cur.len() || expert as u32 >= self.n_experts {
            return;
        }
        if self.cur[l].contains(&expert) {
            return;
        }
        let u = self.unit(layer, expert) as usize;
        self.count[u] += self.w;
        if l > 0 {
            for i in 0..self.cur[l - 1].len() {
                let src = self.cur[l - 1][i];
                let row = &mut self.trans[(l - 1) * self.n_experts as usize + src as usize];
                match row.iter_mut().find(|(e, _)| *e == expert) {
                    Some(x) => x.1 += self.w,
                    None => row.push((expert, self.w)),
                }
            }
        }
        if self.cur_t[l].is_none() {
            self.cur_t[l] = Some(t_ns);
            // Layer-time estimate from the nearest earlier observed layer.
            if let Some((pl, pt)) = (0..l).rev().find_map(|p| self.cur_t[p].map(|t| (p, t))) {
                if t_ns > pt {
                    let gap = (t_ns - pt) as f64 / (l - pl) as f64;
                    self.t_layer_ns = if self.t_layer_ns == 0.0 {
                        gap
                    } else {
                        0.9 * self.t_layer_ns + 0.1 * gap
                    };
                }
            }
        }
        self.cur[l].push(expert);
    }

    /// Close the current token. `pinned(u)` tells which units were pinned
    /// during it (only matters for the `MissOnly` regime's exposure).
    pub fn end_token(&mut self, pinned: &dyn Fn(u32) -> bool) {
        match self.regime {
            Observation::Full => self.exposure_all += self.w,
            Observation::MissOnly => {
                for (u, e) in self.exposure.iter_mut().enumerate() {
                    if !pinned(u as u32) {
                        *e += self.w;
                    }
                }
            }
        }
        for c in &mut self.cur {
            c.clear();
        }
        self.cur_t.iter_mut().for_each(|t| *t = None);
        self.tokens += 1;
        self.w *= self.g;
        if self.w > 1e100 {
            self.renormalise();
        }
    }

    fn renormalise(&mut self) {
        let s = 1.0 / self.w;
        self.count.iter_mut().for_each(|x| *x *= s);
        self.exposure.iter_mut().for_each(|x| *x *= s);
        self.exposure_all *= s;
        for row in &mut self.trans {
            row.iter_mut().for_each(|x| x.1 *= s);
        }
        self.w = 1.0;
    }

    /// Estimated uses (Full) or unpinned misses (MissOnly) per token.
    pub fn rate(&self, u: u32) -> f64 {
        let u = u as usize;
        let exp = match self.regime {
            Observation::Full => self.exposure_all,
            Observation::MissOnly => self.exposure[u],
        };
        if exp <= 0.0 {
            self.prior
        } else {
            // Light shrinkage toward the prior (5 % of one token's weight) so a
            // unit exposed for a single token is not pinned at exactly 0 or 1.
            let k = 0.05 * self.w;
            (self.count[u] + self.prior * k) / (exp + k)
        }
    }

    /// Experts observed at `layer` in the current token.
    pub fn observed(&self, layer: u16) -> &[u16] {
        self.cur.get(layer as usize).map_or(&[], |v| v.as_slice())
    }

    /// Scores for experts at `layer + 1` given the experts observed at `layer`
    /// in the current token: summed decayed transition counts. Sorted desc.
    pub fn predict_next(&self, layer: u16) -> Vec<(u16, f64)> {
        let l = layer as usize;
        if l + 1 >= self.n_layers as usize {
            return Vec::new();
        }
        let mut score: Vec<(u16, f64)> = Vec::new();
        for &e in self.observed(layer) {
            for &(e2, c) in &self.trans[l * self.n_experts as usize + e as usize] {
                match score.iter_mut().find(|(x, _)| *x == e2) {
                    Some(s) => s.1 += c,
                    None => score.push((e2, c)),
                }
            }
        }
        score.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        score
    }

    /// Estimated compute time per layer (ns), 0 if unknown.
    pub fn t_layer_ns(&self) -> f64 {
        self.t_layer_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(s: &mut OnlineStats, tokens: u32, pattern: &dyn Fn(u32) -> Vec<(u16, u16, Seen)>) {
        for t in 0..tokens {
            for (i, (l, e, seen)) in pattern(t).into_iter().enumerate() {
                s.observe(l, e, seen, t as u64 * 10_000 + i as u64 * 1_000);
            }
            s.end_token(&|_| false);
        }
    }

    #[test]
    fn full_rate_tracks_usage_frequency() {
        let mut s = OnlineStats::new(1, 4, Observation::Full, 1e9, 0.5);
        // Expert 0 every token, expert 1 every other token.
        feed(&mut s, 1000, &|t| {
            let mut v = vec![(0, 0, Seen::Hit)];
            if t % 2 == 0 {
                v.push((0, 1, Seen::Hit));
            }
            v
        });
        assert!((s.rate(0) - 1.0).abs() < 0.01, "{}", s.rate(0));
        assert!((s.rate(1) - 0.5).abs() < 0.01);
        assert!(s.rate(2) < 0.01);
    }

    #[test]
    fn miss_only_drops_hits_and_freezes_pinned() {
        let mut s = OnlineStats::new(1, 2, Observation::MissOnly, 1e9, 0.25);
        for t in 0..400 {
            s.observe(0, 0, if t % 4 == 0 { Seen::Miss } else { Seen::Hit }, t);
            s.end_token(&|_| false);
        }
        assert!((s.rate(0) - 0.25).abs() < 0.01, "{}", s.rate(0));
        let before = s.rate(0);
        // While pinned, nothing is observed and the estimate does not decay.
        for _ in 0..1000 {
            s.end_token(&|u| u == 0);
        }
        assert!((s.rate(0) - before).abs() < 1e-9);
        // Unit 1 was exposed and never missed: its rate falls toward 0.
        assert!(s.rate(1) < 0.01);
    }

    #[test]
    fn decay_follows_recent_behaviour() {
        let mut s = OnlineStats::new(1, 2, Observation::Full, 10.0, 0.0);
        feed(&mut s, 200, &|_| vec![(0, 0, Seen::Hit)]);
        feed(&mut s, 200, &|_| vec![(0, 1, Seen::Hit)]);
        assert!(s.rate(1) > 0.99 && s.rate(0) < 0.01);
    }

    #[test]
    fn transitions_and_layer_time() {
        let mut s = OnlineStats::new(3, 4, Observation::Full, 50.0, 0.0);
        for t in 0..20u64 {
            let base = t * 100_000;
            s.observe(0, 1, Seen::Hit, base);
            s.observe(1, 2, Seen::Hit, base + 800);
            s.observe(2, 3, Seen::Hit, base + 1_600);
            if t < 19 {
                s.end_token(&|_| false);
            }
        }
        // Current (unfinished) token has layer 0 → expert 1 observed.
        let p = s.predict_next(0);
        assert_eq!(p[0].0, 2);
        assert_eq!(s.predict_next(2), vec![]);
        assert!((s.t_layer_ns() - 800.0).abs() < 1.0);
    }

    #[test]
    fn long_runs_do_not_overflow() {
        let mut s = OnlineStats::new(1, 1, Observation::Full, 1.0, 0.0);
        feed(&mut s, 2000, &|_| vec![(0, 0, Seen::Hit)]);
        assert!(s.rate(0).is_finite() && (s.rate(0) - 1.0).abs() < 0.05);
    }
}
