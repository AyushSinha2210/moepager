//! Small deterministic PRNG (xoshiro256**, seeded by SplitMix64) and a
//! table-based Zipf sampler. Kept in-tree so synthetic traces stay
//! bit-identical across dependency upgrades.

#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        Rng {
            s: [
                splitmix64(&mut x),
                splitmix64(&mut x),
                splitmix64(&mut x),
                splitmix64(&mut x),
            ],
        }
    }

    /// Independent stream derived from this seed and a label.
    pub fn fork(&mut self, label: u64) -> Rng {
        Rng::new(self.next_u64() ^ label.wrapping_mul(0xA24B_AED4_963E_E407))
    }

    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [0, n) without modulo bias (Lemire). `n` must be > 0.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        loop {
            let m = (self.next_u64() as u128) * (n as u128);
            let lo = m as u64;
            if lo >= n.wrapping_neg() % n {
                return (m >> 64) as u64;
            }
        }
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            v.swap(i, j);
        }
    }

    /// Normal(0, 1) via Box–Muller.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(f64::MIN_POSITIVE);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Zipf(s) over n ranks via an explicit CDF table. `s = 0` is uniform.
#[derive(Debug, Clone)]
pub struct Zipf {
    cdf: Vec<f64>,
}

impl Zipf {
    pub fn new(n: usize, s: f64) -> Self {
        assert!(n > 0 && s >= 0.0);
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0;
        for k in 1..=n {
            acc += 1.0 / (k as f64).powf(s);
            cdf.push(acc);
        }
        for c in &mut cdf {
            *c /= acc;
        }
        Zipf { cdf }
    }

    /// A rank in [0, n): 0 is the most popular.
    pub fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.f64();
        self.cdf
            .partition_point(|&c| c <= u)
            .min(self.cdf.len() - 1)
    }

    pub fn prob(&self, rank: usize) -> f64 {
        self.cdf[rank] - if rank == 0 { 0.0 } else { self.cdf[rank - 1] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seed_sensitive() {
        let a: Vec<u64> = (0..5)
            .map({
                let mut r = Rng::new(1);
                move |_| r.next_u64()
            })
            .collect();
        let b: Vec<u64> = (0..5)
            .map({
                let mut r = Rng::new(1);
                move |_| r.next_u64()
            })
            .collect();
        let mut c = Rng::new(2);
        assert_eq!(a, b);
        assert_ne!(a[0], c.next_u64());
        // Golden value (computed independently in Python) guards the algorithm.
        assert_eq!(Rng::new(42).next_u64(), 1_546_998_764_402_558_742);
    }

    #[test]
    fn below_is_in_range_and_roughly_uniform() {
        let mut r = Rng::new(3);
        let mut hist = [0u32; 7];
        for _ in 0..70_000 {
            hist[r.below(7) as usize] += 1;
        }
        assert!(
            hist.iter().all(|&h| (9_000..11_000).contains(&h)),
            "{hist:?}"
        );
    }

    #[test]
    fn zipf_matches_its_pmf() {
        let z = Zipf::new(16, 1.2);
        let mut r = Rng::new(9);
        let mut hist = [0u32; 16];
        let n = 200_000;
        for _ in 0..n {
            hist[z.sample(&mut r)] += 1;
        }
        for (k, &h) in hist.iter().enumerate() {
            let p = z.prob(k);
            let tol = 5.0 * (p * (1.0 - p) / n as f64).sqrt() + 1e-4;
            assert!((h as f64 / n as f64 - p).abs() < tol, "rank {k}");
        }
        assert!((Zipf::new(4, 0.0).prob(3) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut v: Vec<u32> = (0..100).collect();
        Rng::new(5).shuffle(&mut v);
        let mut s = v.clone();
        s.sort();
        assert_eq!(s, (0..100).collect::<Vec<_>>());
        assert_ne!(v, s);
    }
}
