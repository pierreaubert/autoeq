//! Deterministic seeded RNG for validation staging.
//!
//! A single SplitMix64 stream keeps every staged artifact (stimuli,
//! holdout splits, bootstrap resamples) reproducible from one `u64` seed
//! without depending on any upstream RNG crate version. Integer state
//! transitions are bit-exact across platforms; float synthesis built on
//! top may differ in the last ulp across libm implementations (recorded
//! in stimulus manifests via the platform string).

/// Seeded SplitMix64 generator.
#[derive(Debug, Clone)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    /// New stream from an explicit seed. Seed 0 is valid.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next `u64` of the stream.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    /// Next `f64` uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    /// Next standard-normal sample (Box-Muller, exact pair consumption).
    pub fn next_gaussian(&mut self) -> f64 {
        let mut uniform = self.next_f64();
        // Guard the log domain without biasing the stream shape: redraw is
        // deterministic given the seed.
        while uniform <= 0.0 {
            uniform = self.next_f64();
        }
        let radius = (-2.0 * uniform.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.next_f64();
        radius * angle.cos()
    }

    /// Uniform index into `len` (panics on empty, like indexing).
    pub fn next_below(&mut self, len: usize) -> usize {
        (self.next_f64() * (len as f64)) as usize % len
    }
}

#[cfg(test)]
mod seeded_tests {
    use super::*;

    #[test]
    fn stream_is_deterministic() {
        let first: Vec<u64> = {
            let mut rng = SeededRng::new(20260905);
            (0..16).map(|_| rng.next_u64()).collect()
        };
        let second: Vec<u64> = {
            let mut rng = SeededRng::new(20260905);
            (0..16).map(|_| rng.next_u64()).collect()
        };
        assert_eq!(first, second);
        assert_ne!(first, vec![0u64; 16]);
    }

    #[test]
    fn uniform_stays_in_range() {
        let mut rng = SeededRng::new(7);
        for _ in 0..1024 {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value));
        }
    }

    #[test]
    fn gaussian_has_unit_shape() {
        // Loose moment check on a fixed seed: mean ~0, variance ~1.
        let mut rng = SeededRng::new(99);
        let samples: Vec<f64> = (0..20_000).map(|_| rng.next_gaussian()).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.05, "variance {variance}");
    }
}
