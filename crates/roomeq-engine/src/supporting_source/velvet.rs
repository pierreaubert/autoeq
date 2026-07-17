//! Velvet-noise decorrelation sequence generation.
//!
//! Velvet noise is a sparse pseudo-random impulse sequence whose density is
//! low enough to sound smooth (like noise) but high enough to preserve
//! timbre. When convolved with the supporting-source FIR it makes the
//! support path incoherent with the primary path so the two sum
//! energetically rather than coherently.

/// Generate a sparse velvet-noise sequence.
///
/// # Arguments
/// * `n_taps` - Total length of the output sequence.
/// * `density` - Average number of non-zero impulses per unit length. The
///   paper's recommended velvet noise has about 1 impulse per 3–5 ms.
///   Default behaviour uses a fixed seed-derived grid for reproducibility.
/// * `seed` - Deterministic seed.
///
/// # Returns
/// A vector of `n_taps` floats, most of which are zero. Non-zero samples are
/// ±1 with equal probability.
pub fn generate_velvet_noise(n_taps: usize, density: f64, seed: u64) -> Vec<f64> {
    if n_taps == 0 {
        return Vec::new();
    }

    let mut taps = vec![0.0; n_taps];
    if density <= 0.0 {
        return taps;
    }

    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut pos = 0_usize;
    let mean_spacing = (1.0 / density).max(1.0);

    while pos < n_taps {
        // Random sign
        state = xorshift64(state);
        let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
        taps[pos] = sign;

        // Random spacing between impulses
        state = xorshift64(state);
        let spacing = 1.0 + mean_spacing * ((state as f64) / (u64::MAX as f64));
        pos += spacing as usize;
    }

    taps
}

fn xorshift64(mut state: u64) -> u64 {
    if state == 0 {
        state = 0xdeadbeefcafebabe;
    }
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn velvet_noise_is_sparse() {
        let seq = generate_velvet_noise(4096, 0.1, 1);
        let non_zero = seq.iter().filter(|&&x| x != 0.0).count();
        assert!(non_zero > 0, "velvet noise should have some impulses");
        assert!(
            non_zero < seq.len() / 2,
            "velvet noise should be sparse: {} / {}",
            non_zero,
            seq.len()
        );
    }

    #[test]
    fn velvet_noise_is_deterministic() {
        let a = generate_velvet_noise(4096, 0.1, 42);
        let b = generate_velvet_noise(4096, 0.1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn velvet_noise_values_are_plus_minus_one() {
        let seq = generate_velvet_noise(4096, 0.1, 7);
        for &x in &seq {
            assert!(x == 0.0 || x == 1.0 || x == -1.0);
        }
    }

    #[test]
    fn zero_density_returns_zeros() {
        let seq = generate_velvet_noise(100, 0.0, 1);
        assert!(seq.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn zero_length_returns_empty() {
        let seq = generate_velvet_noise(0, 0.1, 1);
        assert!(seq.is_empty());
    }
}
