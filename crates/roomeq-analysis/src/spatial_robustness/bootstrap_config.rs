/// Configuration for bootstrap confidence-band estimation.
///
/// Implements case-bootstrap on the input measurement curves: each of `num_resamples`
/// resamples draws N curves with replacement from the N input measurements, then
/// computes the RMS-averaged response. Per-frequency percentile bands are extracted
/// from the resulting B resampled means.
///
/// This estimates sampled listening-area variability under an independent-position
/// assumption. It does **not** estimate repeat-sweep noise, microphone calibration,
/// time variance, or interpolation uncertainty. Nearby positions are correlated, so
/// reports must label this as a spatial case-bootstrap and should pair it with
/// held-out/leave-one-position-out evidence.
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Effective independent spatial sample size. When smaller than the
    /// nominal curve count, each resample draws this many cases.
    pub effective_sample_size: Option<f64>,
    /// Number of bootstrap resamples B. Typical: 200..1000. Default: 400.
    pub num_resamples: usize,
    /// Two-sided confidence level α — band covers `[α/2, 1-α/2]`. Default: 0.10 (90 % CI).
    pub alpha: f64,
    /// PRNG seed for determinism.
    pub seed: u64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            effective_sample_size: None,
            num_resamples: 400,
            alpha: 0.10,
            seed: 0xC0FFEE,
        }
    }
}
