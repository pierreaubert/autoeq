//! Spatial interpolation of audio measurements over a continuous listening area.
//!
//! Pairs with `math_audio_optimisation::continuous_area` to provide a
//! continuous-prior alternative to the discrete `MultiSeatMeasurements`.
//!
//! # Inputs
//!
//! - `K` calibration positions `p_k ∈ R^D` (typically D=2 for an MLP rectangle,
//!   D=1 for a couch line, D=3 for a head-volume sweep).
//! - One [`Curve`] per (subwoofer, position) pair. Each curve must carry phase.
//!
//! # Output
//!
//! At any query point `p` inside the listening area's bounding box, return one
//! [`Curve`] per sub representing the *spatially interpolated* response at `p`.
//!
//! # Method
//!
//! Inverse-distance weighting (IDW) on log-magnitude (dB SPL is already
//! log-magnitude) and permutation-invariant complex (phasor) averaging of
//! per-bin phase. IDW is parameter-light, basis-free, and well-conditioned
//! for K=4..16 scattered calibration points — the realistic regime for room
//! measurements. For tighter fits with smooth response fields, swap in
//! RBF / kriging here.
//!
//! Phase interpolation computes the weighted circular mean per bin: each
//! calibration phase θₖ contributes its unit phasor `wₖ·e^(iθₖ)` and the
//! interpolated phase is the argument of the weighted resultant. This is
//! invariant to calibration-point ordering (no arbitrary reference point)
//! and degrades gracefully for broad spreads: the resultant magnitude
//! `R ∈ [0, 1]` is reported as per-bin confidence (see
//! [`ListeningArea::interpolate_with_evidence`]). Bins with
//! `R < ambiguity_threshold` are flagged ambiguous — near-total phasor
//! cancellation means the mean angle is arbitrary under perturbation —
//! while SPL interpolation is unaffected.
//!
//! Queries outside the calibration bounding box are *out of support*.
//! [`ListeningArea::interpolate_at`] extrapolates there for backward
//! compatibility; new code should use [`ListeningArea::try_interpolate_at`]
//! or [`ListeningArea::interpolate_with_evidence`], which reject
//! non-finite and out-of-support queries.

use crate::Curve;
use crate::error::{AutoeqError, Result};
use ndarray::Array1;

/// Configuration for the spatial interpolator.
#[derive(Debug, Clone)]
pub struct ListeningAreaInterpolatorConfig {
    /// IDW power exponent. Higher values concentrate weight on the nearest
    /// calibration points; default 2.0 gives a smooth fall-off in 2D rooms.
    pub idw_power: f64,
    /// Distance offset added to the IDW denominator to avoid division by zero
    /// when a query point lands exactly on a calibration point. Has units of
    /// position. Default `1e-9`.
    pub epsilon: f64,
    /// Resultant-magnitude threshold below which an interpolated phase is
    /// flagged ambiguous (see [`InterpolatedResponse::phase_ambiguous`]).
    /// Default [`PHASE_AMBIGUITY_THRESHOLD`].
    pub ambiguity_threshold: f64,
}

/// Resultant-magnitude threshold below which an interpolated phase is
/// reported as ambiguous.
///
/// Weighted phasor sums carry ~1e-15 relative floating-point noise, so
/// `1e-6` is far above numerical noise yet far below any physically
/// meaningful inter-position agreement.
pub const PHASE_AMBIGUITY_THRESHOLD: f64 = 1e-6;

impl Default for ListeningAreaInterpolatorConfig {
    fn default() -> Self {
        Self {
            idw_power: 2.0,
            epsilon: 1e-9,
            ambiguity_threshold: PHASE_AMBIGUITY_THRESHOLD,
        }
    }
}

/// Calibration grid of measurements at K positions in R^D.
///
/// Use [`ListeningArea::interpolate_at`] to obtain virtual measurements at
/// any query point inside the bounding box of the calibration set.
#[derive(Debug, Clone)]
pub struct ListeningArea<const D: usize> {
    /// Calibration positions in R^D, indexed `[k]`.
    positions: Vec<[f64; D]>,
    /// Per-(sub, position) measurements, indexed `[sub][k]`.
    measurements: Vec<Vec<Curve>>,
    /// Number of subwoofers / drivers.
    num_subs: usize,
    /// Number of calibration positions K.
    num_positions: usize,
    /// Interpolator configuration.
    config: ListeningAreaInterpolatorConfig,
    /// Precomputed unit-phasor real parts, indexed `[sub][k]` per bin:
    /// `phasor_re[sub][k][bin] = cos(phase)`. Built once in `::new` so
    /// repeated queries (e.g. fixed quadrature points) do not redo the
    /// degree-to-radian plus cos/sin work per query.
    phasor_re: Vec<Vec<Array1<f64>>>,
    /// Precomputed unit-phasor imaginary parts (`sin(phase)`), same layout.
    phasor_im: Vec<Vec<Array1<f64>>>,
}

impl<const D: usize> ListeningArea<D> {
    /// Construct from a list of calibration positions and per-(sub, position)
    /// measurements.
    ///
    /// `measurements[sub_idx][pos_idx]` is the curve recorded by sub
    /// `sub_idx` at position `positions[pos_idx]`. All curves must:
    /// - share the same frequency grid,
    /// - carry phase data,
    /// - be aligned with the position list (same length per sub).
    pub fn new(
        positions: Vec<[f64; D]>,
        measurements: Vec<Vec<Curve>>,
        config: ListeningAreaInterpolatorConfig,
    ) -> Result<Self> {
        if measurements.is_empty() {
            return Err(AutoeqError::InvalidConfiguration {
                message: "ListeningArea requires at least one subwoofer".into(),
            });
        }
        let num_subs = measurements.len();
        let num_positions = measurements[0].len();
        if num_positions == 0 {
            return Err(AutoeqError::InvalidConfiguration {
                message: "ListeningArea requires at least one calibration position".into(),
            });
        }
        if positions.len() != num_positions {
            return Err(AutoeqError::InvalidConfiguration {
                message: format!(
                    "ListeningArea: {} positions but {} measurements per sub",
                    positions.len(),
                    num_positions
                ),
            });
        }
        if positions
            .iter()
            .flat_map(|position| position.iter())
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(AutoeqError::InvalidConfiguration {
                message: "ListeningArea calibration positions must be finite".into(),
            });
        }
        if !config.idw_power.is_finite() || config.idw_power <= 0.0 {
            return Err(AutoeqError::InvalidConfiguration {
                message: "ListeningArea idw_power must be finite and positive".into(),
            });
        }
        if !config.epsilon.is_finite() || config.epsilon < 0.0 {
            return Err(AutoeqError::InvalidConfiguration {
                message: "ListeningArea epsilon must be finite and non-negative".into(),
            });
        }
        if !config.ambiguity_threshold.is_finite() || config.ambiguity_threshold < 0.0 {
            return Err(AutoeqError::InvalidConfiguration {
                message: "ListeningArea ambiguity_threshold must be finite and non-negative"
                    .into(),
            });
        }

        // Validate all subs have the same number of positions and that all
        // curves carry phase data and share a frequency grid.
        let reference_freq = &measurements[0][0].freq;
        for (sub_idx, sub) in measurements.iter().enumerate() {
            if sub.len() != num_positions {
                return Err(AutoeqError::InvalidConfiguration {
                    message: format!(
                        "ListeningArea: sub {} has {} positions, expected {}",
                        sub_idx,
                        sub.len(),
                        num_positions
                    ),
                });
            }
            for (pos_idx, curve) in sub.iter().enumerate() {
                if curve.spl.len() != curve.freq.len() {
                    return Err(AutoeqError::InvalidMeasurement {
                        message: format!(
                            "ListeningArea: sub {} pos {} freq/spl length mismatch",
                            sub_idx, pos_idx
                        ),
                    });
                }
                if curve.phase.is_none() {
                    return Err(AutoeqError::InvalidMeasurement {
                        message: format!(
                            "ListeningArea: sub {} pos {} is missing phase data; \
                             complex spatial interpolation requires phase",
                            sub_idx, pos_idx
                        ),
                    });
                }
                if curve
                    .phase
                    .as_ref()
                    .is_some_and(|phase| phase.len() != curve.freq.len())
                {
                    return Err(AutoeqError::InvalidMeasurement {
                        message: format!(
                            "ListeningArea: sub {} pos {} freq/phase length mismatch",
                            sub_idx, pos_idx
                        ),
                    });
                }
                if !crate::frequency_grid::same_frequency_grid(reference_freq, &curve.freq) {
                    return Err(AutoeqError::InvalidMeasurement {
                        message: format!(
                            "ListeningArea: sub {} pos {} has a different frequency grid \
                             from the reference (sub 0, pos 0)",
                            sub_idx, pos_idx
                        ),
                    });
                }
                // Validate numeric content (finite SPL/phase/frequency values
                // and well-formed grids) via the shared curve contract.
                // Structural checks above run first so their specific error
                // messages are preserved.
                curve.validate(&format!("ListeningArea: sub {sub_idx} pos {pos_idx}"))?;
            }
        }

        // Precompute the unit-phasor representation once: phase presence and
        // numeric validity were established above, so the unwraps here are
        // safe. Per-query interpolation then only forms the weighted sum.
        let mut phasor_re = Vec::with_capacity(num_subs);
        let mut phasor_im = Vec::with_capacity(num_subs);
        for sub in &measurements {
            let mut sub_re = Vec::with_capacity(num_positions);
            let mut sub_im = Vec::with_capacity(num_positions);
            for curve in sub {
                let phase = curve.phase.as_ref().expect("validated above");
                sub_re.push(phase.mapv(|deg| deg.to_radians().cos()));
                sub_im.push(phase.mapv(|deg| deg.to_radians().sin()));
            }
            phasor_re.push(sub_re);
            phasor_im.push(sub_im);
        }

        Ok(Self {
            positions,
            measurements,
            num_subs,
            num_positions,
            config,
            phasor_re,
            phasor_im,
        })
    }

    /// Number of subwoofers / drivers.
    pub fn num_subs(&self) -> usize {
        self.num_subs
    }

    /// Number of calibration positions K.
    pub fn num_positions(&self) -> usize {
        self.num_positions
    }

    /// Calibration positions, indexed `[k]`.
    pub fn positions(&self) -> &[[f64; D]] {
        &self.positions
    }

    /// Axis-aligned bounding box `(lo, hi)` per axis derived from the
    /// calibration positions. Useful as a default for prior bounds.
    pub fn bounding_box(&self) -> [(f64, f64); D] {
        let mut bounds = [(f64::INFINITY, f64::NEG_INFINITY); D];
        for p in &self.positions {
            for i in 0..D {
                if p[i] < bounds[i].0 {
                    bounds[i].0 = p[i];
                }
                if p[i] > bounds[i].1 {
                    bounds[i].1 = p[i];
                }
            }
        }
        bounds
    }

    /// Returns true when `p` is a supported query: all coordinates finite
    /// and inside the calibration bounding box (inclusive).
    pub fn contains(&self, p: [f64; D]) -> bool {
        if p.iter().any(|coordinate| !coordinate.is_finite()) {
            return false;
        }
        self.bounding_box()
            .iter()
            .zip(p.iter())
            .all(|((lo, hi), x)| *x >= *lo && *x <= *hi)
    }

    /// Interpolate per-sub curves at a query position `p`.
    ///
    /// Uses inverse-distance weighting on log-magnitude (dB SPL is already
    /// log-magnitude) and permutation-invariant complex averaging of
    /// per-bin phase. Returns one [`Curve`] per sub.
    ///
    /// # Legacy support contract
    ///
    /// This method never fails: it panics on a non-finite query and
    /// *extrapolates* with IDW for out-of-support queries. New code should
    /// prefer [`Self::try_interpolate_at`] or
    /// [`Self::interpolate_with_evidence`], which reject non-finite and
    /// out-of-support queries and additionally report per-bin confidence.
    /// Kept infallible so existing downstream quadrature consumers
    /// (which may sample outside the calibration box) keep compiling and
    /// behaving as before.
    pub fn interpolate_at(&self, p: [f64; D]) -> Vec<Curve> {
        if p.iter().any(|coordinate| !coordinate.is_finite()) {
            panic!(
                "ListeningArea::interpolate_at query position must be finite; \
                 use try_interpolate_at for a fallible query"
            );
        }
        // For finite queries the fallible weight computation fails only on
        // numerically degenerate weight totals; collapse to the nearest
        // neighbour then. Out-of-support queries extrapolate with IDW
        // (legacy behavior).
        let weights = match self.idw_weights(p) {
            Ok(weights) => weights,
            Err(_) => self.nearest_weights(p),
        };
        self.interpolate_core(&weights).curves
    }

    /// Fallible interpolation: like [`Self::interpolate_at`] but rejects
    /// non-finite queries and queries outside the calibration bounding box.
    pub fn try_interpolate_at(&self, p: [f64; D]) -> Result<Vec<Curve>> {
        self.interpolate_with_evidence(p).map(|response| response.curves)
    }

    /// Fallible interpolation with support/confidence evidence.
    ///
    /// Returns the interpolated curves plus the IDW weights, the per-bin
    /// phasor resultant magnitude (`confidence`, in `[0, 1]`) and the
    /// per-bin ambiguity flags for every sub. Errors on non-finite or
    /// out-of-support queries.
    pub fn interpolate_with_evidence(&self, p: [f64; D]) -> Result<InterpolatedResponse> {
        if p.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(AutoeqError::InvalidMeasurement {
                message: "ListeningArea query position must be finite".into(),
            });
        }
        if !self.contains(p) {
            let bounds = self.bounding_box();
            return Err(AutoeqError::InvalidMeasurement {
                message: format!(
                    "ListeningArea query position {p:?} is outside the calibration \
                     support {bounds:?}; use interpolate_at to extrapolate instead"
                ),
            });
        }
        let weights = self.idw_weights(p)?;
        Ok(self.interpolate_core(&weights))
    }

    /// Shared interpolation core: weighted dB mean for SPL and weighted
    /// circular (complex) mean for phase, plus confidence evidence.
    fn interpolate_core(&self, weights: &[f64]) -> InterpolatedResponse {
        let reference_freq = self.measurements[0][0].freq.clone();
        let num_bins = reference_freq.len();
        let threshold = self.config.ambiguity_threshold;

        let mut curves: Vec<Curve> = Vec::with_capacity(self.num_subs);
        let mut confidence: Vec<Array1<f64>> = Vec::with_capacity(self.num_subs);
        let mut phase_ambiguous: Vec<Array1<bool>> = Vec::with_capacity(self.num_subs);
        for sub_idx in 0..self.num_subs {
            let mut spl = Array1::<f64>::zeros(num_bins);
            let mut phase = Array1::<f64>::zeros(num_bins);
            let mut conf = Array1::<f64>::zeros(num_bins);
            let mut ambiguous = Vec::with_capacity(num_bins);

            for bin in 0..num_bins {
                // SPL: weighted mean in dB.
                let mut spl_acc = 0.0_f64;
                for (k, &w) in weights.iter().enumerate() {
                    spl_acc += w * self.measurements[sub_idx][k].spl[bin];
                }
                spl[bin] = spl_acc;

                // Phase: weighted circular mean via the precomputed unit
                // phasors. The sum is order-independent, so no
                // calibration point is privileged as an unwrap reference.
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (k, &w) in weights.iter().enumerate() {
                    re += w * self.phasor_re[sub_idx][k][bin];
                    im += w * self.phasor_im[sub_idx][k][bin];
                }
                let resultant = re.hypot(im);
                // atan2(0, 0) is defined as 0: deterministic under total
                // cancellation; the bin is flagged ambiguous below.
                phase[bin] = wrap_degrees(im.atan2(re).to_degrees());
                conf[bin] = resultant;
                ambiguous.push(resultant < threshold);
            }

            curves.push(Curve {
                freq: reference_freq.clone(),
                spl,
                phase: Some(phase),
                ..Default::default()
            });
            confidence.push(conf);
            phase_ambiguous.push(Array1::from_vec(ambiguous));
        }

        InterpolatedResponse {
            curves,
            weights: weights.to_vec(),
            confidence,
            phase_ambiguous,
        }
    }

    /// Fallible IDW weights: errors on non-finite queries and on
    /// numerically degenerate weight totals. Never falls back to uniform
    /// weights: every calibration point keeps its distance-derived share.
    fn idw_weights(&self, p: [f64; D]) -> Result<Vec<f64>> {
        if p.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(AutoeqError::InvalidMeasurement {
                message: "ListeningArea query position must be finite".into(),
            });
        }
        let weights = self.idw_weights_unchecked(p)?;
        Ok(weights)
    }

    /// Raw IDW weights for a finite query; errors only on a degenerate
    /// (non-finite or non-positive) weight total.
    fn idw_weights_unchecked(&self, p: [f64; D]) -> Result<Vec<f64>> {
        let eps = self.config.epsilon.max(0.0);
        let power = self.config.idw_power;
        let mut weights: Vec<f64> = Vec::with_capacity(self.num_positions);

        // If the query point is on a calibration point (within eps), collapse
        // to that point exactly to avoid floating-point noise.
        for pk in &self.positions {
            let mut d2 = 0.0_f64;
            for i in 0..D {
                let dx = pk[i] - p[i];
                d2 += dx * dx;
            }
            let d = d2.sqrt();
            if d <= eps {
                let mut w = vec![0.0_f64; self.num_positions];
                let idx = self
                    .positions
                    .iter()
                    .position(|q| q == pk)
                    .expect("pk is one of self.positions");
                w[idx] = 1.0;
                return Ok(w);
            }
            weights.push(1.0 / (d + eps).powf(power));
        }

        let total: f64 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(AutoeqError::InvalidMeasurement {
                message: "ListeningArea IDW weights are degenerate \
                          (non-finite or non-positive total)"
                    .into(),
            });
        }
        for w in weights.iter_mut() {
            *w /= total;
        }
        Ok(weights)
    }

    /// Deterministic last-resort weights: all mass on the nearest
    /// calibration point. Used only by the legacy infallible path when the
    /// weight total is numerically degenerate.
    fn nearest_weights(&self, p: [f64; D]) -> Vec<f64> {
        let mut best_idx = 0_usize;
        let mut best_d2 = f64::INFINITY;
        for (idx, pk) in self.positions.iter().enumerate() {
            let mut d2 = 0.0_f64;
            for i in 0..D {
                let dx = pk[i] - p[i];
                d2 += dx * dx;
            }
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = idx;
            }
        }
        let mut w = vec![0.0_f64; self.num_positions];
        w[best_idx] = 1.0;
        w
    }
}

/// Interpolated response with support/confidence evidence.
///
/// Returned by [`ListeningArea::interpolate_with_evidence`].
#[derive(Debug, Clone)]
pub struct InterpolatedResponse {
    /// One interpolated [`Curve`] per subwoofer / driver.
    pub curves: Vec<Curve>,
    /// IDW weights over the K calibration positions (sum to 1).
    pub weights: Vec<f64>,
    /// Per-sub per-bin phasor resultant magnitude `R ∈ [0, 1]`.
    ///
    /// `R = 1` at a calibration point (full agreement); `R ≈ 0` under
    /// total inter-position cancellation. Treat low values as low
    /// confidence in the reported phase; SPL is unaffected.
    pub confidence: Vec<Array1<f64>>,
    /// Per-sub per-bin ambiguity flags: true where `confidence` falls
    /// below the configured `ambiguity_threshold`. Flagged phases are the
    /// circular mean but are arbitrary under perturbation — down-weight
    /// or exclude them downstream instead of trusting the angle.
    pub phase_ambiguous: Vec<Array1<bool>>,
}

/// Wrap a phase in degrees to `[-180, 180]`.
fn wrap_degrees(phase: f64) -> f64 {
    phase - 360.0 * (phase / 360.0).round()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_curve(freq: Vec<f64>, spl: Vec<f64>, phase: Vec<f64>) -> Curve {
        Curve {
            freq: Array1::from_vec(freq),
            spl: Array1::from_vec(spl),
            phase: Some(Array1::from_vec(phase)),
            ..Default::default()
        }
    }

    #[test]
    fn interpolate_at_calibration_point_returns_calibration_curve() {
        let positions = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let curves: Vec<Curve> = (0..4)
            .map(|k| {
                make_curve(
                    vec![100.0, 1000.0],
                    vec![80.0 + k as f64, 85.0 + k as f64],
                    vec![10.0 * k as f64, 20.0 * k as f64],
                )
            })
            .collect();
        let area: ListeningArea<2> = ListeningArea::new(
            positions.clone(),
            vec![curves.clone()],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("constructible");

        for (k, p) in positions.iter().enumerate() {
            let interp = area.interpolate_at(*p);
            assert_eq!(interp.len(), 1);
            assert!(
                (interp[0].spl[0] - curves[k].spl[0]).abs() < 1e-6,
                "at {:?}: spl[0] expected {}, got {}",
                p,
                curves[k].spl[0],
                interp[0].spl[0]
            );
            assert!(
                (interp[0].spl[1] - curves[k].spl[1]).abs() < 1e-6,
                "at {:?}: spl[1] expected {}, got {}",
                p,
                curves[k].spl[1],
                interp[0].spl[1]
            );
        }
    }

    #[test]
    fn interpolate_midpoint_brackets_calibration_values() {
        // Two calibration points with SPL 70 and 90 at the same freq;
        // midpoint should give a value strictly between them under IDW.
        let positions = vec![[0.0], [1.0]];
        let curves = vec![
            make_curve(vec![100.0, 200.0], vec![70.0, 71.0], vec![0.0, 0.0]),
            make_curve(vec![100.0, 200.0], vec![90.0, 91.0], vec![0.0, 0.0]),
        ];
        let area: ListeningArea<1> = ListeningArea::new(
            positions,
            vec![curves],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("ok");
        let mid = area.interpolate_at([0.5]);
        assert!(mid[0].spl[0] > 70.0 && mid[0].spl[0] < 90.0);
        // With IDW power=2 and equal distance, midpoint is the arithmetic mean.
        assert!((mid[0].spl[0] - 80.0).abs() < 1e-6);
    }

    #[test]
    fn rejects_nonfinite_positions_and_nonpositive_idw_power() {
        let curve = make_curve(vec![100.0, 200.0], vec![80.0, 80.0], vec![0.0, 0.0]);

        let nonfinite_position = ListeningArea::<1>::new(
            vec![[f64::NAN]],
            vec![vec![curve.clone()]],
            ListeningAreaInterpolatorConfig::default(),
        )
        .unwrap_err();
        assert!(
            nonfinite_position
                .to_string()
                .contains("positions must be finite")
        );

        for idw_power in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let error = ListeningArea::<1>::new(
                vec![[0.0]],
                vec![vec![curve.clone()]],
                ListeningAreaInterpolatorConfig {
                    idw_power,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("idw_power must be finite and positive"),
                "unexpected error for idw_power={idw_power}: {error}"
            );
        }
    }

    #[test]
    fn rejects_curves_without_phase() {
        let positions = vec![[0.0]];
        let no_phase = Curve {
            freq: Array1::from_vec(vec![100.0]),
            spl: Array1::from_vec(vec![80.0]),
            phase: None,
            ..Default::default()
        };
        let err = ListeningArea::<1>::new(
            positions,
            vec![vec![no_phase]],
            ListeningAreaInterpolatorConfig::default(),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("phase"));
    }

    #[test]
    fn rejects_curves_with_mismatched_phase_length() {
        let positions = vec![[0.0]];
        let short_phase = Curve {
            freq: Array1::from_vec(vec![100.0, 1_000.0]),
            spl: Array1::from_vec(vec![80.0, 81.0]),
            phase: Some(Array1::from_vec(vec![0.0])),
            ..Default::default()
        };

        let err = ListeningArea::<1>::new(
            positions,
            vec![vec![short_phase]],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect_err("mismatched phase length must be rejected at construction");

        assert!(format!("{err}").contains("freq/phase length mismatch"));
    }

    #[test]
    fn rejects_mismatched_position_count() {
        let positions = vec![[0.0], [1.0]];
        let curves = vec![make_curve(vec![100.0], vec![80.0], vec![0.0])];
        let err = ListeningArea::<1>::new(
            positions,
            vec![curves],
            ListeningAreaInterpolatorConfig::default(),
        )
        .unwrap_err();
        assert!(format!("{err}").contains("positions"));
    }

    #[test]
    fn bounding_box_matches_extremes() {
        let positions = vec![[-1.0, 2.0], [3.0, -4.0], [0.0, 0.0]];
        let curves: Vec<Curve> = (0..3)
            .map(|_| make_curve(vec![100.0, 200.0], vec![80.0, 81.0], vec![0.0, 5.0]))
            .collect();
        let area: ListeningArea<2> = ListeningArea::new(
            positions,
            vec![curves],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("ok");
        let bb = area.bounding_box();
        assert_eq!(bb[0], (-1.0, 3.0));
        assert_eq!(bb[1], (-4.0, 2.0));
    }

    #[test]
    fn phase_interpolation_handles_wraparound() {
        // Two cal points with phases 170° and -170° at the same freq.
        // The shortest arc midpoint should be 180° (or -180°), not 0°.
        let positions = vec![[0.0], [1.0]];
        let curves = vec![
            make_curve(vec![100.0, 200.0], vec![80.0, 81.0], vec![170.0, 170.0]),
            make_curve(vec![100.0, 200.0], vec![80.0, 81.0], vec![-170.0, -170.0]),
        ];
        let area: ListeningArea<1> = ListeningArea::new(
            positions,
            vec![curves],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("ok");
        let mid = area.interpolate_at([0.5]);
        let p = mid[0].phase.as_ref().unwrap()[0];
        // Should be near ±180°, definitely not near 0°.
        assert!(p.abs() > 170.0, "expected near ±180°, got {}", p);
    }

    /// Build the P1 probe area under a calibration-point permutation.
    ///
    /// Equal-SPL curves at equal distances from the query `[0, 0]`:
    /// positions `[-1,0],[0,1],[1,0]`, phases `0/100/-160` deg (spread
    /// >180° in both bins, so the old position-0-referenced unwrap picked
    /// its branch from an arbitrary reference).
    fn probe_area(order: &[usize]) -> ListeningArea<2> {
        let positions_all = [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
        let phases_all = [0.0_f64, 100.0, -160.0];
        let positions: Vec<[f64; 2]> = order.iter().map(|&k| positions_all[k]).collect();
        let curves: Vec<Curve> = order
            .iter()
            .map(|&k| {
                make_curve(
                    vec![100.0, 1000.0],
                    vec![80.0, 82.0],
                    vec![phases_all[k], phases_all[k] + 20.0],
                )
            })
            .collect();
        ListeningArea::new(
            positions,
            vec![curves],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("probe area must construct")
    }

    fn probe_phase(area: &ListeningArea<2>) -> [f64; 2] {
        let out = area
            .try_interpolate_at([0.0, 0.0])
            .expect("query [0,0] is in support");
        let phase = out[0].phase.as_ref().expect("phase present");
        [phase[0], phase[1]]
    }

    #[test]
    fn broad_spread_phase_is_permutation_invariant() {
        // Reproduces P1: the old reference-0 unwrap returned -20° in one
        // order and 100° with the first two pairs swapped (120° shift on
        // identical physics). The complex mean must agree for every order.
        let reference = probe_phase(&probe_area(&[0, 1, 2]));
        for order in [&[1, 0, 2][..], &[2, 1, 0][..], &[0, 2, 1][..], &[2, 0, 1][..]] {
            let got = probe_phase(&probe_area(order));
            for bin in 0..2 {
                assert!(
                    (got[bin] - reference[bin]).abs() < 1e-9,
                    "order {order:?} bin {bin}: {} vs {}",
                    got[bin],
                    reference[bin]
                );
                assert!(got[bin].is_finite(), "non-finite phase for {order:?}");
            }
        }
        // Both bins carry a >180° spread; confidence must still be
        // meaningful (partial agreement, not total cancellation).
        let evidence = probe_area(&[0, 1, 2])
            .interpolate_with_evidence([0.0, 0.0])
            .expect("in support");
        for bin in 0..2 {
            let conf = evidence.confidence[0][bin];
            assert!(
                (0.0..=1.0).contains(&conf) && conf > 1e-6,
                "bin {bin}: unexpected confidence {conf}"
            );
            assert!(
                !evidence.phase_ambiguous[0][bin],
                "bin {bin}: partial agreement must not be flagged ambiguous"
            );
        }
    }

    #[test]
    fn broad_spread_invariance_holds_for_five_point_field() {
        // Five phases spanning the full circle at five 1D positions.
        let positions = vec![[-2.0], [-1.0], [0.0], [1.0], [2.0]];
        let phases = [0.0_f64, 120.0, -120.0, 45.0, -90.0];
        let build = |order: &[usize]| {
            let ordered_positions: Vec<[f64; 1]> =
                order.iter().map(|&k| positions[k]).collect();
            let curves: Vec<Curve> = order
                .iter()
                .map(|&k| {
                    make_curve(
                        vec![100.0, 200.0],
                        vec![80.0, 81.0],
                        vec![phases[k], phases[k]],
                    )
                })
                .collect();
            ListeningArea::new(
                ordered_positions,
                vec![curves],
                ListeningAreaInterpolatorConfig::default(),
            )
            .expect("constructible")
        };
        let reference = build(&[0, 1, 2, 3, 4]).interpolate_at([0.3]);
        let ref_phase = reference[0].phase.as_ref().unwrap().to_vec();
        for order in [&[4, 3, 2, 1, 0][..], &[2, 0, 4, 1, 3][..]] {
            let got = build(order).interpolate_at([0.3]);
            let got_phase = got[0].phase.as_ref().unwrap();
            for (bin, (&g, &r)) in got_phase.iter().zip(ref_phase.iter()).enumerate() {
                assert!(
                    (g - r).abs() < 1e-9,
                    "order {order:?} bin {bin}: {g} vs {r}"
                );
            }
        }
    }

    #[test]
    fn total_cancellation_flags_phase_ambiguous() {
        // Equal and opposite phasors cancel: the mean angle is arbitrary
        // and must be flagged, while SPL still interpolates.
        let positions = vec![[0.0], [1.0]];
        let curves = vec![
            make_curve(vec![100.0, 200.0], vec![80.0, 81.0], vec![0.0, 0.0]),
            make_curve(vec![100.0, 200.0], vec![80.0, 81.0], vec![180.0, 180.0]),
        ];
        let area: ListeningArea<1> = ListeningArea::new(
            positions,
            vec![curves],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("ok");
        let evidence = area
            .interpolate_with_evidence([0.5])
            .expect("midpoint is in support");
        for bin in 0..2 {
            assert!(
                evidence.confidence[0][bin] < PHASE_AMBIGUITY_THRESHOLD,
                "bin {bin}: opposing phasors must (near-)cancel, got R={}",
                evidence.confidence[0][bin]
            );
            assert!(
                evidence.phase_ambiguous[0][bin],
                "bin {bin}: cancelled phase must be flagged ambiguous"
            );
        }
        assert!((evidence.curves[0].spl[0] - 80.0).abs() < 1e-9);
    }

    #[test]
    fn identical_calibration_positions_interpolate_cleanly() {
        // Duplicate positions: exact-hit collapse picks one copy, and
        // off-point queries stay finite with weights summing to 1.
        let positions = vec![[0.0], [0.0], [1.0]];
        let curves = vec![
            make_curve(vec![100.0, 200.0], vec![80.0, 81.0], vec![10.0, 10.0]),
            make_curve(vec![100.0, 200.0], vec![80.0, 81.0], vec![10.0, 10.0]),
            make_curve(vec![100.0, 200.0], vec![90.0, 91.0], vec![20.0, 20.0]),
        ];
        let area: ListeningArea<1> = ListeningArea::new(
            positions,
            vec![curves.clone()],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("ok");
        let at_origin = area.interpolate_at([0.0]);
        assert!((at_origin[0].spl[0] - 80.0).abs() < 1e-9);
        let evidence = area
            .interpolate_with_evidence([0.5])
            .expect("in support");
        assert!((evidence.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            evidence.curves[0]
                .spl
                .iter()
                .chain(evidence.curves[0].phase.as_ref().unwrap().iter())
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn fallible_query_rejects_nonfinite_and_out_of_support() {
        let positions = vec![[0.0], [1.0]];
        let curves = vec![
            make_curve(vec![100.0, 200.0], vec![80.0, 81.0], vec![0.0, 5.0]),
            make_curve(vec![100.0, 200.0], vec![90.0, 91.0], vec![10.0, 15.0]),
        ];
        let area: ListeningArea<1> = ListeningArea::new(
            positions,
            vec![curves],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("ok");

        assert!(!area.contains([f64::NAN]));
        assert!(!area.contains([2.0]));
        assert!(area.contains([0.5]));
        // Boundary queries are in support (inclusive box).
        assert!(area.contains([0.0]));
        assert!(area.contains([1.0]));

        let nonfinite = area.try_interpolate_at([f64::NAN]).unwrap_err();
        assert!(format!("{nonfinite}").contains("finite"));

        let outside = area.try_interpolate_at([2.0]).unwrap_err();
        assert!(format!("{outside}").contains("support"));

        // In-support evidence: weights sum to 1, confidence in [0, 1].
        let evidence = area
            .interpolate_with_evidence([0.5])
            .expect("in support");
        assert!((evidence.weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        for conf in &evidence.confidence[0] {
            assert!((0.0..=1.0).contains(conf), "confidence out of range: {conf}");
        }

        // At a calibration point the evidence collapses exactly with full
        // confidence.
        let at_cal = area
            .interpolate_with_evidence([0.0])
            .expect("calibration point is in support");
        assert_eq!(at_cal.weights, vec![1.0, 0.0]);
        assert!((at_cal.confidence[0][0] - 1.0).abs() < 1e-12);
        assert!(!at_cal.phase_ambiguous[0][0]);

        // Legacy path still extrapolates finitely outside support.
        let extrapolated = area.interpolate_at([2.0]);
        assert!(
            extrapolated[0]
                .spl
                .iter()
                .chain(extrapolated[0].phase.as_ref().unwrap().iter())
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn new_rejects_nonfinite_curve_values() {
        // Curve::validate contract: non-finite SPL / phase rejected.
        for (spl, phase) in [
            (vec![80.0, f64::NAN], vec![0.0, 0.0]),
            (vec![80.0, 81.0], vec![0.0, f64::INFINITY]),
        ] {
            let bad = make_curve(vec![100.0, 200.0], spl, phase);
            let err = ListeningArea::<1>::new(
                vec![[0.0]],
                vec![vec![bad]],
                ListeningAreaInterpolatorConfig::default(),
            )
            .expect_err("non-finite curve values must be rejected");
            assert!(
                format!("{err}").contains("finite"),
                "unexpected error: {err}"
            );
        }
        // Single-bin curves violate the shared curve contract (≥2 points).
        let single = make_curve(vec![100.0], vec![80.0], vec![0.0]);
        assert!(
            ListeningArea::<1>::new(
                vec![[0.0]],
                vec![vec![single]],
                ListeningAreaInterpolatorConfig::default(),
            )
            .is_err()
        );
        // Invalid ambiguity thresholds rejected like the other config knobs.
        let good = make_curve(vec![100.0, 200.0], vec![80.0, 81.0], vec![0.0, 0.0]);
        for threshold in [f64::NAN, -1.0] {
            let err = ListeningArea::<1>::new(
                vec![[0.0]],
                vec![vec![good.clone()]],
                ListeningAreaInterpolatorConfig {
                    ambiguity_threshold: threshold,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(format!("{err}").contains("ambiguity_threshold"));
        }
    }

    /// Hold-out on a smooth spatial field with room-like physics: SPL varies
    /// linearly with position and phase advances linearly (pure delay).
    /// Symmetric IDW weights recover odd-symmetric fields exactly, so the
    /// held-out centre must match to numerical precision with ~1 confidence.
    #[test]
    fn holdout_recovers_smooth_spatial_field() {
        let freq = vec![100.0, 200.0];
        // Hold out x = 0; train on ±0.5 and ±1.
        let train = [-1.0_f64, -0.5, 0.5, 1.0];
        let positions: Vec<[f64; 1]> = train.map(|x| [x]).to_vec();
        let curves: Vec<Curve> = train
            .iter()
            .map(|&x| make_curve(freq.clone(), vec![80.0 + 3.0 * x, 82.0], vec![20.0 * x, 0.0]))
            .collect();
        let area: ListeningArea<1> = ListeningArea::new(
            positions,
            vec![curves],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("ok");
        let evidence = area
            .interpolate_with_evidence([0.0])
            .expect("held-out centre is in support");
        assert!((evidence.curves[0].spl[0] - 80.0).abs() < 1e-9);
        let phase = evidence.curves[0].phase.as_ref().unwrap()[0];
        assert!(phase.abs() < 1e-9, "expected ~0°, got {phase}");
        // ±20° calibration spread at bin 0 gives R ≈ 0.976: high but
        // honestly below 1.
        assert!(evidence.confidence[0][0] > 0.95);
        assert!(!evidence.phase_ambiguous[0][0]);
    }

    /// Hold-out on a gently curved field (standing-wave-like SPL ripple
    /// plus delay-like phase slope) from an asymmetric stencil: error must
    /// stay bounded and confidence high.
    #[test]
    fn holdout_bounds_error_on_curved_spatial_field() {
        let freq = vec![100.0, 200.0];
        let field = |x: f64| (80.0 + 0.5 * (std::f64::consts::PI * x).sin(), 10.0 * x);
        // Asymmetric stencil around the held-out x = 0.5.
        let train = [-1.0_f64, -0.5, 0.0, 1.0];
        let positions: Vec<[f64; 1]> = train.map(|x| [x]).to_vec();
        let curves: Vec<Curve> = train
            .iter()
            .map(|&x| {
                let (spl, phase) = field(x);
                make_curve(freq.clone(), vec![spl, 82.0], vec![phase, 0.0])
            })
            .collect();
        let area: ListeningArea<1> = ListeningArea::new(
            positions,
            vec![curves],
            ListeningAreaInterpolatorConfig::default(),
        )
        .expect("ok");
        let evidence = area
            .interpolate_with_evidence([0.5])
            .expect("held-out point is in support");
        let (true_spl, true_phase) = field(0.5);
        let spl_err = (evidence.curves[0].spl[0] - true_spl).abs();
        assert!(spl_err < 1.0, "SPL hold-out error too large: {spl_err}");
        let got_phase = evidence.curves[0].phase.as_ref().unwrap()[0];
        let mut phase_err = (got_phase - true_phase).abs();
        phase_err -= 360.0 * (phase_err / 360.0).round();
        assert!(phase_err.abs() < 3.0, "phase hold-out error too large: {phase_err}");
        assert!(evidence.confidence[0][0] > 0.9);
        assert!(!evidence.phase_ambiguous[0][0]);
    }
}
