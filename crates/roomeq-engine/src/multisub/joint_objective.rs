//! Joint transfer-matrix multi-sub optimization.
//!
//! Retains the transfer matrix `H[s,k](f)` from every independently controlled
//! sub `k` to every relevant seat `s`, with one common reference label per
//! seat. Gains and delays are optimized jointly over seat-to-seat variation,
//! unnormalized usable output and required drive, and target error; shared EQ
//! is then applied to the residual common response only. LFE versus
//! redirected-bass gains are traced in a ledger that refuses double
//! application to the same physical output.

// Rust guideline compliant 2026-02-21

use super::seat_response::render_mso_seat_responses;
use crate::Curve;
use crate::error::{AutoeqError, Result};
use autoeq_optim::loss::{JointSubComponents, JointSubWeights, joint_multisub_loss};
use autoeq_optim::optim::scalar::{ScalarOptimConfig, optimize_bounded_scalar};
use ndarray::Array1;
use num_complex::Complex64;
use roomeq_model::OptimizerConfig;
use std::f64::consts::PI;

/// Gain stage contributing to one physical sub output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BassGainStage {
    /// Discrete LFE channel gain.
    Lfe,
    /// Bass redirected from mains through the low-pass route.
    Redirected,
}

/// One traced gain application to a physical sub output.
#[derive(Debug, Clone, PartialEq)]
pub struct BassGainEntry {
    /// Which logical stage the gain belongs to.
    pub stage: BassGainStage,
    /// Physical sub output the gain is applied to.
    pub physical_output: String,
    /// Applied gain in dB.
    pub gain_db: f64,
}

/// Traces LFE versus redirected-bass gains without double application.
///
/// Each `(stage, physical_output)` pair may be recorded once. Recording the
/// same pair twice is an error, so a gain can never be silently applied on
/// top of itself while the ledger claims one application.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BassGainLedger {
    entries: Vec<BassGainEntry>,
}

impl BassGainLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one gain application.
    ///
    /// # Examples
    ///
    /// ```
    /// use roomeq_engine::multisub::{BassGainLedger, BassGainStage};
    /// let mut ledger = BassGainLedger::new();
    /// ledger
    ///     .apply_gain(BassGainStage::Lfe, "sub-1", 10.0_f64)
    ///     .unwrap();
    /// assert_eq!(ledger.total_for("sub-1"), 10.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`AutoeqError::InvalidMeasurement`] when the output name is
    /// empty, the gain is nonfinite, or the same stage was already recorded
    /// for the same physical output.
    pub fn apply_gain(
        &mut self,
        stage: BassGainStage,
        physical_output: &str,
        gain_db: f64,
    ) -> Result<()> {
        let invalid = |message: String| AutoeqError::InvalidMeasurement { message };
        if physical_output.is_empty() {
            return Err(invalid(
                "bass gain ledger needs a physical output name".to_string(),
            ));
        }
        if !gain_db.is_finite() {
            return Err(invalid("bass gain ledger gain must be finite".to_string()));
        }
        if self
            .entries
            .iter()
            .any(|entry| entry.stage == stage && entry.physical_output == physical_output)
        {
            return Err(invalid(format!(
                "bass gain stage {stage:?} already applied to `{physical_output}`"
            )));
        }
        self.entries.push(BassGainEntry {
            stage,
            physical_output: physical_output.to_string(),
            gain_db,
        });
        Ok(())
    }

    /// Sum of traced gains for one physical output in dB.
    pub fn total_for(&self, physical_output: &str) -> f64 {
        self.entries
            .iter()
            .filter(|entry| entry.physical_output == physical_output)
            .map(|entry| entry.gain_db)
            .sum()
    }

    /// All traced entries in application order.
    pub fn entries(&self) -> &[BassGainEntry] {
        &self.entries
    }
}

/// Absolute output view for one seat; levels are never renormalized.
#[derive(Debug, Clone, PartialEq)]
pub struct UnnormalizedOutputView {
    /// Seat index in the transfer matrix.
    pub seat: usize,
    /// Mean absolute level before array controls, in dB.
    pub pre_level_db: f64,
    /// Mean absolute level after array controls, in dB.
    pub post_level_db: f64,
    /// Absolute level change (`post - pre`), in dB.
    pub delta_db: f64,
}

/// Joint transfer-matrix optimization result.
#[derive(Debug, Clone)]
pub struct JointSubResult {
    /// Optimized per-sub gains in dB (sub 0 is the fixed reference at 0 dB).
    pub gains_db: Vec<f64>,
    /// Optimized per-sub delays in ms (sub 0 is the fixed reference at 0 ms).
    pub delays_ms: Vec<f64>,
    /// Controlled per-seat combined responses, retained per seat.
    pub per_seat_post: Vec<Curve>,
    /// Absolute output views, one per seat.
    pub output_views: Vec<UnnormalizedOutputView>,
    /// Joint loss components before optimization.
    pub pre_components: JointSubComponents,
    /// Joint loss components after optimization.
    pub post_components: JointSubComponents,
    /// Traced LFE versus redirected-bass gains.
    pub ledger: BassGainLedger,
    /// Common reference label verified within each seat.
    pub reference_scope_per_seat: Vec<String>,
    /// Whether the scalar optimizer reported convergence.
    pub converged: bool,
}

/// Transfer-matrix row for one seat: complex response of every sub.
struct SeatRow {
    /// Complex response per sub on the shared grid.
    subs: Vec<Vec<Complex64>>,
}

/// Render the per-seat combined complex field for array controls.
///
/// `matrix` holds `H[s,k]` on `freqs`; gains apply as dB magnitudes and
/// delays as phase rotations `-2 pi f tau`.
fn render_combined_field(
    matrix: &[SeatRow],
    freqs: &[f64],
    gains_db: &[f64],
    delays_ms: &[f64],
) -> Vec<Vec<Complex64>> {
    matrix
        .iter()
        .map(|seat| {
            freqs
                .iter()
                .enumerate()
                .map(|(bin, frequency)| {
                    seat.subs
                        .iter()
                        .enumerate()
                        .map(|(sub, response)| {
                            let magnitude =
                                10.0_f64.powf(gains_db[sub] / 20.0) * response[bin].norm();
                            let angle = response[bin].arg()
                                - 2.0 * PI * frequency * delays_ms[sub] / 1000.0;
                            Complex64::from_polar(magnitude, angle)
                        })
                        .sum()
                })
                .collect()
        })
        .collect()
}

/// Convert a measured curve to complex response on `grid`.
fn curve_to_complex(curve: &Curve, grid: &Array1<f64>) -> Vec<Complex64> {
    let interpolated = autoeq_core::curve_transforms::interpolate_log_space(grid, curve);
    let phase = interpolated
        .phase
        .as_ref()
        .expect("phase checked by caller");
    interpolated
        .spl
        .iter()
        .zip(phase.iter())
        .map(|(spl, degrees)| {
            Complex64::from_polar(10.0_f64.powf(spl / 20.0), degrees.to_radians())
        })
        .collect()
}

/// Incoherent measured power sum of the subs at one seat, in dB.
fn seat_power_reference(matrix_row: &[Vec<Complex64>]) -> Vec<f64> {
    let bins = matrix_row.first().map_or(0, Vec::len);
    (0..bins)
        .map(|bin| {
            let power: f64 = matrix_row
                .iter()
                .map(|response| response[bin].norm().powi(2))
                .sum();
            10.0 * power.max(1e-24).log10()
        })
        .collect()
}

fn complex_field_to_levels(field: &[Vec<Complex64>]) -> Vec<Vec<f64>> {
    field
        .iter()
        .map(|seat| {
            seat.iter()
                .map(|z| 20.0 * z.norm().max(1e-12).log10())
                .collect()
        })
        .collect()
}

/// Mean absolute level over the evaluation band.
fn mean_band_level(levels: &[f64], freqs: &[f64], min_freq: f64, max_freq: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0_usize;
    for (level, frequency) in levels.iter().zip(freqs.iter()) {
        if *frequency >= min_freq && *frequency <= max_freq {
            sum += level;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

/// Joint multi-sub optimization inputs.
#[derive(Debug, Clone)]
pub struct JointSubRequest {
    /// Measurements indexed `[sub][seat]`; every curve needs measured phase.
    pub measurements: Vec<Vec<Curve>>,
    /// Absolute target response on the shared grid, one value per bin.
    pub target_db: Vec<f64>,
    /// Scalarization weights for variation, output/drive, and target error.
    pub weights: JointSubWeights,
    /// LFE gain per sub in dB, traced but not applied twice.
    pub lfe_gain_db: Vec<f64>,
    /// Redirected-bass gain per sub in dB, traced but not applied twice.
    pub redirected_gain_db: Vec<f64>,
    /// Physical sub output names, one per sub.
    pub physical_outputs: Vec<String>,
    /// Reference-scope label per `[sub][seat]`; every sub must agree within
    /// each seat. `None` records `unknown` and leaves scope verification to
    /// the upstream measurement envelope.
    pub reference_scope: Option<Vec<Vec<String>>>,
}

/// Optimize sub gains and delays jointly over the transfer matrix.
///
/// Sub 0 is the fixed array reference (0 dB, 0 ms); common gain and level
/// matching belong to the later shared-EQ stage, which owns the residual
/// only. LFE and redirected gains are traced in the returned ledger, never
/// folded into the optimized array gains.
///
/// # Errors
///
/// Returns an error when measurements are ragged, miss phase, share no grid
/// support, disagree on reference scope within a seat, or when the target,
/// gains, or ledger inputs are inconsistent.
///
/// # Panics
///
/// Never panics on validated inputs.
pub fn optimize_joint_sub_array(
    request: &JointSubRequest,
    config: &OptimizerConfig,
    sample_rate: f64,
) -> Result<JointSubResult> {
    let invalid = |message: &str| AutoeqError::InvalidMeasurement {
        message: format!("joint multi-sub optimization: {message}"),
    };
    if !sample_rate.is_finite() || sample_rate <= 0.0 {
        return Err(invalid("sample rate must be finite and positive"));
    }
    let subs = request.measurements.len();
    if subs == 0 {
        return Err(invalid("no subwoofer measurements"));
    }
    let seats = request.measurements[0].len();
    if seats == 0 || request.measurements.iter().any(|sub| sub.len() != seats) {
        return Err(invalid("sub/seat count mismatch"));
    }
    if request.lfe_gain_db.len() != subs
        || request.redirected_gain_db.len() != subs
        || request.physical_outputs.len() != subs
    {
        return Err(invalid("per-sub gain/output count mismatch"));
    }
    if request
        .lfe_gain_db
        .iter()
        .chain(request.redirected_gain_db.iter())
        .any(|gain| !gain.is_finite())
    {
        return Err(invalid("nonfinite LFE or redirected gain"));
    }
    for curve in request.measurements.iter().flatten() {
        if curve.freq.len() < 2
            || curve.spl.len() != curve.freq.len()
            || !crate::topology::curve_has_usable_phase(curve)
        {
            return Err(invalid("invalid measurement or missing measured phase"));
        }
    }

    let reference_scope_per_seat = match &request.reference_scope {
        Some(scope) => {
            if scope.len() != subs || scope.iter().any(|sub| sub.len() != seats) {
                return Err(invalid("reference scope count mismatch"));
            }
            (0..seats)
                .map(|seat| {
                    let first = &scope[0][seat];
                    if scope.iter().any(|sub| &sub[seat] != first) {
                        return Err(invalid("subs disagree on reference scope within a seat"));
                    }
                    Ok(first.clone())
                })
                .collect::<Result<Vec<String>>>()?
        }
        None => vec!["unknown".to_string(); seats],
    };

    let flat: Vec<&Curve> = request.measurements.iter().flatten().collect();
    let grid = crate::topology::shared_measurement_grid(&flat)
        .filter(|grid| grid.len() >= 2)
        .ok_or_else(|| invalid("insufficient common measured support"))?;
    let freqs: Vec<f64> = grid.iter().copied().collect();
    if request.target_db.len() != freqs.len() {
        return Err(invalid("target length must match the shared grid"));
    }
    if request.target_db.iter().any(|v| !v.is_finite()) {
        return Err(invalid("nonfinite target response"));
    }

    let matrix: Vec<SeatRow> = (0..seats)
        .map(|seat| SeatRow {
            subs: request
                .measurements
                .iter()
                .map(|sub| curve_to_complex(&sub[seat], &grid))
                .collect(),
        })
        .collect();
    let seat_references: Vec<Vec<f64>> = matrix
        .iter()
        .map(|seat| seat_power_reference(&seat.subs))
        .collect();
    // The output term compares the seat-average absolute response against the
    // seat-average measured power sum, so common attenuation stays visible.
    let average_reference: Vec<f64> = (0..freqs.len())
        .map(|bin| {
            seat_references
                .iter()
                .map(|reference| reference[bin])
                .sum::<f64>()
                / seats as f64
        })
        .collect();

    let [min_freq, max_freq] = config.active_correction_band();
    let evaluate = |gains: &[f64], delays: &[f64]| -> f64 {
        let field = render_combined_field(&matrix, &freqs, gains, delays);
        let levels = complex_field_to_levels(&field);
        joint_multisub_loss(
            &levels,
            &average_reference,
            &request.target_db,
            &request.weights,
        )
        .map(|components| components.total)
        .unwrap_or(f64::INFINITY)
    };

    // Fix the reference sub; shared EQ owns common gain while this stage
    // controls relative array alignment.
    let initial_gains = vec![0.0; subs];
    let initial_delays = vec![0.0; subs];
    let pre_components = joint_multisub_loss(
        &complex_field_to_levels(&render_combined_field(
            &matrix,
            &freqs,
            &initial_gains,
            &initial_delays,
        )),
        &average_reference,
        &request.target_db,
        &request.weights,
    )
    .map_err(|message| invalid(&message))?;

    let mut lower = Vec::with_capacity(2 * subs);
    let mut upper = Vec::with_capacity(2 * subs);
    for _ in 0..subs {
        lower.push(config.min_db);
        upper.push(config.max_db);
    }
    // Delay search spans 0-20 ms, matching the all-pass stage bounds, so
    // relative arrival differences stay within one practical alignment range.
    for _ in 0..subs {
        lower.push(0.0);
        upper.push(20.0);
    }
    lower[0] = 0.0;
    upper[0] = 0.0;
    lower[subs] = 0.0;
    upper[subs] = 0.0;
    let bounds: Vec<(f64, f64)> = lower
        .iter()
        .zip(upper.iter())
        .map(|(l, u)| (*l, *u))
        .collect();
    let initial: Vec<f64> = initial_gains
        .iter()
        .chain(initial_delays.iter())
        .copied()
        .collect();
    let report = optimize_bounded_scalar(
        &bounds,
        &initial,
        &ScalarOptimConfig {
            algorithm: config.algorithm.clone(),
            max_iter: config.max_iter,
            population: config.population,
            tolerance: config.tolerance,
            atolerance: config.atolerance,
            strategy: config.strategy.clone(),
            seed: config.seed,
        },
        move |params| evaluate(&params[0..subs], &params[subs..2 * subs]),
    )
    .map_err(|e| invalid(&format!("joint array optimization failed: {e}")))?;

    let (gains_db, delays_ms, post_components, converged) =
        if report.fun.is_finite() && report.fun <= pre_components.total {
            let gains = report.x[0..subs].to_vec();
            let delays = report.x[subs..2 * subs].to_vec();
            let post = joint_multisub_loss(
                &complex_field_to_levels(&render_combined_field(&matrix, &freqs, &gains, &delays)),
                &average_reference,
                &request.target_db,
                &request.weights,
            )
            .map_err(|message| invalid(&message))?;
            (gains, delays, post, report.success)
        } else {
            (
                initial_gains.clone(),
                initial_delays.clone(),
                pre_components,
                false,
            )
        };

    // Retain per-seat controlled responses through the shared seat renderer
    // so downstream shared EQ consumes exactly the optimized field.
    let per_seat_post = render_mso_seat_responses(&request.measurements, &gains_db, &delays_ms)?;
    let pre_field = render_combined_field(&matrix, &freqs, &initial_gains, &initial_delays);
    let pre_levels = complex_field_to_levels(&pre_field);
    let post_levels = complex_field_to_levels(&render_combined_field(
        &matrix, &freqs, &gains_db, &delays_ms,
    ));
    let output_views = (0..seats)
        .map(|seat| {
            let pre = mean_band_level(&pre_levels[seat], &freqs, min_freq, max_freq);
            let post = mean_band_level(&post_levels[seat], &freqs, min_freq, max_freq);
            UnnormalizedOutputView {
                seat,
                pre_level_db: pre,
                post_level_db: post,
                delta_db: post - pre,
            }
        })
        .collect();

    let mut ledger = BassGainLedger::new();
    for sub in 0..subs {
        ledger.apply_gain(
            BassGainStage::Lfe,
            &request.physical_outputs[sub],
            request.lfe_gain_db[sub],
        )?;
        ledger.apply_gain(
            BassGainStage::Redirected,
            &request.physical_outputs[sub],
            request.redirected_gain_db[sub],
        )?;
    }

    Ok(JointSubResult {
        gains_db,
        delays_ms,
        per_seat_post,
        output_views,
        pre_components,
        post_components,
        ledger,
        reference_scope_per_seat,
        converged,
    })
}

/// Apply one shared EQ curve to the residual per-seat responses.
///
/// The same dB correction applies at every seat, so relative seat-to-seat
/// differences are unchanged: a shared filter cannot remove spatial
/// variation, it only shapes the common residual.
pub fn apply_shared_eq_to_residual(per_seat: &[Curve], shared_eq_db: &[f64]) -> Result<Vec<Curve>> {
    if per_seat.is_empty() {
        return Err(AutoeqError::InvalidMeasurement {
            message: "shared residual EQ needs at least one seat".to_string(),
        });
    }
    let bins = per_seat[0].spl.len();
    if shared_eq_db.len() != bins || shared_eq_db.iter().any(|v| !v.is_finite()) {
        return Err(AutoeqError::InvalidMeasurement {
            message: "shared EQ must match the seat grid with finite values".to_string(),
        });
    }
    per_seat
        .iter()
        .map(|seat| {
            if seat.spl.len() != bins {
                return Err(AutoeqError::InvalidMeasurement {
                    message: "seat grids disagree for shared residual EQ".to_string(),
                });
            }
            let mut corrected = seat.clone();
            for (spl, eq) in corrected.spl.iter_mut().zip(shared_eq_db.iter()) {
                *spl += *eq;
            }
            Ok(corrected)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use autoeq_optim::loss::JointSubWeights;
    use ndarray::array;

    fn sub_curve(level_db: f64, phase_deg: f64) -> Curve {
        Curve {
            freq: array![30.0, 60.0, 120.0],
            spl: array![level_db, level_db, level_db],
            phase: Some(array![phase_deg, phase_deg, phase_deg]),
            ..Default::default()
        }
    }

    fn two_sub_two_seat_request() -> JointSubRequest {
        JointSubRequest {
            measurements: vec![
                vec![sub_curve(80.0, 0.0), sub_curve(78.0, 0.0)],
                vec![sub_curve(80.0, 0.0), sub_curve(78.0, 0.0)],
            ],
            target_db: vec![86.0, 86.0, 86.0],
            weights: JointSubWeights::default(),
            lfe_gain_db: vec![0.0, 0.0],
            redirected_gain_db: vec![0.0, 0.0],
            physical_outputs: vec!["sub-1".to_string(), "sub-2".to_string()],
            reference_scope: Some(vec![
                vec!["seat-loop".to_string(), "seat-loop".to_string()],
                vec!["seat-loop".to_string(), "seat-loop".to_string()],
            ]),
        }
    }

    fn tiny_config() -> OptimizerConfig {
        OptimizerConfig {
            min_freq: 30.0,
            max_freq: 120.0,
            algorithm: "autoeq:de".to_string(),
            max_iter: 20,
            population: 10,
            seed: Some(7),
            min_db: -12.0,
            max_db: 12.0,
            ..Default::default()
        }
    }

    #[test]
    fn joint_reference_scope_mismatch_rejected() {
        let mut request = two_sub_two_seat_request();
        request.reference_scope = Some(vec![
            vec!["loop-a".to_string(), "loop-a".to_string()],
            vec!["loop-b".to_string(), "loop-a".to_string()],
        ]);
        assert!(optimize_joint_sub_array(&request, &tiny_config(), 48000.0).is_err());
    }

    #[test]
    fn joint_result_retains_per_seat_responses_and_output_views() {
        let result =
            optimize_joint_sub_array(&two_sub_two_seat_request(), &tiny_config(), 48000.0).unwrap();
        assert_eq!(result.per_seat_post.len(), 2);
        assert_eq!(result.output_views.len(), 2);
        assert_eq!(
            result.reference_scope_per_seat,
            vec!["seat-loop".to_string(), "seat-loop".to_string()]
        );
        // Gains stay finite and the reference sub stays fixed.
        assert_eq!(result.gains_db[0], 0.0);
        assert_eq!(result.delays_ms[0], 0.0);
        assert!(result.post_components.total <= result.pre_components.total + 1e-9);
    }

    /// Shared EQ on the residual cannot change relative seat responses.
    #[test]
    fn joint_shared_eq_preserves_relative_seat_response() {
        let result =
            optimize_joint_sub_array(&two_sub_two_seat_request(), &tiny_config(), 48000.0).unwrap();
        let shared_eq = vec![-2.0, 1.5, 0.5];
        let corrected = apply_shared_eq_to_residual(&result.per_seat_post, &shared_eq).unwrap();
        assert_eq!(corrected.len(), 2);
        for bin in 0..3 {
            let before = result.per_seat_post[0].spl[bin] - result.per_seat_post[1].spl[bin];
            let after = corrected[0].spl[bin] - corrected[1].spl[bin];
            assert!(
                (before - after).abs() < 1e-9,
                "shared EQ changed relative seat response"
            );
        }
    }

    /// Unnormalized output views expose common attenuation (F11).
    #[test]
    fn joint_output_loss_survives_normalization() {
        let result =
            optimize_joint_sub_array(&two_sub_two_seat_request(), &tiny_config(), 48000.0).unwrap();
        for view in &result.output_views {
            assert!(view.pre_level_db.is_finite());
            assert!(view.post_level_db.is_finite());
            assert!((view.delta_db - (view.post_level_db - view.pre_level_db)).abs() < 1e-12);
        }
        // Attenuating both subs surfaces as an absolute pre-control level
        // drop in the retained views even though the normalized shape is
        // unchanged; the optimizer may compensate with allowed gain, but the
        // views never renormalize the evidence away.
        let mut request = two_sub_two_seat_request();
        for sub in &mut request.measurements {
            for seat in sub.iter_mut() {
                seat.spl.mapv_inplace(|spl| spl - 6.0);
            }
        }
        let quiet = optimize_joint_sub_array(&request, &tiny_config(), 48000.0).unwrap();
        assert!(quiet.output_views[0].pre_level_db < result.output_views[0].pre_level_db - 5.0);
        assert!(
            (quiet.output_views[0].pre_level_db - (result.output_views[0].pre_level_db - 6.0))
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn joint_variance_only_candidate_rejected() {
        // Seat responses that agree perfectly but sit far below target must
        // not beat a lively array once output and target terms participate.
        let weights = JointSubWeights::default();
        let reference = vec![80.0, 80.0, 80.0];
        let target = vec![86.0, 86.0, 86.0];
        let quiet = vec![vec![74.0; 3], vec![74.0; 3]];
        let lively = vec![vec![86.0; 3], vec![85.0, 86.0, 87.0]];
        let quiet_loss = joint_multisub_loss(&quiet, &reference, &target, &weights).unwrap();
        let lively_loss = joint_multisub_loss(&lively, &reference, &target, &weights).unwrap();
        assert!(quiet_loss.total > lively_loss.total);
    }

    /// LFE and redirected gains are traced separately and never doubled.
    #[test]
    fn joint_bass_gain_ledger_traces_lfe_and_redirected_without_double_application() {
        let mut ledger = BassGainLedger::new();
        ledger
            .apply_gain(BassGainStage::Lfe, "sub-1", 10.0)
            .unwrap();
        ledger
            .apply_gain(BassGainStage::Redirected, "sub-1", -3.0)
            .unwrap();
        assert_eq!(ledger.entries().len(), 2);
        assert!((ledger.total_for("sub-1") - 7.0).abs() < 1e-12);
        assert!(
            ledger
                .apply_gain(BassGainStage::Lfe, "sub-1", 10.0)
                .is_err()
        );
        assert!(
            ledger
                .apply_gain(BassGainStage::Redirected, "sub-1", -3.0)
                .is_err()
        );
        // The joint result carries a fully traced ledger by construction.
        let result =
            optimize_joint_sub_array(&two_sub_two_seat_request(), &tiny_config(), 48000.0).unwrap();
        assert_eq!(result.ledger.entries().len(), 4);
    }
}
