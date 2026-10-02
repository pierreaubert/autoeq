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
    /// Why the proposed array was rejected; absent is not final-chain approval.
    pub array_rejection_reason: Option<String>,
    /// Physical output identities in the same order as array controls.
    pub physical_outputs: Vec<String>,
    /// Uncontrolled combined response for every retained seat.
    pub per_seat_pre: Vec<Curve>,
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
    /// Whether every seat carries an explicit (non-`unknown`) verified
    /// scope. An `unknown` scope still optimizes the coherent field, but
    /// the result must not be presented as a validated coherent outcome:
    /// production dispatch refuses it before any array search runs.
    pub coherent_scope_validated: bool,
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
    /// each seat. `None` records `unknown` and marks the result
    /// [`JointSubResult::coherent_scope_validated`] false: library callers
    /// see an explicitly unvalidated coherent field, and production
    /// dispatch refuses it before any array search runs.
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
    let mut unique_outputs = std::collections::BTreeSet::new();
    if request
        .physical_outputs
        .iter()
        .any(|output| output.trim().is_empty() || !unique_outputs.insert(output))
    {
        return Err(invalid(
            "physical output identities must be nonempty and unique",
        ));
    }
    for curve in request.measurements.iter().flatten() {
        curve.validate("joint multi-sub measurement")?;
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
    if !min_freq.is_finite() || !max_freq.is_finite() || min_freq <= 0.0 || min_freq >= max_freq {
        return Err(invalid(
            "correction band must be finite, positive, and increasing",
        ));
    }
    let objective_bins: Vec<_> = freqs
        .iter()
        .enumerate()
        .filter_map(|(index, frequency)| {
            (*frequency >= min_freq && *frequency <= max_freq).then_some(index)
        })
        .collect();
    if objective_bins.len() < 2 {
        return Err(invalid(
            "correction band needs at least two shared measurement bins",
        ));
    }
    // Restrict the search, not the retained measurement or full-response safety
    // check. Gain/delay controls still affect the physical response outside it.
    let objective_freqs: Vec<_> = objective_bins.iter().map(|&index| freqs[index]).collect();
    let objective_target: Vec<_> = objective_bins
        .iter()
        .map(|&index| request.target_db[index])
        .collect();
    let objective_reference: Vec<_> = objective_bins
        .iter()
        .map(|&index| average_reference[index])
        .collect();
    let objective_matrix: Vec<_> = matrix
        .iter()
        .map(|seat| SeatRow {
            subs: seat
                .subs
                .iter()
                .map(|sub| objective_bins.iter().map(|&index| sub[index]).collect())
                .collect(),
        })
        .collect();
    let evaluate_components = |gains: &[f64], delays: &[f64]| {
        let field = render_combined_field(&objective_matrix, &objective_freqs, gains, delays);
        let levels = complex_field_to_levels(&field);
        joint_multisub_loss(
            &levels,
            &objective_reference,
            &objective_target,
            &request.weights,
        )
    };
    let evaluate = |gains: &[f64], delays: &[f64]| -> f64 {
        evaluate_components(gains, delays)
            .map(|components| components.total)
            .unwrap_or(f64::INFINITY)
    };

    // Fix the reference sub; shared EQ owns common gain while this stage
    // controls relative array alignment.
    let initial_gains = vec![0.0; subs];
    let initial_delays = vec![0.0; subs];
    let pre_components = evaluate_components(&initial_gains, &initial_delays)
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

    let (mut gains_db, mut delays_ms, mut post_components, converged) =
        if report.fun.is_finite() && report.fun <= pre_components.total {
            let gains = report.x[0..subs].to_vec();
            let delays = report.x[subs..2 * subs].to_vec();
            let post = evaluate_components(&gains, &delays).map_err(|message| invalid(&message))?;
            (gains, delays, post, report.success)
        } else {
            (
                initial_gains.clone(),
                initial_delays.clone(),
                pre_components,
                false,
            )
        };
    // Acceptance-side budget record: the emitted array gains/delays must
    // honor the search budgets on the accepted vector.
    let accepted: Vec<f64> = gains_db.iter().chain(delays_ms.iter()).copied().collect();
    autoeq_optim::optim::verify_joint_budgets("joint-sub-array", &accepted, &lower, &upper)
        .map_err(|reason| {
            invalid(&format!(
                "joint sub candidate refused at emission: {reason}"
            ))
        })?;

    // Retain per-seat controlled responses through the shared seat renderer
    // so downstream shared EQ consumes exactly the optimized field.
    let per_seat_pre =
        render_mso_seat_responses(&request.measurements, &initial_gains, &initial_delays)?;
    let mut per_seat_post =
        render_mso_seat_responses(&request.measurements, &gains_db, &delays_ms)?;
    // Reuse the runtime quality policy: every retained seat must preserve
    // target-weighted RMS within its numerical epsilon. This is an array-stage
    // engineering guard, not an audibility threshold or final-route assessment.
    let target = Curve {
        freq: grid.clone(),
        spl: request.target_db.clone().into(),
        phase: None,
        ..Default::default()
    };
    let assessment = roomeq_quality::evaluate_multi_seat_acceptance(
        &per_seat_pre,
        &per_seat_post,
        &[],
        &[],
        &target,
    );
    let array_rejection_reason = match assessment {
        Ok(assessment) if assessment.accepted() => None,
        Ok(assessment) => Some(format!(
            "joint array reverted: protected-seat runtime acceptance failed: {:?}",
            assessment
                .training
                .seats
                .iter()
                .filter(|seat| !seat.accepted)
                .collect::<Vec<_>>()
        )),
        Err(reason) => Some(format!(
            "joint array reverted: seat acceptance unavailable: {reason}"
        )),
    };
    if let Some(reason) = &array_rejection_reason {
        log::warn!("{reason}");
        gains_db = initial_gains.clone();
        delays_ms = initial_delays.clone();
        post_components = pre_components;
        per_seat_post = per_seat_pre.clone();
    }
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

    let coherent_scope_validated =
        super::types::timing_scopes_are_shared(request.reference_scope.as_deref(), subs, seats);
    Ok(JointSubResult {
        array_rejection_reason,
        physical_outputs: request.physical_outputs.clone(),
        per_seat_pre,
        gains_db,
        delays_ms,
        per_seat_post,
        output_views,
        pre_components,
        post_components,
        ledger,
        reference_scope_per_seat,
        coherent_scope_validated,
        converged,
    })
}

/// Apply one shared EQ curve to the residual per-seat responses.
///
/// The same dB correction applies at every seat, so relative seat-to-seat
/// differences are unchanged: a shared filter cannot remove spatial
/// variation, it only shapes the common residual. Every seat must have the
/// same frequency grid; callers must explicitly resample differing grids.
///
/// # Errors
///
/// Rejects malformed curves, mismatched grids, nonfinite EQ, and overflow.
pub fn apply_shared_eq_to_residual(per_seat: &[Curve], shared_eq_db: &[f64]) -> Result<Vec<Curve>> {
    if per_seat.is_empty() {
        return Err(AutoeqError::InvalidMeasurement {
            message: "shared residual EQ needs at least one seat".to_string(),
        });
    }
    let grid = &per_seat[0].freq;
    let bins = grid.len();
    if shared_eq_db.len() != bins || shared_eq_db.iter().any(|v| !v.is_finite()) {
        return Err(AutoeqError::InvalidMeasurement {
            message: "shared EQ must match the seat grid with finite values".to_string(),
        });
    }
    per_seat
        .iter()
        .map(|seat| {
            seat.validate("shared residual EQ seat")?;
            if &seat.freq != grid {
                return Err(AutoeqError::InvalidMeasurement {
                    message:
                        "seat grids disagree for shared residual EQ; resample explicitly first"
                            .to_string(),
                });
            }
            let mut corrected = seat.clone();
            for (spl, eq) in corrected.spl.iter_mut().zip(shared_eq_db.iter()) {
                *spl += *eq;
                if !spl.is_finite() {
                    return Err(AutoeqError::InvalidMeasurement {
                        message: "shared residual EQ overflowed a seat level".to_string(),
                    });
                }
            }
            Ok(corrected)
        })
        .collect()
}

/// Selected-joint-mode dispatch: build the transfer-matrix request from
/// prepared per-seat measurements, run the joint optimization, and emit
/// the array controls plus the combined response the shared residual EQ
/// stage consumes.
///
/// The absolute target comes from the configured neutral tilt curve;
/// without one the stage preserves the pre-optimization level and the
/// downstream shared EQ owns absolutes, so this stage never substitutes
/// its own target. Scalarization weights are equal across the retained
/// components. LFE and redirected gains carry no information at this
/// layer and are traced as zeros, never applied twice. Timing scope is
/// left unknown for the upstream measurement envelope to verify; an
/// unknown scope is recorded, never upgraded to validated.
///
/// # Errors
///
/// Returns an error for a ragged seat matrix, missing phase on any
/// sub/seat curve, disagreeing grids, or an inconsistent
/// target/weight/ledger input. A missing seat matrix is the caller's
/// fallback decision, not an error here.
/// Production joint multi-sub dispatch over the transfer matrix.
///
/// `reference_scope` carries the verified shared-timing labels per
/// `[sub][seat]`; every sub must agree within each seat. An unknown
/// (`None`) scope is refused before any array search runs: phase arrays
/// alone never authorize a validated coherent result. The caller falls
/// back to the detailed mode loudly instead of silently replacing it.
pub fn process_joint_sub_group(
    channel_name: &str,
    group: &roomeq_model::MultiSubGroup,
    room_config: &roomeq_model::RoomConfig,
    sample_rate: f64,
    prepared: &crate::group_processing::PreparedMultiSubGroup,
    reference_scope: Option<Vec<Vec<String>>>,
) -> Result<(
    autoeq_optim::DriverOptimizationResult,
    super::types::MultiSubCombinedResponse,
)> {
    process_joint_sub_group_detailed(
        channel_name,
        group,
        room_config,
        sample_rate,
        prepared,
        reference_scope,
    )
    .map(|(base, combined, _)| (base, combined))
}

/// Run joint dispatch while retaining the complete stage diagnostics.
///
/// # Errors
///
/// Returns the same measurement, timing, and optimization errors as joint dispatch.
pub(crate) fn process_joint_sub_group_detailed(
    channel_name: &str,
    group: &roomeq_model::MultiSubGroup,
    room_config: &roomeq_model::RoomConfig,
    sample_rate: f64,
    prepared: &crate::group_processing::PreparedMultiSubGroup,
    reference_scope: Option<Vec<Vec<String>>>,
) -> Result<(
    autoeq_optim::DriverOptimizationResult,
    super::types::MultiSubCombinedResponse,
    JointSubResult,
)> {
    let invalid = |message: &str| AutoeqError::InvalidMeasurement {
        message: format!("joint multi-sub dispatch for '{channel_name}': {message}"),
    };
    let reference_scope = reference_scope.ok_or_else(|| AutoeqError::InvalidConfiguration {
        message: format!(
            "joint multi-sub dispatch for '{channel_name}' needs a verified shared timing reference \
             within each seat; scope is unknown, so phase arrays alone cannot authorize coherent \
             optimization — select the detailed mode until scoped measurements are provided"
        ),
    })?;
    let seats = prepared.seat_measurements.as_ref().ok_or_else(|| {
        invalid("no per-seat measurement matrix prepared; select the detailed mode instead")
    })?;
    if seats.is_empty() || prepared.subwoofers.is_empty() {
        return Err(invalid("seat matrix or subwoofer list is empty"));
    }
    let sub_count = prepared.subwoofers.len();
    let seat_count = seats[0].len();
    if !super::types::timing_scopes_are_shared(Some(&reference_scope), sub_count, seat_count) {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "joint multi-sub dispatch for '{channel_name}' requires a nonempty matching timing reference at every source and seat"
            ),
        });
    }
    if seats.len() != sub_count || seat_count == 0 {
        return Err(invalid(
            "transfer matrix must have one nonempty row per subwoofer",
        ));
    }
    for (sub_index, seat) in seats.iter().enumerate() {
        if seat.len() != seat_count {
            return Err(invalid(&format!(
                "sub {sub_index} holds {} seat curves, expected {seat_count}",
                seat.len()
            )));
        }
        for (seat_index, curve) in seat.iter().enumerate() {
            match curve.phase.as_ref() {
                Some(phase) if !phase.is_empty() && phase.len() == curve.freq.len() => {}
                _ => {
                    return Err(invalid(&format!(
                        "coherent joint optimization needs measured phase on sub {sub_index} at seat {seat_index}"
                    )));
                }
            }
        }
    }
    let grid = &seats[0][0].freq;
    for seat in seats {
        for curve in seat {
            if curve.freq.len() != grid.len()
                || curve.freq.iter().zip(grid.iter()).any(|(a, b)| a != b)
            {
                return Err(invalid(
                    "seat matrix grids disagree; resample explicitly first",
                ));
            }
        }
    }
    // The workflow loader and joint objective both use [sub][seat].
    let measurements = seats.clone();
    // Absolute target from the configured neutral tilt curve. Without one
    // the stage preserves the pre-optimization mean level; the shared
    // residual EQ owns absolutes downstream.
    let bins = grid.len();
    let target_db = match crate::channel_target::build_target_tilt_curve(
        channel_name,
        room_config,
        &measurements[0][0],
        false,
    ) {
        Some(tilt) if tilt.spl.len() == bins => tilt.spl.to_vec(),
        Some(_) => {
            return Err(invalid(
                "neutral tilt curve does not cover the seat matrix grid",
            ));
        }
        None => {
            let mut level = vec![0.0; bins];
            for curve in measurements.iter().flatten() {
                for (bin, value) in level.iter_mut().enumerate() {
                    *value += curve.spl[bin];
                }
            }
            let count = (sub_count * seat_count) as f64;
            level.iter().map(|value| value / count).collect()
        }
    };
    let weights = JointSubWeights::default();
    let zeros = vec![0.0; sub_count];
    let physical_outputs: Vec<String> = group
        .subwoofers
        .iter()
        .enumerate()
        // Match build_multisub_dsp_chain_advanced's physical driver IDs.
        // Speaker model names are descriptive and need not be unique.
        .map(|(index, _)| format!("{}_{}", group.name, index + 1))
        .collect();
    let request = JointSubRequest {
        measurements,
        target_db,
        weights,
        lfe_gain_db: zeros.clone(),
        redirected_gain_db: zeros,
        physical_outputs,
        reference_scope: Some(reference_scope),
    };
    let result = optimize_joint_sub_array(&request, &room_config.optimizer, sample_rate)?;
    if !result.coherent_scope_validated {
        return Err(AutoeqError::InvalidConfiguration {
            message: format!(
                "joint multi-sub dispatch for '{channel_name}' refused an unvalidated coherent \
                 result (seat scopes {:?}); select the detailed mode until the shared timing \
                 reference is verified",
                result.reference_scope_per_seat
            ),
        });
    }
    // The downstream shared EQ consumes the same combined shape as the
    // detailed mode: mean magnitude across seats plus the seat-0 complex
    // response the coherent stage was built on.
    let mut mean_spl = vec![0.0; bins];
    for seat_curve in &result.per_seat_post {
        for (bin, value) in mean_spl.iter_mut().enumerate() {
            *value += seat_curve.spl[bin];
        }
    }
    let seats_f64 = result.per_seat_post.len().max(1) as f64;
    for value in mean_spl.iter_mut() {
        *value /= seats_f64;
    }
    let combined = super::types::MultiSubCombinedResponse {
        spatial_magnitude: crate::Curve {
            freq: grid.clone(),
            spl: Array1::from(mean_spl),
            phase: None,
            ..Default::default()
        },
        primary_seat_complex: result.per_seat_post.first().cloned(),
    };
    let base = autoeq_optim::DriverOptimizationResult {
        gains: result.gains_db.clone(),
        delays: result.delays_ms.clone(),
        crossover_freqs: Vec::new(),
        pre_objective: result.pre_components.total,
        post_objective: result.post_components.total,
        converged: result.converged,
    };
    Ok((base, combined, result))
}

/// Build stage diagnostics without normalizing away output changes.
///
/// # Errors
///
/// Returns an error if the channel controls cannot be fingerprinted.
pub(crate) fn joint_sub_diagnostics(
    result: &JointSubResult,
    shared_eq: &[math_audio_iir_fir::Biquad],
    sample_rate: f64,
    level_band_hz: [f64; 2],
    chain: &roomeq_model::ChannelDspChain,
) -> Result<roomeq_model::JointSubDiagnostics> {
    use roomeq_model::{
        JointSubDiagnostics, JointSubGainApplication, JointSubObjectiveReport, JointSubSeatReport,
    };
    let components = |value: &JointSubComponents| JointSubObjectiveReport {
        variation_db2: value.variation,
        output_drive_penalty: value.output_drive,
        target_error_db2: value.target_error,
        total: value.total,
    };
    let seats = result
        .per_seat_pre
        .iter()
        .zip(&result.per_seat_post)
        .enumerate()
        .map(|(index, (before, after_array))| {
            let response = crate::response::compute_peq_complex_response(
                shared_eq,
                &after_array.freq,
                sample_rate,
            );
            let after_shared = crate::response::apply_complex_response(after_array, &response);
            let level = |curve: &Curve| {
                mean_band_level(
                    &curve.spl.to_vec(),
                    &curve.freq.to_vec(),
                    level_band_hz[0],
                    level_band_hz[1],
                )
            };
            JointSubSeatReport {
                seat_index: index,
                reference_scope: result.reference_scope_per_seat[index].clone(),
                before: before.into(),
                after_array: after_array.into(),
                after_shared_eq: (&after_shared).into(),
                before_level_db: level(before),
                after_array_level_db: level(after_array),
                after_shared_eq_level_db: level(&after_shared),
            }
        })
        .collect();
    let assessed_channel_processing =
        roomeq_model::joint_sub_report::joint_sub_processing_identity(chain).map_err(|error| {
            AutoeqError::InvalidConfiguration {
                message: format!("joint-sub report binding failed: {error}"),
            }
        })?;
    Ok(JointSubDiagnostics {
        shared_eq_rejection_reason: None,
        array_rejection_reason: result.array_rejection_reason.clone(),
        scope: "joint array and shared-EQ stage prediction; objective restricted to level_band_hz; full measured responses retained for protected-seat checks; measurement-relative levels; excludes later trims, global routing, physical output safety, and recorded playback".into(),
        level_band_hz,
        physical_outputs: result.physical_outputs.clone(),
        array_gains_db: result.gains_db.clone(),
        array_delays_ms: result.delays_ms.clone(),
        converged: result.converged,
        before_objective: components(&result.pre_components),
        after_array_objective: components(&result.post_components),
        seats,
        gain_applications: result.ledger.entries().iter().map(|entry| JointSubGainApplication {
            stage: match entry.stage { BassGainStage::Lfe => "lfe", BassGainStage::Redirected => "redirected" }.into(),
            physical_output: entry.physical_output.clone(), gain_db: entry.gain_db,
        }).collect(),
        assessed_channel_processing,
        channel_processing_matches: true,
    })
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
    fn shared_residual_eq_rejects_misaligned_and_invalid_curves() {
        let reference = sub_curve(80.0, 0.0);
        let mut shifted = reference.clone();
        shifted.freq[1] = 65.0;
        let error =
            apply_shared_eq_to_residual(&[reference.clone(), shifted], &[0.0; 3]).unwrap_err();
        assert!(error.to_string().contains("resample explicitly"));
        for mutation in 0..4 {
            let mut malformed = reference.clone();
            match mutation {
                0 => malformed.freq[1] = malformed.freq[0],
                1 => malformed.spl[1] = f64::NAN,
                2 => malformed.phase = Some(array![0.0]),
                _ => malformed.spl = array![80.0],
            }
            assert!(apply_shared_eq_to_residual(&[malformed], &[0.0; 3]).is_err());
        }
        let mut huge = reference;
        huge.spl[0] = f64::MAX;
        assert!(apply_shared_eq_to_residual(&[huge], &[f64::MAX, 0.0, 0.0]).is_err());
    }

    #[test]
    fn joint_invalid_measurement_and_output_identity_refused_before_search() {
        let mut config = tiny_config();
        config.algorithm = "must-not-run".into();
        for mutation in 0..4 {
            let mut request = two_sub_two_seat_request();
            match mutation {
                0 => request.measurements[0][0].freq[1] = 30.0,
                1 => request.measurements[0][0].spl[1] = f64::NAN,
                2 => request.physical_outputs[1] = request.physical_outputs[0].clone(),
                _ => request.physical_outputs[1] = " ".into(),
            }
            let error = optimize_joint_sub_array(&request, &config, 48_000.0).unwrap_err();
            assert!(
                !error.to_string().contains("optimization failed"),
                "{error}"
            );
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
    fn roadmap_correction_joint_invalid_weights_refused_before_search() {
        let mut request = two_sub_two_seat_request();
        request.weights.target_error = -1.0;
        let mut config = tiny_config();
        config.algorithm = "must-not-run".into();
        let error = optimize_joint_sub_array(&request, &config, 48_000.0).unwrap_err();
        assert!(
            error.to_string().contains("weight `target_error`"),
            "{error}"
        );
    }

    #[test]
    fn roadmap_correction_joint_objective_respects_requested_band() {
        let mut request = two_sub_two_seat_request();
        for curve in request.measurements.iter_mut().flatten() {
            curve.freq = array![30.0, 60.0, 120.0, 1000.0];
            curve.spl = array![80.0, 80.0, 80.0, 80.0];
            curve.phase = Some(array![0.0, 0.0, 0.0, 0.0]);
        }
        request.target_db = vec![86.0, 86.0, 86.0, 86.0];
        let first = optimize_joint_sub_array(&request, &tiny_config(), 48_000.0).unwrap();
        request.target_db[3] = 20.0;
        let second = optimize_joint_sub_array(&request, &tiny_config(), 48_000.0).unwrap();
        let expected_target_error = second
            .per_seat_post
            .iter()
            .flat_map(|curve| curve.spl.iter().take(3).map(|level| (level - 86.0).powi(2)))
            .sum::<f64>()
            / 6.0;
        assert!(
            (second.post_components.target_error - expected_target_error).abs() < 1e-9,
            "reported candidate score must describe retained in-band responses"
        );
        assert_eq!(
            first.pre_components, second.pre_components,
            "an out-of-band target must not change the joint search objective"
        );
        assert_eq!(
            second.per_seat_pre[0].freq.len(),
            4,
            "full measured response must remain available for safety and reporting"
        );
        let in_band = two_sub_two_seat_request();
        let mut unsupported = tiny_config();
        unsupported.min_freq = 500.0;
        unsupported.max_freq = 2000.0;
        unsupported.algorithm = "must-not-run".into();
        let error = optimize_joint_sub_array(&in_band, &unsupported, 48_000.0).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("at least two shared measurement bins"),
            "{error}"
        );
    }

    #[test]
    fn roadmap_correction_joint_array_protects_seat_despite_better_mean() {
        let mut request = two_sub_two_seat_request();
        // Seat 0 already meets the target; the other seat is 20 dB low.
        // A common increase improves mean squared error but damages seat 0.
        request.measurements = vec![
            vec![sub_curve(80.0, 0.0), sub_curve(60.0, 0.0)],
            vec![sub_curve(80.0, 0.0), sub_curve(60.0, 0.0)],
        ];
        request.target_db.fill(80.0 + 20.0 * 2.0_f64.log10());
        request.weights = JointSubWeights::new(0.0, 0.0, 1.0).unwrap();
        let result = optimize_joint_sub_array(&request, &tiny_config(), 48_000.0).unwrap();
        for (actual, target) in result.per_seat_post[0].spl.iter().zip(&request.target_db) {
            assert!(
                (actual - target).abs() < 1e-6,
                "protected seat regressed: {actual} vs {target}"
            );
        }
        assert!(
            result
                .array_rejection_reason
                .as_deref()
                .unwrap()
                .contains("seat_target_weighted_rms_regressed")
        );
        assert_eq!(result.gains_db, vec![0.0; 2]);
        assert_eq!(result.delays_ms, vec![0.0; 2]);
        assert_eq!(result.post_components, result.pre_components);
    }

    #[test]
    fn roadmap_correction_joint_array_accepts_improvement_at_every_seat() {
        let mut request = two_sub_two_seat_request();
        request.target_db.fill(92.0);
        request.weights = JointSubWeights::new(0.0, 0.0, 1.0).unwrap();
        let result = optimize_joint_sub_array(&request, &tiny_config(), 48_000.0).unwrap();
        assert!(result.array_rejection_reason.is_none());
        assert!(result.post_components.total < result.pre_components.total);
        for (pre, post) in result.per_seat_pre.iter().zip(&result.per_seat_post) {
            for ((before, after), target) in pre.spl.iter().zip(&post.spl).zip(&request.target_db) {
                assert!((after - target).abs() < (before - target).abs());
            }
        }
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

    fn dispatch_group() -> roomeq_model::MultiSubGroup {
        roomeq_model::MultiSubGroup {
            name: String::from("subs"),
            speaker_name: None,
            subwoofers: vec![
                roomeq_model::MeasurementSource::InMemory(sub_curve(80.0, 0.0)),
                roomeq_model::MeasurementSource::InMemory(sub_curve(80.0, 0.0)),
            ],
            allpass_optimization: false,
            joint_optimization: true,
        }
    }

    fn dispatch_room_config() -> roomeq_model::RoomConfig {
        roomeq_model::RoomConfig {
            optimizer: tiny_config(),
            ..Default::default()
        }
    }

    fn dispatch_prepared() -> crate::group_processing::PreparedMultiSubGroup {
        crate::group_processing::PreparedMultiSubGroup {
            subwoofers: vec![sub_curve(80.0, 0.0), sub_curve(80.0, 0.0)],
            seat_measurements: Some(vec![vec![sub_curve(80.0, 0.0)], vec![sub_curve(78.0, 0.0)]]),
            reference_scope: None,
        }
    }

    /// Unknown scope never authorizes a coherent result: dispatch refuses
    /// before any array search runs, even with measured phase everywhere.
    #[test]
    fn roadmap_correction_joint_unknown_scope_refused_before_search() {
        let error = process_joint_sub_group(
            "subs",
            &dispatch_group(),
            &dispatch_room_config(),
            48_000.0,
            &dispatch_prepared(),
            None,
        )
        .expect_err("unknown scope must refuse coherent dispatch");
        assert!(error.to_string().contains("timing reference"), "{error}");
    }

    #[test]
    fn roadmap_correction_joint_outer_dispatch_preserves_admission_refusal() {
        let resources = crate::eq::EqResources::default();
        for missing_matrix in [false, true] {
            let mut prepared = dispatch_prepared();
            if missing_matrix {
                prepared.seat_measurements = None;
            }
            let error = crate::group_processing::process_multisub_group(
                "subs",
                &dispatch_group(),
                &dispatch_room_config(),
                48_000.0,
                &prepared,
                &resources,
                &resources,
            )
            .expect_err("selected joint mode must not bypass refusal through another optimizer");
            let message = error.to_string();
            assert!(
                message.contains(if missing_matrix {
                    "seat matrix"
                } else {
                    "timing reference"
                }),
                "{message}"
            );
        }
    }

    #[test]
    fn allpass_outer_dispatch_requires_matching_timing_scope() {
        let resources = crate::eq::EqResources::default();
        let mut group = dispatch_group();
        group.joint_optimization = false;
        group.allpass_optimization = true;
        for scope in [
            None,
            Some(vec![vec!["clock-a".into()], vec!["clock-b".into()]]),
            Some(vec![vec!["unknown".into()]; 2]),
        ] {
            let mut prepared = dispatch_prepared();
            prepared.reference_scope = scope;
            let error = crate::group_processing::process_multisub_group(
                "subs",
                &group,
                &dispatch_room_config(),
                48_000.0,
                &prepared,
                &resources,
                &resources,
            )
            .err()
            .expect("all-pass workflow must refuse unverified timing");
            assert!(error.to_string().contains("timing reference"), "{error}");
        }
    }

    #[test]
    fn joint_dispatch_refuses_placeholder_timing_before_search() {
        for reference in ["", " ", "UNKNOWN", " unknown "] {
            let mut prepared = dispatch_prepared();
            prepared.reference_scope = Some(vec![vec![reference.into()]; 2]);
            let mut room = dispatch_room_config();
            // An invalid backend proves timing admission precedes search.
            room.optimizer.algorithm = "invalid-diagnostic-backend".into();
            let error = process_joint_sub_group(
                "subs",
                &dispatch_group(),
                &room,
                48_000.0,
                &prepared,
                prepared.reference_scope.clone(),
            )
            .err()
            .expect("placeholder timing cannot authorize joint processing");
            assert!(
                error.to_string().contains("timing reference"),
                "{reference:?}: {error}"
            );
        }
    }

    /// The library objective labels unknown scope honestly: optimization
    /// still runs, but the result is explicitly unvalidated.
    #[test]
    fn roadmap_correction_joint_unknown_scope_marks_result_unvalidated() {
        let mut request = two_sub_two_seat_request();
        request.reference_scope = None;
        let result = optimize_joint_sub_array(&request, &tiny_config(), 48000.0).unwrap();
        assert!(!result.coherent_scope_validated);
        assert_eq!(
            result.reference_scope_per_seat,
            vec!["unknown".to_string(), "unknown".to_string()]
        );
    }

    /// A verified agreeing scope reaches the objective and emits the
    /// chosen graph: per-sub gains/delays plus the combined response.
    #[test]
    fn roadmap_correction_joint_dispatch_preserves_source_seat_axes() {
        let mut prepared = dispatch_prepared();
        prepared.seat_measurements = Some(vec![
            vec![
                sub_curve(80.0, 0.0),
                sub_curve(79.0, 0.0),
                sub_curve(78.0, 0.0),
            ],
            vec![
                sub_curve(78.0, 0.0),
                sub_curve(79.0, 0.0),
                sub_curve(80.0, 0.0),
            ],
        ]);
        let scope = vec![vec!["seat-a".into(), "seat-b".into(), "seat-c".into()]; 2];
        let (base, combined) = process_joint_sub_group(
            "subs",
            &dispatch_group(),
            &dispatch_room_config(),
            48_000.0,
            &prepared,
            Some(scope),
        )
        .expect("loader matrix is [sub][seat], including nonsquare matrices");
        assert_eq!(base.gains.len(), 2);
        assert_eq!(base.delays.len(), 2);
        assert!(combined.primary_seat_complex.is_some());
    }

    #[test]
    fn roadmap_correction_joint_selection_precedes_legacy_multiseat() {
        use crate::eq::EqResources;
        use crate::group_processing::process_multisub_group;

        let mut prepared = dispatch_prepared();
        prepared.reference_scope = Some(vec![vec!["stationary-seat-a".into()]; 2]);
        let mut room = dispatch_room_config();
        room.optimizer.num_filters = 1;
        room.optimizer.refine = false;
        let resources = EqResources::default();
        let standalone = process_multisub_group(
            "subs",
            &dispatch_group(),
            &room,
            48_000.0,
            &prepared,
            &resources,
            &resources,
        )
        .expect("joint group emits a graph");
        room.optimizer.multi_seat = Some(roomeq_model::MultiSeatConfig {
            enabled: true,
            per_sub_peq: false,
            global_eq: false,
            ..Default::default()
        });
        let with_legacy = process_multisub_group(
            "subs",
            &dispatch_group(),
            &room,
            48_000.0,
            &prepared,
            &resources,
            &resources,
        )
        .expect("explicit joint selection takes precedence");
        assert_eq!(
            serde_json::to_value(&standalone.0.drivers).unwrap(),
            serde_json::to_value(&with_legacy.0.drivers).unwrap(),
            "legacy multi-seat settings must not replace joint array controls",
        );
        for (expected, actual) in standalone.3.spl.iter().zip(with_legacy.3.spl.iter()) {
            assert!(
                (expected - actual).abs() < 1e-9,
                "joint pre-EQ response changed"
            );
        }

        prepared.seat_measurements.as_mut().unwrap()[1][0].phase = None;
        let error = process_multisub_group(
            "subs",
            &dispatch_group(),
            &room,
            48_000.0,
            &prepared,
            &resources,
            &resources,
        )
        .expect_err("selected coherent joint mode must refuse missing phase");
        assert!(error.to_string().contains("phase"), "{error}");
    }

    #[test]
    fn roadmap_correction_joint_report_replays_serialized_controls_and_detects_mutation() {
        use crate::dsp_realization::{NoConvolutionIr, RealizedDsp};
        use crate::eq::EqResources;
        use crate::group_processing::process_multisub_group;
        let mut prepared = dispatch_prepared();
        prepared.seat_measurements = Some(vec![
            vec![
                sub_curve(80.0, 0.0),
                sub_curve(78.0, 30.0),
                sub_curve(79.0, -20.0),
            ],
            vec![
                sub_curve(78.0, 25.0),
                sub_curve(80.0, -40.0),
                sub_curve(77.0, 10.0),
            ],
        ]);
        prepared.reference_scope =
            Some(vec![
                vec!["seat-a".into(), "seat-b".into(), "seat-c".into()];
                2
            ]);
        let mut room = dispatch_room_config();
        room.optimizer.num_filters = 1;
        room.optimizer.refine = false;
        let resources = EqResources::default();
        let (chain, ..) = process_multisub_group(
            "subs",
            &dispatch_group(),
            &room,
            48_000.0,
            &prepared,
            &resources,
            &resources,
        )
        .unwrap();
        let serialized = serde_json::to_string(&chain).unwrap();
        let mut roundtrip: roomeq_model::ChannelDspChain =
            serde_json::from_str(&serialized).unwrap();
        let report = roundtrip.joint_sub.as_ref().expect("joint report emitted");
        assert!(report.channel_processing_matches);
        assert_eq!(report.seats.len(), 3);
        assert_eq!(report.gain_applications.len(), 4);
        let measurements = prepared.seat_measurements.as_ref().unwrap();
        let mut shared_chain = roundtrip.clone();
        shared_chain.drivers = None;
        for (seat, prediction) in report.seats.iter().enumerate() {
            for (bin, frequency) in prediction.after_shared_eq.freq.iter().enumerate() {
                let mut sum = Complex64::new(0.0, 0.0);
                for (sub, driver) in roundtrip.drivers.as_ref().unwrap().iter().enumerate() {
                    let mut branch = shared_chain.clone();
                    branch.plugins = driver.plugins.clone();
                    let mut ir = NoConvolutionIr;
                    let control = RealizedDsp::new(&branch, 48_000.0, &mut ir)
                        .unwrap()
                        .response_at(*frequency)
                        .unwrap();
                    let measured = &measurements[sub][seat];
                    sum += Complex64::from_polar(
                        10.0_f64.powf(measured.spl[bin] / 20.0),
                        measured.phase.as_ref().unwrap()[bin].to_radians(),
                    ) * control;
                }
                let mut ir = NoConvolutionIr;
                sum *= RealizedDsp::new(&shared_chain, 48_000.0, &mut ir)
                    .unwrap()
                    .response_at(*frequency)
                    .unwrap();
                let replay_db = 20.0 * sum.norm().log10();
                assert!(
                    (replay_db - prediction.after_shared_eq.spl[bin]).abs() < 1e-8,
                    "seat {seat}, bin {bin}: {replay_db} != {}",
                    prediction.after_shared_eq.spl[bin]
                );
            }
        }
        roundtrip.drivers.as_mut().unwrap()[1]
            .plugins
            .push(crate::output::create_gain_plugin(1.0));
        let output = crate::output::create_dsp_chain_output(
            std::collections::HashMap::from([("subs".into(), roundtrip)]),
            None,
        );
        let historical = output.channels["subs"].joint_sub.as_ref().unwrap();
        assert!(
            !historical.channel_processing_matches,
            "mutated controls invalidate current prediction"
        );
        assert_eq!(
            historical.seats.len(),
            3,
            "historical evidence is not discarded"
        );
    }

    #[test]
    fn roadmap_correction_joint_emission_preserves_small_array_controls() {
        use crate::dsp_realization::{NoConvolutionIr, RealizedDsp};
        let gain_db = 0.005;
        let delay_ms = 0.0005;
        let chain = crate::output::build_multisub_dsp_chain_with_allpass(
            "subs",
            "subs",
            2,
            &[0.0, gain_db],
            &[0.0, delay_ms],
            &[],
            None,
            None,
            None,
            None,
            48_000.0,
        );
        let frequency = 1_000.0;
        let mut ir = NoConvolutionIr;
        let actual = RealizedDsp::new(&chain, 48_000.0, &mut ir)
            .unwrap()
            .response_at(frequency)
            .unwrap();
        let expected = Complex64::new(1.0, 0.0)
            + Complex64::from_polar(
                10.0_f64.powf(gain_db / 20.0),
                -2.0 * PI * frequency * delay_ms / 1000.0,
            );
        assert!(
            (actual - expected).norm() < 1e-12,
            "emission must not silently drop optimized controls"
        );
    }

    #[test]
    fn roadmap_correction_joint_validated_scope_emits_graph() {
        let scope = Some(vec![
            vec!["seat-loop".to_string()],
            vec!["seat-loop".to_string()],
        ]);
        let (base, combined) = process_joint_sub_group(
            "subs",
            &dispatch_group(),
            &dispatch_room_config(),
            48_000.0,
            &dispatch_prepared(),
            scope,
        )
        .expect("verified scope must reach joint optimization");
        assert_eq!(base.gains.len(), 2);
        assert_eq!(base.delays.len(), 2);
        assert_eq!(base.gains[0], 0.0);
        assert_eq!(base.delays[0], 0.0);
        assert!(base.gains.iter().all(|gain| gain.is_finite()));
        assert!(
            base.delays
                .iter()
                .all(|delay| *delay >= 0.0 && *delay <= 20.0)
        );
        assert_eq!(combined.spatial_magnitude.spl.len(), 3);
        assert!(combined.primary_seat_complex.is_some());
        assert!(base.post_objective <= base.pre_objective + 1e-9);
    }

    /// A ragged transfer matrix is refused: every sub must cover every seat.
    #[test]
    fn roadmap_correction_joint_ragged_matrix_refused() {
        let mut request = two_sub_two_seat_request();
        request.measurements[1].pop();
        request.reference_scope = Some(vec![
            vec!["seat-loop".to_string(), "seat-loop".to_string()],
            vec!["seat-loop".to_string()],
        ]);
        assert!(optimize_joint_sub_array(&request, &tiny_config(), 48000.0).is_err());
    }

    /// Coherent cancellation cannot beat the protected baseline claim:
    /// two subs 180° apart keep raw output views, bounded gains, and a
    /// post total no worse than pre — never a normalized illusion.
    #[test]
    fn roadmap_correction_joint_cancellation_keeps_protected_baseline() {
        let mut request = two_sub_two_seat_request();
        for seat in request.measurements[1].iter_mut() {
            seat.phase = Some(ndarray::array![180.0, 180.0, 180.0]);
        }
        let result = optimize_joint_sub_array(&request, &tiny_config(), 48000.0).unwrap();
        assert!(result.post_components.total <= result.pre_components.total + 1e-9);
        assert!(
            result
                .gains_db
                .iter()
                .all(|gain| *gain >= -12.0 && *gain <= 12.0)
        );
        assert!(
            result
                .delays_ms
                .iter()
                .all(|delay| *delay >= 0.0 && *delay <= 20.0)
        );
        for view in &result.output_views {
            assert!(view.pre_level_db.is_finite());
            assert!(view.post_level_db.is_finite());
            assert!((view.delta_db - (view.post_level_db - view.pre_level_db)).abs() < 1e-12);
        }
    }
}
