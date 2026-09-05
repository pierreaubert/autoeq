//! Per-filter audibility veto (Phase A of the audibility plan).
//!
//! Each emitted biquad is priced in perceptual units on the ERB-rate axis:
//! peak with/without level difference, affected ERB width, and an
//! approximate masked-loudness delta at calibrated SPL. Verdicts carry
//! machine-readable reason codes; removal is logged, never silent.
//!
//! The masked-loudness model here is deliberately simplified and
//! report-first: per-band excitation with upward masking spread, compressive
//! specific loudness, integrated over ERB-rate measure. It is grounded to
//! order-of-magnitude sones (a flat 75-phon spectrum yields ~11 sones) but
//! is NOT a reference-grade ISO 532 implementation — that arrives with the
//! Phase D objective. Threshold numerics are starting calibrations and must
//! be re-verified against primary publications before any default flip.
//!
//! Loudness background convention: bands are anchored at the calibrated
//! level with the *filter-composed* shape around it, i.e. the veto prices
//! the filters' own audible contribution on a flat-phon background rather
//! than depending on seat-dependent measurement shape.

use autoeq_core::auditory_frequency::{erb_rate, try_erb_rate_cell_widths};
use math_audio_iir_fir::Biquad;
use ndarray::Array1;
use roomeq_model::{FilterAudibilityConfig, FilterVetoVerdict, VetoDecision, VetoReason};

/// Upward masking spread in dB per ERB (documented simplification of the
/// level-dependent Zwicker slopes; no downward spread).
const MASKING_SPREAD_DB_PER_ERB: f64 = 10.0;
/// Flat hearing-threshold approximation in dB for the simplified model.
const HEARING_THRESHOLD_DB: f64 = 10.0;
/// Compressive specific-loudness exponent (Stevens/Zwicker-inspired).
const LOUDNESS_EXPONENT: f64 = 0.23;
/// Scale grounding the simplified integral to order-of-magnitude sones: a
/// flat 75-phon spectrum over ~41 ERBs sums to ~107 unscaled units, and
/// 75 phons is 2^3.5 ≈ 11.3 sones, hence ~0.1.
const LOUDNESS_SCALE: f64 = 0.1;

/// Inputs for one veto evaluation over an emitted filter set.
pub struct VetoEvaluation<'a> {
    /// Emitted filters in output order.
    pub filters: &'a [Biquad],
    /// Ascending response grid in Hz (normally the objective grid).
    pub freqs: &'a Array1<f64>,
    /// Calibrated evaluation level in phons.
    pub listening_phon: f64,
    /// Veto thresholds and switches.
    pub config: FilterAudibilityConfig,
    /// Resolved HF guard start in Hz.
    pub hf_guard_start_hz: f64,
}

/// dB magnitude response of one filter on the grid.
fn filter_db_response(filter: &Biquad, freqs: &Array1<f64>) -> Array1<f64> {
    filter.np_log_result(freqs)
}

/// ERB-rate cell widths for the grid, or `None` when the axis is invalid
/// (width vetoes are then skipped: unknown width never vetoes).
fn erb_weights(freqs: &Array1<f64>) -> Option<Array1<f64>> {
    try_erb_rate_cell_widths(freqs)
}

/// ERB-rate positions of the grid bins.
fn erb_positions(freqs: &Array1<f64>) -> Vec<f64> {
    freqs.iter().map(|frequency| erb_rate(*frequency)).collect()
}

/// Approximate total masked loudness in sones for a composite dB shape
/// anchored so its ERB-weighted mean sits at `listening_phon`.
///
/// `shape_db` is a perturbation shape (e.g. summed filter responses);
/// anchoring puts the background at the calibrated level independent of
/// measurement shape (see module docs).
fn approximate_loudness_sones(
    shape_db: &Array1<f64>,
    erb: &[f64],
    weights: Option<&Array1<f64>>,
    listening_phon: f64,
) -> f64 {
    let count = shape_db.len();
    if count == 0 {
        return 0.0;
    }
    let weight_sum: f64 = match weights {
        Some(w) => w.iter().sum(),
        None => count as f64,
    };
    if weight_sum <= 0.0 {
        return 0.0;
    }
    let mean: f64 = match weights {
        Some(w) => shape_db
            .iter()
            .zip(w.iter())
            .map(|(s, w)| s * w)
            .sum::<f64>()
            / weight_sum,
        None => shape_db.iter().sum::<f64>() / count as f64,
    };
    // Band levels at calibrated SPL.
    let levels: Vec<f64> = shape_db.iter().map(|s| listening_phon + (s - mean)).collect();
    // Upward masking spread: each band is masked by lower bands decaying
    // at MASKING_SPREAD_DB_PER_ERB.
    let mut total = 0.0;
    for (j, _) in levels.iter().enumerate() {
        let mut excitation = levels[j];
        for (k, _) in levels.iter().enumerate().take(j) {
            let spread = levels[k] - MASKING_SPREAD_DB_PER_ERB * (erb[j] - erb[k]);
            if spread > excitation {
                excitation = spread;
            }
        }
        let above_threshold = excitation - HEARING_THRESHOLD_DB;
        if above_threshold > 0.0 {
            let weight = weights.map(|w| w[j]).unwrap_or(1.0);
            total += weight * above_threshold.powf(LOUDNESS_EXPONENT);
        }
    }
    LOUDNESS_SCALE * total
}

/// Masked-loudness delta in sones between the full filter set and the set
/// without filter `without_index`.
fn loudness_delta_sones(
    responses: &[Array1<f64>],
    without_index: usize,
    erb: &[f64],
    weights: Option<&Array1<f64>>,
    listening_phon: f64,
) -> f64 {
    let count = responses.first().map(|r| r.len()).unwrap_or(0);
    let mut full = Array1::<f64>::zeros(count);
    for response in responses {
        full = full + response;
    }
    let mut partial = full.clone();
    if let Some(removed) = responses.get(without_index) {
        partial = partial - removed;
    }
    (approximate_loudness_sones(&full, erb, weights, listening_phon)
        - approximate_loudness_sones(&partial, erb, weights, listening_phon))
    .abs()
}

/// ERB-rate width of the region where `|response|` reaches at least
/// `threshold_db`, with fractional edge interpolation.
///
/// Biquad |Δ| shapes are single-lobed, so the width is the ERB distance
/// between the linearly interpolated threshold crossings around the peak
/// bin. Interpolation gives sub-bin resolution: production 200-point grids
/// have ~0.25 ERB bins at 1 kHz, and whole-bin sums would quantize every
/// narrow width up past the audibility floor. Shelf-like shapes run into
/// the grid edge, yielding correctly large widths.
///
/// The ERB axis must be non-decreasing; production clipped grids can start
/// with float-dust duplicates (e.g. `19.999999999999996, 20.0`) that map to
/// one ERB coordinate, and zero-width steps contribute nothing to the walk.
/// A decreasing grid yields infinity (unknown width never width-vetoes).
fn affected_erb_width(response: &Array1<f64>, erb: &[f64], threshold_db: f64) -> f64 {
    if response.len() < 2 || erb.len() != response.len() {
        return f64::INFINITY;
    }
    let ordered = erb
        .windows(2)
        .all(|pair| pair[1] >= pair[0] && pair[0].is_finite() && pair[1].is_finite());
    if !ordered {
        return f64::INFINITY;
    }
    let mut peak_idx = 0;
    for (i, value) in response.iter().enumerate() {
        if value.abs() > response[peak_idx].abs() {
            peak_idx = i;
        }
    }
    if response[peak_idx].abs() < threshold_db {
        return 0.0;
    }
    let crossing = |mut index: usize, step: isize| -> f64 {
        loop {
            let next = index.wrapping_add_signed(step);
            if next >= response.len() {
                return erb[index];
            }
            let here = response[index].abs();
            let there = response[next].abs();
            if there < threshold_db {
                // Fractional crossing between `index` (above) and `next`
                // (below), interpolated in ERB coordinates.
                let denom = (here - there).abs().max(f64::MIN_POSITIVE);
                let frac = ((here - threshold_db) / denom).clamp(0.0, 1.0);
                return erb[index] + frac * (erb[next] - erb[index]);
            }
            index = next;
        }
    };
    (crossing(peak_idx, 1) - crossing(peak_idx, -1)).max(0.0)
}

/// Evaluate the audibility veto over an emitted filter set, in order.
///
/// Pure computation: no filtering, no logging. See [`apply_audibility_veto`]
/// for the enforcing wrapper.
pub fn evaluate_audibility_veto(evaluation: &VetoEvaluation<'_>) -> Vec<FilterVetoVerdict> {
    let config = evaluation.config;
    let responses: Vec<Array1<f64>> = evaluation
        .filters
        .iter()
        .map(|filter| filter_db_response(filter, evaluation.freqs))
        .collect();
    let weights = erb_weights(evaluation.freqs);
    let erb = erb_positions(evaluation.freqs);
    let weights_ref = weights.as_ref();

    evaluation
        .filters
        .iter()
        .enumerate()
        .map(|(index, filter)| {
            let response = &responses[index];
            let peak_delta_db = response.iter().fold(0.0_f64, |max, value| max.max(value.abs()));
            let width_threshold_db = (peak_delta_db / 2.0).max(config.jnd_db / 2.0);
            let width = affected_erb_width(response, &erb, width_threshold_db);
            let loudness_delta = loudness_delta_sones(
                &responses,
                index,
                &erb,
                weights_ref,
                evaluation.listening_phon,
            );
            let (decision, reason) = if peak_delta_db < config.jnd_db {
                (VetoDecision::Remove, VetoReason::SubJnd)
            } else if width < config.min_audible_erb_width {
                (VetoDecision::Remove, VetoReason::SubErbWidth)
            } else if config.hf_guard_enabled
                && filter.freq > evaluation.hf_guard_start_hz
                && filter.q > config.hf_guard_max_q
            {
                (VetoDecision::Remove, VetoReason::HighQAboveGuard)
            } else {
                (VetoDecision::Keep, VetoReason::Audible)
            };
            FilterVetoVerdict {
                index,
                center_hz: filter.freq,
                q: filter.q,
                gain_db: filter.db_gain,
                peak_delta_db,
                affected_erb_width: width,
                loudness_delta_sones: loudness_delta,
                decision,
                reason,
                enforced: false,
            }
        })
        .collect()
}

/// Evaluate the veto and, unless `report_only`, remove `Remove` filters.
///
/// Convenience wrapper over [`evaluate_audibility_veto`] plus
/// [`enforce_veto_verdicts`]. Returns the input untouched with no verdicts
/// when the config is disabled or the set is empty.
pub fn apply_audibility_veto(
    filters: Vec<Biquad>,
    evaluation: &VetoEvaluation<'_>,
) -> (Vec<Biquad>, Vec<FilterVetoVerdict>) {
    if !evaluation.config.enabled || filters.is_empty() {
        return (filters, Vec::new());
    }
    let verdicts = evaluate_audibility_veto(evaluation);
    enforce_veto_verdicts(filters, verdicts, !evaluation.config.report_only)
}

/// Enforce evaluated verdicts: drop `Remove` filters when `enforce` is true.
///
/// Returns the kept filters plus verdicts for every evaluated filter
/// (`enforced` marks verdicts that actually removed a filter). Removal is
/// logged with reason codes; keeps are debug-logged with a summary histogram
/// at info level. With `enforce == false` (report-only) every filter is kept
/// and nothing is marked enforced.
pub fn enforce_veto_verdicts(
    filters: Vec<Biquad>,
    mut verdicts: Vec<FilterVetoVerdict>,
    enforce: bool,
) -> (Vec<Biquad>, Vec<FilterVetoVerdict>) {
    let mut kept = Vec::with_capacity(filters.len());
    let mut removed = 0_usize;
    for (filter, verdict) in filters.into_iter().zip(verdicts.iter_mut()) {
        if verdict.decision == VetoDecision::Remove && enforce {
            verdict.enforced = true;
            removed += 1;
            log::info!(
                "  Audibility veto: removing filter #{index} ({center:.0} Hz, Q={q:.2}, {gain:+.1} dB): \
                 reason={reason:?} peakΔ={peak:.2} dB width={width:.2} ERB loudnessΔ={loud:.4} sones",
                index = verdict.index,
                center = verdict.center_hz,
                q = verdict.q,
                gain = verdict.gain_db,
                reason = verdict.reason,
                peak = verdict.peak_delta_db,
                width = verdict.affected_erb_width,
                loud = verdict.loudness_delta_sones,
            );
        } else {
            if verdict.decision == VetoDecision::Remove {
                log::debug!(
                    "  Audibility veto (report-only): filter #{index} would be removed: reason={reason:?}",
                    index = verdict.index,
                    reason = verdict.reason,
                );
            }
            kept.push(filter);
        }
    }
    let kept_count = kept.len();
    let mut histogram: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    for verdict in &verdicts {
        *histogram.entry(format!("{:?}", verdict.reason)).or_insert(0) += 1;
    }
    log::info!(
        "  Audibility veto: {kept_count} kept, {removed} removed (enforce={enforce}); reasons: {histogram:?}",
    );
    (kept, verdicts)
}

/// Greedy backward elimination in veto (loudness-delta) units.
///
/// Repeatedly removes the filter whose removal changes total masked
/// loudness least, while that change stays below `threshold_sones` (same
/// greedy shape as [`super::consts::backward_eliminate`], different units).
/// Returns the kept filters; loss is recomputed by the caller in raw units.
pub fn backward_eliminate_veto_units(
    filters: Vec<Biquad>,
    freqs: &Array1<f64>,
    listening_phon: f64,
    threshold_sones: f64,
) -> Vec<Biquad> {
    let mut remaining: Vec<Array1<f64>> = filters
        .iter()
        .map(|filter| filter_db_response(filter, freqs))
        .collect();
    let mut kept: Vec<Biquad> = filters;
    let weights = erb_weights(freqs);
    let erb = erb_positions(freqs);

    loop {
        if remaining.len() <= 1 {
            break;
        }
        let mut min_impact = f64::INFINITY;
        let mut min_idx = 0;
        for i in 0..remaining.len() {
            let impact = loudness_delta_sones(
                &remaining,
                i,
                &erb,
                weights.as_ref(),
                listening_phon,
            );
            if impact < min_impact {
                min_impact = impact;
                min_idx = i;
            }
        }
        if min_impact < threshold_sones {
            log::info!(
                "  Veto-units elimination: removing filter #{min_idx} \
                 (loudness impact={min_impact:.4} < threshold={threshold_sones:.4} sones)"
            );
            remaining.remove(min_idx);
            kept.remove(min_idx);
        } else {
            break;
        }
    }
    kept
}

#[cfg(test)]
mod audibility_veto_tests {
    use super::*;
    use math_audio_iir_fir::BiquadFilterType;

    const SAMPLE_RATE: f64 = 48_000.0;

    fn grid() -> Array1<f64> {
        Array1::from(
            (0..200)
                .map(|i| 20.0 * (1000.0_f64).powf(i as f64 / 199.0))
                .collect::<Vec<_>>(),
        )
    }

    fn config() -> FilterAudibilityConfig {
        FilterAudibilityConfig {
            report_only: true,
            ..FilterAudibilityConfig::default()
        }
    }

    fn evaluate(filters: &[Biquad]) -> Vec<FilterVetoVerdict> {
        let freqs = grid();
        let evaluation = VetoEvaluation {
            filters,
            freqs: &freqs,
            listening_phon: 75.0,
            config: config(),
            hf_guard_start_hz: 1600.0,
        };
        evaluate_audibility_veto(&evaluation)
    }

    fn peak(db_gain: f64, freq: f64, q: f64) -> Biquad {
        Biquad::new(BiquadFilterType::Peak, freq, SAMPLE_RATE, q, db_gain)
    }

    #[test]
    fn sub_jnd_ripple_is_vetoed() {
        // 0.2 dB ripple at 500 Hz: far below the 1 dB JND floor.
        let verdicts = evaluate(&[peak(0.2, 500.0, 1.0)]);
        assert_eq!(verdicts.len(), 1);
        assert_eq!(verdicts[0].decision, VetoDecision::Remove);
        assert_eq!(verdicts[0].reason, VetoReason::SubJnd);
        assert!(verdicts[0].peak_delta_db < 1.0);
    }

    #[test]
    fn audible_midrange_peak_is_kept() {
        // 3 dB correction at 500 Hz, Q 1: clearly audible, wide enough.
        let verdicts = evaluate(&[peak(3.0, 500.0, 1.0)]);
        assert_eq!(verdicts[0].decision, VetoDecision::Keep);
        assert_eq!(verdicts[0].reason, VetoReason::Audible);
        assert!(verdicts[0].peak_delta_db > 1.0);
        assert!(verdicts[0].loudness_delta_sones > 0.0);
    }

    #[test]
    fn sub_erb_notch_is_vetoed() {
        // Deep but ultra-narrow notch (-6 dB, Q25 spans ~0.3 ERB at 1 kHz):
        // peak clears JND, ERB width does not.
        let verdicts = evaluate(&[peak(-6.0, 1000.0, 25.0)]);
        assert_eq!(verdicts[0].decision, VetoDecision::Remove);
        assert_eq!(verdicts[0].reason, VetoReason::SubErbWidth);
        assert!(verdicts[0].peak_delta_db >= 1.0);
        assert!(verdicts[0].affected_erb_width < 0.5);
    }

    #[test]
    fn borderline_width_notch_is_kept() {
        // Same depth at Q12 spans ~0.63 ERB: above the 0.5 floor, so kept.
        let verdicts = evaluate(&[peak(-6.0, 1000.0, 12.0)]);
        assert_eq!(verdicts[0].decision, VetoDecision::Keep);
    }

    #[test]
    fn hf_high_q_is_vetoed() {
        // Narrow HF correction above the guard start.
        let verdicts = evaluate(&[peak(2.0, 8000.0, 6.0)]);
        assert_eq!(verdicts[0].decision, VetoDecision::Remove);
        assert_eq!(verdicts[0].reason, VetoReason::HighQAboveGuard);
    }

    #[test]
    fn broad_hf_shelf_is_kept() {
        // Broad HF shelf: above guard start but low Q, wide effect.
        let verdicts = evaluate(&[Biquad::new(
            BiquadFilterType::Highshelf,
            8000.0,
            SAMPLE_RATE,
            0.7,
            -2.0,
        )]);
        assert_eq!(verdicts[0].decision, VetoDecision::Keep);
        assert_eq!(verdicts[0].reason, VetoReason::Audible);
    }

    fn apply(filters: Vec<Biquad>, report_only: bool) -> (Vec<Biquad>, Vec<FilterVetoVerdict>) {
        let freqs = grid();
        let evaluation = VetoEvaluation {
            filters: &filters,
            freqs: &freqs,
            listening_phon: 75.0,
            config: FilterAudibilityConfig {
                report_only,
                ..FilterAudibilityConfig::default()
            },
            hf_guard_start_hz: 1600.0,
        };
        let verdicts = evaluate_audibility_veto(&evaluation);
        enforce_veto_verdicts(filters, verdicts, !report_only)
    }

    #[test]
    fn report_only_records_without_removing() {
        let filters = vec![peak(0.2, 500.0, 1.0), peak(3.0, 500.0, 1.0)];
        let (kept, verdicts) = apply(filters, true);
        assert_eq!(kept.len(), 2, "report-only must not remove");
        assert_eq!(verdicts[0].decision, VetoDecision::Remove);
        assert!(!verdicts[0].enforced);
    }

    #[test]
    fn enforcement_removes_with_reason_code() {
        let filters = vec![peak(0.2, 500.0, 1.0), peak(3.0, 500.0, 1.0)];
        let (kept, verdicts) = apply(filters, false);
        assert_eq!(kept.len(), 1);
        assert!(verdicts[0].enforced);
        assert_eq!(verdicts[0].reason, VetoReason::SubJnd);
        assert!(!verdicts[1].enforced);
    }

    #[test]
    fn disabled_config_passes_through_untouched() {
        let filters = vec![peak(0.2, 500.0, 1.0)];
        let freqs = grid();
        let evaluation = VetoEvaluation {
            filters: &filters,
            freqs: &freqs,
            listening_phon: 75.0,
            config: FilterAudibilityConfig {
                enabled: false,
                ..FilterAudibilityConfig::default()
            },
            hf_guard_start_hz: 1600.0,
        };
        // The clone is consumed while `evaluation` borrows the original;
        // the disabled path must return the set untouched with no verdicts.
        let (kept, verdicts) = apply_audibility_veto(filters.clone(), &evaluation);
        assert_eq!(kept.len(), 1);
        assert!(verdicts.is_empty());
    }

    #[test]
    fn veto_units_elimination_drops_inaudible_first() {
        let filters = vec![peak(0.1, 400.0, 1.0), peak(4.0, 400.0, 1.0)];
        let freqs = grid();
        let kept = backward_eliminate_veto_units(filters, &freqs, 75.0, 0.05);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].db_gain - 4.0).abs() < 1e-9);
    }

    #[test]
    fn float_dust_duplicate_endpoint_keeps_finite_width() {
        // Production clipped grids can start with float-dust duplicates
        // (measurement `exp(ln(20))` vs canonical `20.0`) mapping to one
        // ERB coordinate. Widths must stay finite, not collapse to infinity.
        let mut freqs = grid();
        freqs[0] = 19.999999999999996;
        freqs[1] = 20.0;
        let filter = peak(-6.0, 1000.0, 25.0);
        let response = filter_db_response(&filter, &freqs);
        let peak_db = response.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let erb = erb_positions(&freqs);
        assert_eq!(erb[0], erb[1], "test needs a dust-duplicate ERB step");
        let width = affected_erb_width(&response, &erb, (peak_db / 2.0).max(0.5));
        assert!(
            width.is_finite() && width < 0.5,
            "dust duplicates must not poison the width, got {width}"
        );
    }

    #[test]
    fn flat_background_loudness_is_order_of_magnitude_sones() {
        // Grounding check: a flat shape at 75 phons must yield ~11 sones.
        let freqs = grid();
        let shape = Array1::<f64>::zeros(freqs.len());
        let erb = erb_positions(&freqs);
        let weights = erb_weights(&freqs);
        let loudness = approximate_loudness_sones(&shape, &erb, weights.as_ref(), 75.0);
        assert!(
            (loudness - 11.3).abs() < 3.0,
            "flat-75phon loudness should be ~11 sones, got {loudness:.2}"
        );
    }
}
