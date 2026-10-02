//! Frequency-dependent optimizer constraint envelopes and neutral diagnostics.
//!
//! O1 applies local-Q and composite-gain limits through initial guesses,
//! candidate evaluation, refinement, extraction, and Pareto selection.
//! O2 exposes neutral per-candidate diagnostics and outcome classes without
//! depending on RoomEQ types.
//!
//! Envelope knots reuse the established `(frequency_hz, bound_db)` convention
//! already carried by `ObjectiveData::max_boost_envelope`: linear
//! interpolation in log frequency with endpoint hold. The frozen K3 value
//! types from `autoeq-core` (G1) are not yet available at this HEAD, so this
//! module evaluates caller-supplied knots in that same convention instead of
//! defining a rival envelope type; see the handoff note in the lane report.
//! Absent envelopes keep legacy results: every entry point below is an exact
//! no-op (bit-identical parameters, no diagnostics) when no envelope is
//! configured.

// Rust guideline compliant 2026-02-21

use super::ObjectiveData;
use super::OptimizerRunEvidence;
use super::OptimizerTermination;
use super::misc::interpolate_boost_envelope;
use super::pareto::ParetoFilter;
use crate::LossType;
use crate::PeqModel;
use crate::iir::BiquadFilterType;
use crate::param_utils;
use crate::param_utils::PeqLayout;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Subdivisions per coarse optimization bin for the validated composite grid.
///
/// Eight sub-intervals resolve peaks far narrower than a coarse bin (a Q=12
/// filter is about 0.12 octaves wide at -3 dB) while keeping the post-hoc
/// check cheap. This grid is evaluation-only; it never steers the optimizer.
pub const VALIDATED_SUBDIVISIONS_PER_BIN: usize = 8;

/// Tolerance for the composite comparison in dB.
///
/// Covers floating-point realization noise around an exactly-at-bound
/// response. Genuine breaches in the fixtures exceed their bound by whole
/// decibels, so this tolerance cannot mask a real violation.
pub const COMPOSITE_COMPARISON_EPS_DB: f64 = 1e-9;

/// Which constraint a diagnostic refers to.
///
/// Cut kinds are breached when the observed value falls *below* the bound;
/// every other kind is breached when it rises *above* it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintKind {
    /// Per-filter boost above the gain envelope at its center.
    BoostEnvelope,
    /// Per-filter cut below the cut envelope at its center.
    CutEnvelope,
    /// Composite correction response above the boost envelope.
    CompositeBoost,
    /// Composite correction response below the cut envelope.
    CompositeCut,
    /// Filter Q above the local (frequency-dependent) cap.
    LocalQ,
    /// Filter Q above the global cap.
    GlobalQ,
}

/// Neutral diagnostic for one bound interaction of one candidate.
///
/// Carries the evaluated frequency or band, the observed value, the bound,
/// the constraint kind, and the candidate identity. It never claims physical
/// impossibility: see [`ClassifiedOutcome::implies_physical_impossibility`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ConstraintDiagnostic {
    /// Stable candidate identity from the nominating stage.
    pub candidate_id: String,
    /// Which constraint was evaluated.
    pub kind: ConstraintKind,
    /// Evaluated center or breach frequency in Hz, when localized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_hz: Option<f64>,
    /// Evaluated band in Hz, when the check is not a single point.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_hz: Option<(f64, f64)>,
    /// Observed value in the bound's units (dB for gains, linear for Q).
    pub observed: f64,
    /// Bound value in the same units.
    pub bound: f64,
}

impl ConstraintDiagnostic {
    /// Amount by which the observation exceeds the bound.
    ///
    /// Always non-negative; zero means compliant. Cut kinds measure the
    /// shortfall below the bound, every other kind the overshoot above it.
    pub fn excess(&self) -> f64 {
        match self.kind {
            ConstraintKind::CutEnvelope | ConstraintKind::CompositeCut => {
                (self.bound - self.observed).max(0.0)
            }
            _ => (self.observed - self.bound).max(0.0),
        }
    }

    /// Whether the observation breaches the bound beyond tolerance `eps`.
    pub fn is_binding(&self, eps: f64) -> bool {
        self.excess() > eps
    }
}

/// Validate envelope knots against the K3 shape rules.
///
/// Knots must be non-empty with finite positive strictly increasing Hz
/// frequencies and finite bounds. Q envelopes additionally require positive
/// bounds; gain envelopes allow any finite bound (cuts are negative).
///
/// # Errors
///
/// Returns a description when the knots are empty, non-finite, non-positive
/// in frequency, unordered, or (for Q) non-positive in bound.
pub fn validate_envelope_knots(
    knots: &[(f64, f64)],
    name: &str,
    bounds_must_be_positive: bool,
) -> Result<(), String> {
    if knots.is_empty() {
        return Err(format!("{name}: envelope needs at least one knot"));
    }
    let mut previous_hz = 0.0;
    for (index, &(freq_hz, bound)) in knots.iter().enumerate() {
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            return Err(format!(
                "{name}: knot {index} has a non-positive or non-finite frequency ({freq_hz})"
            ));
        }
        if freq_hz <= previous_hz {
            return Err(format!(
                "{name}: knot frequencies must be strictly increasing ({freq_hz} after {previous_hz})"
            ));
        }
        previous_hz = freq_hz;
        if !bound.is_finite() {
            return Err(format!("{name}: knot {index} has a non-finite bound"));
        }
        if bounds_must_be_positive && bound <= 0.0 {
            return Err(format!(
                "{name}: knot {index} has a non-positive Q cap ({bound})"
            ));
        }
    }
    Ok(())
}

/// Bound at a frequency in Hz.
///
/// Linear interpolation in log frequency with endpoint hold, shared with the
/// legacy per-filter gain clamps. An empty knot list means unbounded
/// (`+inf`), preserving the legacy convention.
pub fn envelope_bound_at(knots: &[(f64, f64)], freq_hz: f64) -> f64 {
    interpolate_boost_envelope(knots, freq_hz)
}

/// Whether a loss uses the PEQ parameter layout.
///
/// Driver and multi-sub layouts carry gains/delays/crossovers rather than
/// frequency/Q/gain triplets, so per-filter and composite PEQ envelope rules
/// do not apply to them. Mirrors the guard in `compute.rs`.
pub fn is_peq_layout_loss(loss: LossType) -> bool {
    !matches!(loss, LossType::DriversFlat | LossType::MultiSubFlat)
}

/// One per-filter gain projection onto its center-frequency envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GainAdjustment {
    /// Index of the adjusted filter.
    pub filter_index: usize,
    /// Decoded filter center in Hz.
    pub center_hz: f64,
    /// Gain before projection in dB.
    pub gain_before_db: f64,
    /// Gain after projection in dB.
    pub gain_after_db: f64,
    /// Envelope bound applied in dB.
    pub bound_db: f64,
    /// True for a boost clamp, false for a cut clamp.
    pub boost: bool,
}

impl GainAdjustment {
    /// Neutral diagnostic for this adjustment.
    pub fn diagnostic(&self, candidate_id: &str) -> ConstraintDiagnostic {
        ConstraintDiagnostic {
            candidate_id: String::from(candidate_id),
            kind: if self.boost {
                ConstraintKind::BoostEnvelope
            } else {
                ConstraintKind::CutEnvelope
            },
            frequency_hz: Some(self.center_hz),
            band_hz: None,
            observed: self.gain_before_db,
            bound: self.bound_db,
        }
    }
}

/// Project per-filter gains onto center-frequency envelopes.
///
/// Positive gains clamp to the boost envelope, negative gains to the cut
/// envelope, each evaluated at the filter's decoded center. Returns the
/// projected vector with one record per clamped filter. With no envelopes
/// (or a non-PEQ layout) the output is bit-identical to the input.
pub fn project_gains_onto_envelopes(
    x: &[f64],
    peq_model: PeqModel,
    loss: LossType,
    boost_knots: Option<&[(f64, f64)]>,
    cut_knots: Option<&[(f64, f64)]>,
) -> (Vec<f64>, Vec<GainAdjustment>) {
    if !is_peq_layout_loss(loss) || (boost_knots.is_none() && cut_knots.is_none()) {
        return (x.to_vec(), Vec::new());
    }
    let ppf = param_utils::params_per_filter(peq_model);
    if ppf == 0 || !x.len().is_multiple_of(ppf) {
        return (x.to_vec(), Vec::new());
    }
    let gain_idx = peq_model.layout().gain_idx;
    let mut projected = x.to_vec();
    let mut adjustments = Vec::new();
    let n = param_utils::num_filters(x, peq_model);
    for i in 0..n {
        let params = param_utils::get_filter_params(x, i, peq_model);
        let center_hz = 10f64.powf(params.freq);
        if !center_hz.is_finite() || center_hz <= 0.0 {
            continue;
        }
        let gain_offset = i * ppf + gain_idx;
        if params.gain > 0.0
            && let Some(knots) = boost_knots
        {
            let bound = envelope_bound_at(knots, center_hz);
            if params.gain > bound {
                projected[gain_offset] = bound;
                adjustments.push(GainAdjustment {
                    filter_index: i,
                    center_hz,
                    gain_before_db: params.gain,
                    gain_after_db: bound,
                    bound_db: bound,
                    boost: true,
                });
            }
        } else if params.gain < 0.0
            && let Some(knots) = cut_knots
        {
            let bound = envelope_bound_at(knots, center_hz);
            if params.gain < bound {
                projected[gain_offset] = bound;
                adjustments.push(GainAdjustment {
                    filter_index: i,
                    center_hz,
                    gain_before_db: params.gain,
                    gain_after_db: bound,
                    bound_db: bound,
                    boost: false,
                });
            }
        }
    }
    (projected, adjustments)
}

/// One local-Q projection at a decoded filter center.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct QAdjustment {
    /// Index of the adjusted filter.
    pub filter_index: usize,
    /// Decoded filter center in Hz.
    pub center_hz: f64,
    /// Q before projection.
    pub q_before: f64,
    /// Q after projection (the effective cap).
    pub q_after: f64,
    /// Global cap supplied by the caller.
    pub global_cap: f64,
    /// Local envelope cap at the center (`+inf` when unconfigured).
    pub local_cap: f64,
}

impl QAdjustment {
    /// Whether the global cap bound this filter rather than the local one.
    pub fn bound_by_global(&self) -> bool {
        self.global_cap <= self.local_cap
    }

    /// Neutral diagnostic reporting the binding bound.
    pub fn diagnostic(&self, candidate_id: &str) -> ConstraintDiagnostic {
        ConstraintDiagnostic {
            candidate_id: String::from(candidate_id),
            kind: if self.bound_by_global() {
                ConstraintKind::GlobalQ
            } else {
                ConstraintKind::LocalQ
            },
            frequency_hz: Some(self.center_hz),
            band_hz: None,
            observed: self.q_before,
            bound: self.q_after,
        }
    }
}

/// Decode the filter type without panicking on out-of-range free parameters.
///
/// `decode_filter_type` asserts its input range; an optimized free-filter
/// type value outside `[0, 12)` is already invalid input, so it is treated
/// conservatively as a peak (Q enforcement applies) instead of aborting.
fn peak_or_decoded(
    i: usize,
    n: usize,
    peq_model: PeqModel,
    type_param: Option<f64>,
) -> BiquadFilterType {
    if let Some(t) = type_param
        && (!t.is_finite() || !(0.0..12.0).contains(&t))
    {
        return BiquadFilterType::Peak;
    }
    param_utils::determine_filter_type(i, n, peq_model, type_param)
}

/// Enforce the stricter global/local Q bound at decoded filter centers.
///
/// Each peak filter's Q projects down to `min(global_max_q, local cap at its
/// center)`. Non-peak filters keep their values: shelf Q is unused by the
/// DSP core and HP/LP Q is governed by its own bounds, mirroring the
/// per-filter precedent in `viol_min_gain_from_xs`. Bass filters keep high Q
/// when actually placed in bass; only the center position matters, never the
/// width of the optimizer interval.
///
/// # Errors
///
/// Returns a description when the knots are invalid, `global_max_q` is not
/// finite-or-infinite positive, or the vector is not a whole number of
/// filters.
pub fn enforce_local_q_at_centers(
    x: &[f64],
    peq_model: PeqModel,
    loss: LossType,
    global_max_q: f64,
    local_q_knots: Option<&[(f64, f64)]>,
) -> Result<(Vec<f64>, Vec<QAdjustment>), String> {
    // NaN compares false against everything, so an explicit positive-or-infinite
    // check (rather than a negated comparison) keeps the intent readable.
    let capped = matches!(global_max_q.partial_cmp(&0.0), Some(Ordering::Greater))
        || global_max_q == f64::INFINITY;
    if !capped {
        return Err(format!(
            "global_max_q must be positive (got {global_max_q}); pass +inf for no global cap"
        ));
    }
    if let Some(knots) = local_q_knots {
        validate_envelope_knots(knots, "local_q", true)?;
    }
    if !is_peq_layout_loss(loss) {
        return Ok((x.to_vec(), Vec::new()));
    }
    if global_max_q.is_infinite() && local_q_knots.is_none() {
        return Ok((x.to_vec(), Vec::new()));
    }
    let ppf = param_utils::params_per_filter(peq_model);
    if ppf == 0 || !x.len().is_multiple_of(ppf) {
        return Err(format!(
            "parameter vector length {} is not a whole number of filters",
            x.len()
        ));
    }
    let q_idx = peq_model.layout().q_idx;
    let n = param_utils::num_filters(x, peq_model);
    let mut projected = x.to_vec();
    let mut adjustments = Vec::new();
    for i in 0..n {
        let params = param_utils::get_filter_params(x, i, peq_model);
        if peak_or_decoded(i, n, peq_model, params.filter_type) != BiquadFilterType::Peak {
            continue;
        }
        let center_hz = 10f64.powf(params.freq);
        if !center_hz.is_finite() || center_hz <= 0.0 {
            continue;
        }
        let local_cap =
            local_q_knots.map_or(f64::INFINITY, |knots| envelope_bound_at(knots, center_hz));
        let effective = global_max_q.min(local_cap);
        if params.q > effective {
            projected[i * ppf + q_idx] = effective;
            adjustments.push(QAdjustment {
                filter_index: i,
                center_hz,
                q_before: params.q,
                q_after: effective,
                global_cap: global_max_q,
                local_cap,
            });
        }
    }
    Ok((projected, adjustments))
}

/// Enforce a frozen K3 local-Q envelope at decoded filter centers.
///
/// Thin adapter over [`enforce_local_q_at_centers`] evaluating the canonical
/// G1 K3 envelope instead of raw knot slices. The effective bound is the
/// stricter of `global_max_q` and the envelope value at each peak filter
/// center; non-peak filters keep their values. Unlike
/// `autoeq_core::constraint_envelope::effective_max_q`, an infinite global
/// cap is allowed and means envelope-only enforcement.
///
/// # Errors
///
/// Returns a description for a NaN or non-positive global cap, a malformed
/// parameter vector, or an invalid envelope or query frequency.
pub fn enforce_local_q_envelope_at_centers(
    x: &[f64],
    peq_model: PeqModel,
    loss: LossType,
    global_max_q: f64,
    envelope: Option<&crate::core::constraint_envelope::LocalQEnvelope>,
) -> Result<(Vec<f64>, Vec<QAdjustment>), String> {
    let capped = matches!(global_max_q.partial_cmp(&0.0), Some(Ordering::Greater))
        || global_max_q == f64::INFINITY;
    if !capped {
        return Err(format!(
            "global_max_q must be positive (got {global_max_q}); pass +inf for no global cap"
        ));
    }
    if let Some(envelope) = envelope {
        envelope
            .validate("local-Q envelope")
            .map_err(|error| error.to_string())?;
    }
    if !is_peq_layout_loss(loss) {
        return Ok((x.to_vec(), Vec::new()));
    }
    if global_max_q.is_infinite() && envelope.is_none() {
        return Ok((x.to_vec(), Vec::new()));
    }
    let ppf = param_utils::params_per_filter(peq_model);
    if ppf == 0 || !x.len().is_multiple_of(ppf) {
        return Err(format!(
            "parameter vector length {} is not a whole number of filters",
            x.len()
        ));
    }
    let q_idx = peq_model.layout().q_idx;
    let n = param_utils::num_filters(x, peq_model);
    let mut projected = x.to_vec();
    let mut adjustments = Vec::new();
    for i in 0..n {
        let params = param_utils::get_filter_params(x, i, peq_model);
        if peak_or_decoded(i, n, peq_model, params.filter_type) != BiquadFilterType::Peak {
            continue;
        }
        let center_hz = 10f64.powf(params.freq);
        if !center_hz.is_finite() || center_hz <= 0.0 {
            continue;
        }
        let local_cap = match envelope {
            Some(envelope) => envelope
                .max_q_at_freq(center_hz)
                .map_err(|error| error.to_string())?,
            None => f64::INFINITY,
        };
        let effective = global_max_q.min(local_cap);
        if params.q > effective {
            projected[i * ppf + q_idx] = effective;
            adjustments.push(QAdjustment {
                filter_index: i,
                center_hz,
                q_before: params.q,
                q_after: effective,
                global_cap: global_max_q,
                local_cap,
            });
        }
    }
    Ok((projected, adjustments))
}

/// Build the validated composite grid for one correction band.
///
/// Each coarse interval inside `[min_freq, max_freq]` gains
/// `subdivisions_per_bin - 1` geometric interior points, so response extrema
/// between coarse optimization bins are evaluated rather than assumed away.
/// Output is strictly increasing.
///
/// # Errors
///
/// Returns a description when the band or subdivision count is invalid, the
/// coarse grid has fewer than two in-band points, or it is not ascending.
pub fn validated_composite_grid(
    coarse_freqs: &[f64],
    min_freq: f64,
    max_freq: f64,
    subdivisions_per_bin: usize,
) -> Result<Vec<f64>, String> {
    if subdivisions_per_bin == 0 {
        return Err(String::from(
            "composite grid needs at least one subdivision per bin",
        ));
    }
    if !(min_freq.is_finite() && max_freq.is_finite() && min_freq > 0.0 && min_freq < max_freq) {
        return Err(format!(
            "composite grid needs a finite positive band (got {min_freq}..{max_freq})"
        ));
    }
    let in_band: Vec<f64> = coarse_freqs
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f >= min_freq && *f <= max_freq)
        .collect();
    if in_band.len() < 2 {
        return Err(String::from(
            "composite grid needs at least two in-band coarse points",
        ));
    }
    for pair in in_band.windows(2) {
        if pair[1] <= pair[0] {
            return Err(String::from(
                "composite grid needs a strictly ascending coarse grid",
            ));
        }
    }
    let mut dense = Vec::with_capacity(in_band.len() * subdivisions_per_bin);
    for pair in in_band.windows(2) {
        let (low, high) = (pair[0], pair[1]);
        dense.push(low);
        let ratio = high / low;
        for k in 1..subdivisions_per_bin {
            dense.push(low * ratio.powf(k as f64 / subdivisions_per_bin as f64));
        }
    }
    dense.push(*in_band.last().expect("in-band grid is non-empty"));
    Ok(dense)
}

/// One composite response breach of a gain envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CompositeBreach {
    /// Breach frequency in Hz on the validated grid.
    pub frequency_hz: f64,
    /// Composite correction response in dB.
    pub observed_db: f64,
    /// Envelope bound at the breach frequency in dB.
    pub bound_db: f64,
    /// True for a boost breach, false for a cut breach.
    pub boost: bool,
}

impl CompositeBreach {
    /// Neutral diagnostic reporting the binding bound.
    pub fn diagnostic(&self, candidate_id: &str) -> ConstraintDiagnostic {
        ConstraintDiagnostic {
            candidate_id: String::from(candidate_id),
            kind: if self.boost {
                ConstraintKind::CompositeBoost
            } else {
                ConstraintKind::CompositeCut
            },
            frequency_hz: Some(self.frequency_hz),
            band_hz: None,
            observed: self.observed_db,
            bound: self.bound_db,
        }
    }
}

/// Check the composite correction response against gain envelopes.
///
/// Renders the full realized response on the validated dense grid and
/// records every point outside the envelopes, including extrema between
/// coarse optimization bins that per-filter clamps cannot see (stacked
/// filters sum past any per-filter bound). Non-finite responses breach on
/// the configured side rather than passing silently. With no envelopes (or
/// a non-PEQ layout) the result is empty.
///
/// # Errors
///
/// Returns a description when the knots or grid are invalid.
#[allow(clippy::too_many_arguments)]
pub fn check_composite_gain_envelope(
    x: &[f64],
    coarse_freqs: &[f64],
    srate: f64,
    peq_model: PeqModel,
    loss: LossType,
    min_freq: f64,
    max_freq: f64,
    boost_knots: Option<&[(f64, f64)]>,
    cut_knots: Option<&[(f64, f64)]>,
    subdivisions_per_bin: usize,
) -> Result<Vec<CompositeBreach>, String> {
    if !is_peq_layout_loss(loss) || (boost_knots.is_none() && cut_knots.is_none()) {
        return Ok(Vec::new());
    }
    if let Some(knots) = boost_knots {
        validate_envelope_knots(knots, "composite_boost", false)?;
    }
    if let Some(knots) = cut_knots {
        validate_envelope_knots(knots, "composite_cut", false)?;
    }
    let grid = validated_composite_grid(coarse_freqs, min_freq, max_freq, subdivisions_per_bin)?;
    if !srate.is_finite() || srate <= 0.0 {
        return Err(String::from(
            "composite check needs a finite positive sample rate",
        ));
    }
    let response = crate::x2peq::x2spl(
        &ndarray::Array1::from_vec(grid.clone()),
        x,
        srate,
        peq_model,
    );
    let mut breaches = Vec::new();
    for (freq_hz, observed) in grid.iter().zip(response.iter()) {
        if observed.is_finite() {
            if let Some(knots) = boost_knots {
                let bound = envelope_bound_at(knots, *freq_hz);
                if *observed > bound + COMPOSITE_COMPARISON_EPS_DB {
                    breaches.push(CompositeBreach {
                        frequency_hz: *freq_hz,
                        observed_db: *observed,
                        bound_db: bound,
                        boost: true,
                    });
                    continue;
                }
            }
            if let Some(knots) = cut_knots {
                let bound = envelope_bound_at(knots, *freq_hz);
                if *observed < bound - COMPOSITE_COMPARISON_EPS_DB {
                    breaches.push(CompositeBreach {
                        frequency_hz: *freq_hz,
                        observed_db: *observed,
                        bound_db: bound,
                        boost: false,
                    });
                }
            }
        } else if *observed == f64::INFINITY || observed.is_nan() {
            if let Some(knots) = boost_knots.or(cut_knots) {
                breaches.push(CompositeBreach {
                    frequency_hz: *freq_hz,
                    observed_db: *observed,
                    bound_db: envelope_bound_at(knots, *freq_hz),
                    boost: true,
                });
            }
        } else if let Some(knots) = cut_knots.or(boost_knots) {
            breaches.push(CompositeBreach {
                frequency_hz: *freq_hz,
                observed_db: *observed,
                bound_db: envelope_bound_at(knots, *freq_hz),
                boost: false,
            });
        }
    }
    Ok(breaches)
}

/// Caller-supplied constraint envelope configuration.
///
/// Knots follow the `(frequency_hz, bound)` convention; composite checks
/// fall back to the objective's own gain envelopes when unset here, so an
/// absent envelope everywhere keeps legacy results. Borrows its knots: it
/// carries no rival envelope value type.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConstraintSpec<'a> {
    /// Global Q cap applied with the local envelope (`+inf` disables).
    pub global_max_q: f64,
    /// Local Q caps as `(frequency_hz, max_q)` knots.
    pub local_q_knots: Option<&'a [(f64, f64)]>,
    /// Composite boost limit knots (defaults to the objective envelope).
    pub boost_knots: Option<&'a [(f64, f64)]>,
    /// Composite cut limit knots (defaults to the objective envelope).
    pub cut_knots: Option<&'a [(f64, f64)]>,
    /// Dense-grid subdivisions per coarse bin.
    pub subdivisions_per_bin: usize,
}

impl<'a> ConstraintSpec<'a> {
    /// Unconstrained spec: absent envelopes keep legacy results.
    pub fn unconstrained() -> Self {
        Self {
            global_max_q: f64::INFINITY,
            local_q_knots: None,
            boost_knots: None,
            cut_knots: None,
            subdivisions_per_bin: VALIDATED_SUBDIVISIONS_PER_BIN,
        }
    }

    /// Check knot shapes, the Q cap, and the subdivision count.
    ///
    /// # Errors
    ///
    /// Returns a description when any configured envelope is invalid.
    pub fn validate(&self) -> Result<(), String> {
        let capped = matches!(self.global_max_q.partial_cmp(&0.0), Some(Ordering::Greater))
            || self.global_max_q == f64::INFINITY;
        if !capped {
            return Err(String::from(
                "constraint spec needs a positive global_max_q or +inf",
            ));
        }
        if let Some(knots) = self.local_q_knots {
            validate_envelope_knots(knots, "local_q", true)?;
        }
        if let Some(knots) = self.boost_knots {
            validate_envelope_knots(knots, "spec_boost", false)?;
        }
        if let Some(knots) = self.cut_knots {
            validate_envelope_knots(knots, "spec_cut", false)?;
        }
        if self.subdivisions_per_bin == 0 {
            return Err(String::from(
                "constraint spec needs at least one subdivision per bin",
            ));
        }
        Ok(())
    }
}

/// One candidate after the shared envelope choke-point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ConstrainedCandidate {
    /// Stable candidate identity from the nominating stage.
    pub candidate_id: String,
    /// Envelope-projected parameter vector.
    pub params: Vec<f64>,
    /// Per-filter gain projections applied.
    pub gain_adjustments: Vec<GainAdjustment>,
    /// Local-Q projections applied.
    pub q_adjustments: Vec<QAdjustment>,
    /// Composite breaches remaining after projection (stacked filters).
    pub composite_breaches: Vec<CompositeBreach>,
    /// True when no composite breach remains.
    pub feasible: bool,
}

impl ConstrainedCandidate {
    /// Neutral diagnostics for every adjustment and breach.
    pub fn diagnostics(&self) -> Vec<ConstraintDiagnostic> {
        let mut out = Vec::with_capacity(
            self.gain_adjustments.len() + self.q_adjustments.len() + self.composite_breaches.len(),
        );
        for adjustment in &self.gain_adjustments {
            out.push(adjustment.diagnostic(&self.candidate_id));
        }
        for adjustment in &self.q_adjustments {
            out.push(adjustment.diagnostic(&self.candidate_id));
        }
        for breach in &self.composite_breaches {
            out.push(breach.diagnostic(&self.candidate_id));
        }
        out
    }
}

/// Run one candidate through the shared envelope choke-point.
///
/// Applies the same rules to every candidate source (initial guesses,
/// seeds, refinement outputs, Pareto members): per-filter gain projection,
/// local-Q projection at decoded centers, then the composite check on the
/// validated grid. Per-filter fixes are applied; composite breaches are
/// reported as infeasibility evidence rather than silently repaired.
/// Absent envelopes return the input bit-identical and feasible.
///
/// # Errors
///
/// Returns a description when the spec, knots, grid, or vector shape is
/// invalid, or for non-PEQ layouts (which pass through untouched, so this
/// only errors on malformed PEQ input).
pub fn constrain_candidate(
    candidate_id: &str,
    x: &[f64],
    data: &ObjectiveData,
    spec: &ConstraintSpec<'_>,
) -> Result<ConstrainedCandidate, String> {
    spec.validate()?;
    if let Some(index) = x.iter().position(|value| !value.is_finite()) {
        return Err(format!(
            "candidate `{candidate_id}` has a nonfinite parameter at index {index}"
        ));
    }
    if !is_peq_layout_loss(data.loss_type) {
        return Ok(ConstrainedCandidate {
            candidate_id: String::from(candidate_id),
            params: x.to_vec(),
            gain_adjustments: Vec::new(),
            q_adjustments: Vec::new(),
            composite_breaches: Vec::new(),
            feasible: true,
        });
    }
    let (params, gain_adjustments) = project_gains_onto_envelopes(
        x,
        data.peq_model,
        data.loss_type,
        data.max_boost_envelope.as_deref(),
        data.min_cut_envelope.as_deref(),
    );
    let (params, q_adjustments) = enforce_local_q_at_centers(
        &params,
        data.peq_model,
        data.loss_type,
        spec.global_max_q,
        spec.local_q_knots,
    )?;
    let boost = spec.boost_knots.or(data.max_boost_envelope.as_deref());
    let cut = spec.cut_knots.or(data.min_cut_envelope.as_deref());
    let composite_breaches = check_composite_gain_envelope(
        &params,
        data.freqs.as_slice().ok_or_else(|| {
            String::from("constrain_candidate needs a contiguous objective frequency grid")
        })?,
        data.srate,
        data.peq_model,
        data.loss_type,
        data.min_freq,
        data.max_freq,
        boost,
        cut,
        spec.subdivisions_per_bin,
    )?;
    Ok(ConstrainedCandidate {
        candidate_id: String::from(candidate_id),
        params,
        gain_adjustments,
        q_adjustments,
        feasible: composite_breaches.is_empty(),
        composite_breaches,
    })
}

/// Encode a [`FrequencyQPolicy`](super::params::FrequencyQPolicy) as
/// center-evaluation knots for [`enforce_local_q_at_centers`].
///
/// Bound-time conservatism caps whole frequency intervals; center
/// evaluation caps each filter at its own center. The faithful center
/// rule steps at the lower hinge: centers below it cap low, centers at or
/// above it cap high, because centers at or above the lower hinge can
/// reach the guarded band. Single-sided policies step against the finite
/// global cap. The step is a one-ulp transition: knot validation needs
/// strictly increasing frequencies, and a one-ulp sliver interpolates
/// instead of stepping only for centers inside it.
///
/// Returns `None` when the policy declares nothing (global cap only).
///
/// # Errors
///
/// Returns a description when a declared hinge or cap is incoherent, or
/// when a single-sided policy meets an infinite global cap it cannot step
/// against: dropping a declared cap silently would be worse than refusing.
pub fn policy_local_q_knots(
    policy: &super::params::FrequencyQPolicy,
    global_max_q: f64,
) -> Result<Option<Vec<(f64, f64)>>, String> {
    let hinge = |freq: Option<f64>,
                 cap: Option<f64>,
                 name: &str|
     -> Result<Option<(f64, f64)>, String> {
        match (freq, cap) {
            (None, None) => Ok(None),
            (Some(frequency), Some(limit)) => {
                if !frequency.is_finite() || frequency <= 0.0 || !limit.is_finite() || limit <= 0.0
                {
                    return Err(format!(
                        "local-Q policy {name} needs a finite positive hinge and cap"
                    ));
                }
                Ok(Some((frequency, limit)))
            }
            _ => Err(format!(
                "local-Q policy {name} needs both hinge and cap, or neither"
            )),
        }
    };
    let low = hinge(policy.schroeder_hz, policy.low_max_q, "low")?;
    let high = hinge(policy.high_start_hz, policy.high_max_q, "high")?;
    let step = |hinge_hz: f64, below: f64, above: f64| -> Result<Vec<(f64, f64)>, String> {
        let knots = vec![(hinge_hz, below), (hinge_hz.next_up(), above)];
        validate_envelope_knots(&knots, "local_q", true)?;
        Ok(knots)
    };
    match (low, high) {
        (None, None) => Ok(None),
        (Some((schroeder_hz, low_max_q)), Some((high_start_hz, high_max_q))) => {
            step(schroeder_hz.min(high_start_hz), low_max_q, high_max_q).map(Some)
        }
        (Some((schroeder_hz, low_max_q)), None) => {
            if !global_max_q.is_finite() || global_max_q <= 0.0 {
                return Err(String::from(
                    "local-Q policy declares only a low cap under an unbounded global cap: no faithful knots exist",
                ));
            }
            step(schroeder_hz, low_max_q, global_max_q).map(Some)
        }
        (None, Some((high_start_hz, high_max_q))) => {
            if !global_max_q.is_finite() || global_max_q <= 0.0 {
                return Err(String::from(
                    "local-Q policy declares only a high cap under an unbounded global cap: no faithful knots exist",
                ));
            }
            step(high_start_hz, global_max_q, high_max_q).map(Some)
        }
    }
}

/// Owned constraint spec built from optimizer params and objective envelopes.
///
/// [`ConstraintSpec`] borrows its knots; this owner carries the
/// policy-derived local-Q knots so dispatchers and engine emission share
/// one spec-building path instead of drifting apart.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OwnedConstraintSpec {
    /// Global Q cap applied with the local envelope (`+inf` disables).
    pub global_max_q: f64,
    /// Local Q caps as owned `(frequency_hz, max_q)` knots.
    pub local_q_knots: Option<Vec<(f64, f64)>>,
    /// Dense-grid subdivisions per coarse bin.
    pub subdivisions_per_bin: usize,
}

impl OwnedConstraintSpec {
    /// Build the enforcement spec from optimizer params.
    ///
    /// The global cap is `params.max_q`; local caps come from the
    /// frequency-Q policy when declared. Composite knots fall back to the
    /// objective's own gain envelopes inside [`constrain_candidate`], so an
    /// absent envelope everywhere keeps legacy results.
    ///
    /// # Errors
    ///
    /// Returns a description when the global cap or the policy knots are
    /// incoherent.
    pub fn from_params(params: &crate::OptimParams) -> Result<Self, String> {
        if params.max_q.is_nan() || params.max_q <= 0.0 {
            return Err(format!(
                "constraint spec needs a finite positive global_max_q or +inf (got {})",
                params.max_q
            ));
        }
        let local_q_knots = match &params.frequency_q_policy {
            Some(policy) => policy_local_q_knots(policy, params.max_q)?,
            None => None,
        };
        Ok(Self {
            global_max_q: params.max_q,
            local_q_knots,
            subdivisions_per_bin: VALIDATED_SUBDIVISIONS_PER_BIN,
        })
    }

    /// Borrow this owner as a [`ConstraintSpec`].
    pub fn as_spec(&self) -> ConstraintSpec<'_> {
        ConstraintSpec {
            global_max_q: self.global_max_q,
            local_q_knots: self.local_q_knots.as_deref(),
            boost_knots: None,
            cut_knots: None,
            subdivisions_per_bin: self.subdivisions_per_bin,
        }
    }
}

/// One candidate after shared finalization.
///
/// `params` are the repaired parameters actually emitted; `loss` is the
/// scalar fitness re-verified at those parameters, never the stale winner
/// loss. `constrained` carries the per-filter adjustments and the
/// feasibility evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalizedCandidate {
    /// Candidate identity carried through finalization.
    pub candidate_id: String,
    /// Repaired parameters actually emitted.
    pub params: Vec<f64>,
    /// Scalar fitness re-verified at the repaired parameters.
    pub loss: f64,
    /// Projection adjustments and feasibility evidence.
    pub constrained: ConstrainedCandidate,
}

/// Finalize one optimizer winner through the shared envelope choke-point.
///
/// Gains project onto their envelopes, Q projects onto the global and
/// local caps, and the composite response is checked on the validated
/// grid. Repairs re-verify the scalar fitness at the repaired parameters;
/// composite breaches refuse with the breach evidence instead of emitting
/// a violating candidate under a stale loss.
///
/// # Errors
///
/// Returns the refusal reason when envelopes are incoherent or the
/// repaired response still breaches the composite envelope.
pub fn finalize_candidate(
    candidate_id: &str,
    x: &[f64],
    data: &ObjectiveData,
    spec: &ConstraintSpec<'_>,
) -> Result<FinalizedCandidate, String> {
    let constrained = constrain_candidate(candidate_id, x, data, spec)?;
    if !constrained.feasible {
        let mut breaches: Vec<String> = constrained
            .composite_breaches
            .iter()
            .take(4)
            .map(|breach| {
                format!(
                    "{:.1} Hz: {:.2} dB vs {:.2} dB {}",
                    breach.frequency_hz,
                    breach.observed_db,
                    breach.bound_db,
                    if breach.boost { "boost" } else { "cut" }
                )
            })
            .collect();
        if constrained.composite_breaches.len() > breaches.len() {
            breaches.push(format!(
                "and {} more",
                constrained.composite_breaches.len() - breaches.len()
            ));
        }
        return Err(format!(
            "candidate '{candidate_id}' refused: composite gain envelope breached at {}",
            breaches.join("; ")
        ));
    }
    let loss = super::compute::compute_fitness_penalties_ref(&constrained.params, data);
    Ok(FinalizedCandidate {
        candidate_id: String::from(candidate_id),
        params: constrained.params.clone(),
        loss,
        constrained,
    })
}

/// Feasibility of one Pareto front member under the shared rules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ParetoFeasibility {
    /// Front position identity (`pareto-{index}`).
    pub candidate_id: String,
    /// Filter count of the member.
    pub num_filters: usize,
    /// True when no composite breach remains.
    pub feasible: bool,
    /// Neutral diagnostics for adjustments and breaches.
    pub diagnostics: Vec<ConstraintDiagnostic>,
}

/// Check every Pareto front member with the shared choke-point.
///
/// Members keep their losses and ranking; this only attaches feasibility so
/// selection can prefer feasible members without hiding the front.
///
/// # Errors
///
/// Returns a description when the spec or any member vector is invalid.
pub fn check_pareto_feasibility(
    front: &[ParetoFilter],
    data: &ObjectiveData,
    spec: &ConstraintSpec<'_>,
) -> Result<Vec<ParetoFeasibility>, String> {
    front
        .iter()
        .enumerate()
        .map(|(index, member)| {
            let candidate =
                constrain_candidate(&format!("pareto-{index}"), &member.params, data, spec)?;
            Ok(ParetoFeasibility {
                candidate_id: candidate.candidate_id.clone(),
                num_filters: member.num_filters,
                feasible: candidate.feasible,
                diagnostics: candidate.diagnostics(),
            })
        })
        .collect()
}

/// One Pareto member judged feasible through the shared choke-point.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgedParetoMember {
    /// Position within the submitted front.
    pub index: usize,
    /// Repaired parameters (bit-identical input when already feasible).
    pub params: Vec<f64>,
    /// Per-axis objectives re-evaluated at `params`, so selection judges
    /// the score the repaired candidate actually keeps.
    pub objectives: Vec<f64>,
}

/// Outcome of judging one submitted Pareto front before selection.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgedParetoFront {
    /// Feasible members in submitted order.
    pub members: Vec<JudgedParetoMember>,
    /// Submitted member count.
    pub submitted: usize,
    /// Refused (infeasible) member count.
    pub refused: usize,
}

/// Judge every submitted Pareto vector through the shared choke-point
/// before selection.
///
/// Feasible members carry the repaired parameters with per-axis objectives
/// re-evaluated at those parameters. Infeasible members are refused and
/// excluded from selection; an empty judged front is an explicit error, not
/// a silent fallback. Non-PEQ layouts pass through untouched (feasible),
/// matching [`constrain_candidate`].
///
/// # Errors
///
/// Returns a description when the spec or a member vector is invalid, or
/// when no member survives the envelopes.
pub fn judge_pareto_members(
    backend_name: &str,
    xs: &[Vec<f64>],
    data: &ObjectiveData,
    spec: &ConstraintSpec<'_>,
) -> Result<JudgedParetoFront, String> {
    let mut members = Vec::with_capacity(xs.len());
    let mut refused = 0_usize;
    for (index, raw) in xs.iter().enumerate() {
        let constrained = constrain_candidate(&format!("{backend_name}-{index}"), raw, data, spec)?;
        if !constrained.feasible {
            refused += 1;
            continue;
        }
        let objectives = super::compute::compute_pareto_objectives(&constrained.params, data);
        members.push(JudgedParetoMember {
            index,
            params: constrained.params,
            objectives,
        });
    }
    if members.is_empty() {
        return Err(format!(
            "{backend_name} refused all {} front member(s): no feasible candidate under the envelopes",
            xs.len()
        ));
    }
    Ok(JudgedParetoFront {
        members,
        submitted: xs.len(),
        refused,
    })
}

/// One out-of-budget joint control (gain, delay, crossover, or all-pass
/// parameter) at acceptance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BudgetBreach {
    /// Position within the candidate vector.
    pub index: usize,
    /// Emitted value.
    pub observed: f64,
    /// Configured lower budget.
    pub lower: f64,
    /// Configured upper budget.
    pub upper: f64,
}

/// Verify a joint/gain candidate against its configured budgets at
/// acceptance: every control finite and inside its bound.
///
/// PEQ layouts are judged by [`constrain_candidate`]; driver, multi-sub,
/// delay, and all-pass layouts carry gains/delays/crossovers rather than
/// filter triplets, so acceptance checks the budgets the search was given.
/// Refuses with the breaching controls instead of emitting an out-of-budget
/// winner. Backend searches must respect their bounds; a breach reports
/// backend misbehavior, never a verdict about the room.
///
/// # Errors
///
/// Returns a description when the vector and bounds disagree in length, a
/// control is non-finite, or any control leaves its budget.
pub fn verify_joint_budgets(
    candidate_id: &str,
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Result<(), String> {
    if x.len() != lower.len() || x.len() != upper.len() {
        return Err(format!(
            "candidate '{candidate_id}' budget check needs matching lengths: x={}, lower={}, upper={}",
            x.len(),
            lower.len(),
            upper.len()
        ));
    }
    let mut breaches: Vec<BudgetBreach> = Vec::new();
    for index in 0..x.len() {
        let (value, lo, hi) = (x[index], lower[index], upper[index]);
        if !value.is_finite() || !lo.is_finite() || !hi.is_finite() || value < lo || value > hi {
            breaches.push(BudgetBreach {
                index,
                observed: value,
                lower: lo,
                upper: hi,
            });
        }
    }
    if breaches.is_empty() {
        return Ok(());
    }
    let detail = breaches
        .iter()
        .take(4)
        .map(|breach| {
            format!(
                "[{}]: {:.4} outside [{:.4}, {:.4}]",
                breach.index, breach.observed, breach.lower, breach.upper
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let more = if breaches.len() > 4 {
        format!(" and {} more", breaches.len() - 4)
    } else {
        String::new()
    };
    Err(format!(
        "candidate '{candidate_id}' refused: {} joint budget breach(es): {detail}{more}",
        breaches.len()
    ))
}

/// Outcome of one bounded optimization run.
///
/// `NoFeasibleCandidate` means no candidate satisfied the envelopes; it is
/// an optimizer-side statement about the search, never a verdict that the
/// room is physically uncorrectable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationOutcomeKind {
    /// A feasible candidate within all envelopes.
    BoundedSuccess,
    /// No feasible candidate found under the envelopes.
    NoFeasibleCandidate,
    /// The evaluation budget ran out before convergence.
    BudgetExhausted,
    /// The optimizer stopped without converging.
    ConvergenceFailure,
}

/// Classified outcome pairing optimizer evidence with envelope feasibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ClassifiedOutcome {
    /// Outcome class.
    pub kind: OptimizationOutcomeKind,
    /// Stable candidate identity.
    pub candidate_id: String,
    /// Neutral diagnostics for adjustments and breaches.
    pub diagnostics: Vec<ConstraintDiagnostic>,
    /// Backend status and objective carried over for traceability.
    pub detail: String,
}

impl ClassifiedOutcome {
    /// Whether this outcome rules out physical correction.
    ///
    /// Always false: an optimizer outcome only describes the search (budget,
    /// convergence, envelope compliance). Declaring a room feature
    /// uncorrectable or prescribing placement needs measurement evidence the
    /// optimizer never sees.
    pub fn implies_physical_impossibility(&self) -> bool {
        false
    }
}

/// Classify one candidate from backend evidence and envelope feasibility.
///
/// A converged-but-infeasible candidate is `NoFeasibleCandidate`, never
/// success: the optimizer status alone cannot bless an envelope breach.
/// Feasible candidates follow the termination evidence (budget,
/// convergence, or bounded success) with existing status text preserved in
/// `detail`.
///
/// # Examples
///
/// ```rust
/// use autoeq_optim::optim::{
///     ClassifiedOutcome, ConstrainedCandidate, OptimizationOutcomeKind,
///     OptimizerRunEvidence, OptimizerTermination, classify_outcome,
/// };
///
/// let evidence = OptimizerRunEvidence::from_backend_result(
///     "autoeq:cobyla",
///     Ok(("converged".to_string(), 0.5)),
///     &[0.5],
///     &[0.0],
///     &[1.0],
///     50,
///     Some(3),
/// );
/// assert_eq!(evidence.termination, OptimizerTermination::Converged);
/// let candidate = ConstrainedCandidate {
///     candidate_id: String::from("seed-0"),
///     params: vec![0.5],
///     gain_adjustments: Vec::new(),
///     q_adjustments: Vec::new(),
///     composite_breaches: Vec::new(),
///     feasible: true,
/// };
/// let outcome = classify_outcome(&evidence, &candidate);
/// assert_eq!(outcome.kind, OptimizationOutcomeKind::BoundedSuccess);
/// assert!(!outcome.implies_physical_impossibility());
/// ```
pub fn classify_outcome(
    evidence: &OptimizerRunEvidence,
    candidate: &ConstrainedCandidate,
) -> ClassifiedOutcome {
    let detail = format!(
        "backend '{}' reported '{}' with objective {:?}; envelope feasible: {}",
        evidence.algorithm, evidence.status, evidence.objective, candidate.feasible
    );
    if !candidate.feasible {
        return ClassifiedOutcome {
            kind: OptimizationOutcomeKind::NoFeasibleCandidate,
            candidate_id: candidate.candidate_id.clone(),
            diagnostics: candidate.diagnostics(),
            detail,
        };
    }
    let kind = match evidence.termination {
        OptimizerTermination::Converged => OptimizationOutcomeKind::BoundedSuccess,
        OptimizerTermination::EvaluationLimit => OptimizationOutcomeKind::BudgetExhausted,
        OptimizerTermination::NonConverged
        | OptimizerTermination::BackendFailure
        | OptimizerTermination::InvalidResult
        | OptimizerTermination::UserStopped => OptimizationOutcomeKind::ConvergenceFailure,
    };
    ClassifiedOutcome {
        kind,
        candidate_id: candidate.candidate_id.clone(),
        diagnostics: candidate.diagnostics(),
        detail,
    }
}

#[cfg(test)]
mod constraint_envelope_tests {
    use super::*;
    use crate::optim::ObjectiveDataBuilder;
    use ndarray::Array1;

    fn flat_objective(freqs: Vec<f64>) -> ObjectiveData {
        let freqs = Array1::from_vec(freqs);
        let n = freqs.len();
        ObjectiveDataBuilder::new(
            freqs,
            Array1::zeros(n),
            Array1::from_elem(n, 5.0),
            48_000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
        )
        .max_db(12.0)
        .min_db(0.0)
        .freq_range(20.0, 20_000.0)
        .build()
        .expect("valid test objective")
    }

    fn log_grid() -> Vec<f64> {
        Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 200).to_vec()
    }

    fn evidence_for(
        result: Result<(String, f64), (String, f64)>,
        x: &[f64],
    ) -> OptimizerRunEvidence {
        // Wide matching bounds: the termination under test comes from the
        // status text, never from a bound violation.
        let lower = vec![-1.0e300; x.len()];
        let upper = vec![1.0e300; x.len()];
        OptimizerRunEvidence::from_backend_result(
            "autoeq:cobyla",
            result,
            x,
            &lower,
            &upper,
            50,
            Some(3),
        )
    }

    #[test]
    fn all_candidate_layouts_reject_nonfinite_parameters_without_envelopes() {
        for loss_type in [
            LossType::SpeakerFlat,
            LossType::DriversFlat,
            LossType::MultiSubFlat,
        ] {
            let mut data = flat_objective(log_grid());
            data.loss_type = loss_type;
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let error = constrain_candidate(
                    "invalid",
                    &[3.0, 1.0, value],
                    &data,
                    &ConstraintSpec::unconstrained(),
                )
                .unwrap_err();
                assert!(error.contains("nonfinite parameter at index 2"), "{error}");
            }
        }
    }

    #[test]
    fn optim_local_q_preserves_bass_and_caps_treble() {
        // Explicit two-cap fixture at an arbitrary 100/1000 Hz hinge: not a
        // production Schroeder crossover, so the test pins the mechanism
        // rather than a hard-coded band split.
        let knots = vec![(100.0, 10.0), (1000.0, 1.0)];
        // Log-frequency interpolation is exact at the geometric mid-hinge.
        assert!((envelope_bound_at(&knots, 316.227_766_016_837_9) - 5.5).abs() < 1e-9);
        let x = vec![60.0_f64.log10(), 8.0, -6.0, 6000.0_f64.log10(), 8.0, -6.0];
        let (out, adjustments) =
            enforce_local_q_at_centers(&x, PeqModel::Pk, LossType::SpeakerFlat, 12.0, Some(&knots))
                .expect("valid knots");
        // Bass filter keeps its high Q: the treble guard must not leak down.
        assert_eq!(out[1], 8.0);
        // Treble filter is capped at the local bound 1.0.
        assert_eq!(out[4], 1.0);
        assert_eq!(adjustments.len(), 1);
        assert_eq!(adjustments[0].filter_index, 1);
        assert!((adjustments[0].local_cap - 1.0).abs() < 1e-12);
        assert!(!adjustments[0].bound_by_global());

        // A stricter global cap binds everywhere, including bass.
        let (out, adjustments) =
            enforce_local_q_at_centers(&x, PeqModel::Pk, LossType::SpeakerFlat, 0.5, Some(&knots))
                .expect("valid knots");
        assert_eq!(out[1], 0.5);
        assert_eq!(out[4], 0.5);
        assert!(adjustments.iter().all(|a| a.bound_by_global()));
    }

    #[test]
    fn optim_local_q_skips_shelf_filters() {
        // LsPk shelf Q is pinned and unused by the DSP core; a treble cap
        // must not rewrite it, while the peak still enforces.
        let knots = vec![(20.0, 0.5), (20_000.0, 0.5)];
        let x = vec![60.0_f64.log10(), 1.0, -3.0, 3000.0_f64.log10(), 4.0, -3.0];
        let (out, adjustments) = enforce_local_q_at_centers(
            &x,
            PeqModel::LsPk,
            LossType::SpeakerFlat,
            12.0,
            Some(&knots),
        )
        .expect("valid knots");
        assert_eq!(out[1], 1.0, "shelf Q must stay pinned");
        assert_eq!(out[4], 0.5, "peak Q enforces the local cap");
        assert_eq!(adjustments.len(), 1);
        assert_eq!(adjustments[0].filter_index, 1);
    }

    #[test]
    fn optim_k3_envelope_adapter_matches_knot_slices() {
        // The frozen G1 K3 value type and the legacy knot slices share one
        // contract (log-frequency interpolation, endpoint hold,
        // stricter-of-global/local): both entry points must agree exactly.
        use crate::core::constraint_envelope::{LocalQEnvelope, LocalQKnot};
        let envelope = LocalQEnvelope::new(vec![
            LocalQKnot {
                freq_hz: 100.0,
                max_q: 10.0,
            },
            LocalQKnot {
                freq_hz: 1000.0,
                max_q: 1.0,
            },
        ])
        .expect("valid K3 envelope");
        let knots = vec![(100.0, 10.0), (1000.0, 1.0)];
        let x = vec![60.0_f64.log10(), 8.0, -6.0, 6000.0_f64.log10(), 8.0, -6.0];
        let (via_k3, k3_adjustments) = enforce_local_q_envelope_at_centers(
            &x,
            PeqModel::Pk,
            LossType::SpeakerFlat,
            12.0,
            Some(&envelope),
        )
        .expect("valid K3 envelope");
        let (via_slices, slice_adjustments) =
            enforce_local_q_at_centers(&x, PeqModel::Pk, LossType::SpeakerFlat, 12.0, Some(&knots))
                .expect("valid knots");
        assert_eq!(via_k3, via_slices);
        assert_eq!(k3_adjustments, slice_adjustments);
        assert_eq!(via_k3[1], 8.0, "bass Q preserved through the K3 object");
        assert_eq!(via_k3[4], 1.0, "treble Q capped through the K3 object");
    }

    fn stacked_boosts() -> (Vec<f64>, Vec<(f64, f64)>) {
        let x = vec![1000.0_f64.log10(), 1.0, 5.0, 1000.0_f64.log10(), 1.0, 5.0];
        (x, vec![(20.0, 6.0), (20_000.0, 6.0)])
    }

    #[test]
    fn optim_composite_gain_envelope_catches_stacked_filters() {
        let (x, envelope) = stacked_boosts();
        let mut data = flat_objective(log_grid());
        data.max_boost_envelope = Some(envelope);
        let candidate = constrain_candidate("stacked", &x, &data, &ConstraintSpec::unconstrained())
            .expect("valid candidate");
        // Each filter alone respects the 6 dB per-filter bound, so nothing
        // is projected; the summed +10 dB composite still breaches.
        assert!(candidate.gain_adjustments.is_empty());
        assert!(!candidate.feasible);
        assert!(!candidate.composite_breaches.is_empty());
        assert!(
            candidate
                .composite_breaches
                .iter()
                .all(|breach| breach.boost)
        );
        let peak = candidate
            .composite_breaches
            .iter()
            .map(|breach| breach.observed_db)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(peak > 9.0, "stacked peak must approach +10 dB, got {peak}");
    }

    #[test]
    fn optim_constraint_extremum_between_grid_bins() {
        // One narrow filter centered on the geometric midpoint of a coarse
        // interval: the coarse bins see only skirts.
        let coarse = vec![100.0, 200.0, 400.0];
        let x = vec![141.421_356_237_309_5_f64.log10(), 4.0, 6.0];
        let coarse_response = crate::x2peq::x2spl(
            &Array1::from_vec(coarse.clone()),
            &x,
            48_000.0,
            PeqModel::Pk,
        );
        let coarse_max = coarse_response
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(
            coarse_max < 3.0,
            "coarse bins must miss the peak, got {coarse_max}"
        );
        let envelope = vec![(20.0, 3.0), (20_000.0, 3.0)];
        let breaches = check_composite_gain_envelope(
            &x,
            &coarse,
            48_000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
            20.0,
            20_000.0,
            Some(&envelope),
            None,
            VALIDATED_SUBDIVISIONS_PER_BIN,
        )
        .expect("valid grid");
        assert!(!breaches.is_empty());
        assert!(
            breaches.iter().any(|breach| breach.frequency_hz > 100.0
                && breach.frequency_hz < 200.0
                && breach.observed_db > 5.0),
            "dense grid must catch the inter-bin extremum: {breaches:?}"
        );
    }

    #[test]
    fn optim_refinement_cannot_escape_envelope() {
        let mut data = flat_objective(log_grid());
        data.max_boost_envelope = Some(vec![(20.0, 6.0), (20_000.0, 6.0)]);
        let spec = ConstraintSpec {
            global_max_q: 12.0,
            local_q_knots: Some(&[(20.0, 1.0), (20_000.0, 1.0)]),
            boost_knots: None,
            cut_knots: None,
            subdivisions_per_bin: VALIDATED_SUBDIVISIONS_PER_BIN,
        };
        let escaped = vec![1000.0_f64.log10(), 9.0, 20.0];
        let first = constrain_candidate("refine-0", &escaped, &data, &spec).expect("valid");
        assert_eq!(first.params[2], 6.0, "gain projects onto the envelope");
        assert_eq!(first.params[1], 1.0, "Q projects onto the local cap");
        assert!(first.feasible);

        // Re-running the choke-point (what every refinement pass ends with)
        // is idempotent.
        let second = constrain_candidate("refine-1", &first.params, &data, &spec).expect("valid");
        assert_eq!(second.params, first.params);
        assert!(second.feasible);

        // A local step that tries to escape is pulled back onto the envelope.
        let mut tried_escape = first.params.clone();
        tried_escape[2] = 11.0;
        tried_escape[1] = 5.0;
        let third = constrain_candidate("refine-2", &tried_escape, &data, &spec).expect("valid");
        assert_eq!(third.params, first.params);
    }

    #[test]
    fn optim_absent_envelope_legacy_regression() {
        // Absent envelopes are bit-identical no-ops everywhere.
        let x = vec![60.0_f64.log10(), 8.0, 20.0, 6000.0_f64.log10(), 9.0, -20.0];
        let (projected, gains) =
            project_gains_onto_envelopes(&x, PeqModel::Pk, LossType::SpeakerFlat, None, None);
        assert_eq!(projected, x);
        assert!(gains.is_empty());
        let (projected, adjustments) = enforce_local_q_at_centers(
            &x,
            PeqModel::Pk,
            LossType::SpeakerFlat,
            f64::INFINITY,
            None,
        )
        .expect("absent envelope is valid");
        assert_eq!(projected, x);
        assert!(adjustments.is_empty());
        let breaches = check_composite_gain_envelope(
            &x,
            &log_grid(),
            48_000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
            20.0,
            20_000.0,
            None,
            None,
            VALIDATED_SUBDIVISIONS_PER_BIN,
        )
        .expect("absent envelope is valid");
        assert!(breaches.is_empty());
        let data = flat_objective(log_grid());
        let candidate = constrain_candidate("legacy", &x, &data, &ConstraintSpec::unconstrained())
            .expect("valid");
        assert_eq!(candidate.params, x);
        assert!(candidate.feasible);
        assert!(candidate.diagnostics().is_empty());

        // Pinned legacy loss for a fixed tiny fixture guards the evaluation
        // path against silent retuning. The fixture grid matches the probe
        // the pin was recorded from; the evaluation code is untouched by
        // this lane.
        let tiny = flat_objective(vec![100.0, 200.0, 400.0, 800.0, 1600.0]);
        let loss = super::super::compute_base_fitness(&[500.0_f64.log10(), 1.0, 3.0], &tiny);
        assert!(
            (loss - 3.931_713_552_896_190_4).abs() < 1e-12,
            "legacy loss moved: {loss}"
        );

        // Pinned seeded guesses guard RNG consumption of the seed path.
        let target = Array1::from_vec(vec![0.0, 3.0, 0.0, -2.0, 0.0]);
        let fgrid = Array1::from_vec(vec![100.0, 200.0, 400.0, 800.0, 1600.0]);
        let log_bounds = (100.0_f64.log10(), 1600.0_f64.log10());
        let bounds = vec![
            log_bounds,
            (0.5, 3.0),
            (-6.0, 6.0),
            log_bounds,
            (0.5, 3.0),
            (-6.0, 6.0),
        ];
        let config = crate::initial_guess::SmartInitConfig {
            num_guesses: 1,
            seed: Some(7),
            ..Default::default()
        };
        let guesses = crate::initial_guess::create_smart_initial_guesses(
            &target,
            &fgrid,
            2,
            &bounds,
            &config,
            PeqModel::Pk,
        );
        let expected = [
            2.258_188_617_898_136_3,
            0.549_909_705_814_324_4,
            2.422_440_597_542_952_4,
            2.906_767_369_759_104_5,
            1.270_678_781_681_284_3,
            -1.590_748_094_006_693_4,
        ];
        assert_eq!(guesses.len(), 1);
        for (actual, wanted) in guesses[0].iter().zip(expected.iter()) {
            assert!(
                (actual - wanted).abs() < 1e-12,
                "seeded guess moved: {actual} versus {wanted}"
            );
        }
    }

    fn feasible_candidate() -> ConstrainedCandidate {
        ConstrainedCandidate {
            candidate_id: String::from("flat"),
            params: vec![500.0_f64.log10(), 1.0, 0.0],
            gain_adjustments: Vec::new(),
            q_adjustments: Vec::new(),
            composite_breaches: Vec::new(),
            feasible: true,
        }
    }

    fn infeasible_candidate() -> ConstrainedCandidate {
        let (x, envelope) = stacked_boosts();
        let mut data = flat_objective(log_grid());
        data.max_boost_envelope = Some(envelope);
        let mut candidate =
            constrain_candidate("stacked", &x, &data, &ConstraintSpec::unconstrained())
                .expect("valid");
        candidate.candidate_id = String::from("stacked");
        assert!(!candidate.feasible);
        candidate
    }

    #[test]
    fn optim_infeasible_and_budget_exhausted_distinct() {
        let flat = feasible_candidate();
        let stacked = infeasible_candidate();
        let converged = evidence_for(Ok(("converged".to_string(), 0.5)), &flat.params);
        let exhausted = evidence_for(
            Ok(("maximum evaluations reached (nfev=50)".to_string(), 1.0)),
            &flat.params,
        );
        let failed = evidence_for(Err(("line search failed".to_string(), 2.0)), &flat.params);

        let success = classify_outcome(&converged, &flat);
        let budget = classify_outcome(&exhausted, &flat);
        let convergence = classify_outcome(&failed, &flat);
        // Converged status cannot bless an envelope breach: infeasible wins.
        let infeasible = classify_outcome(&converged, &stacked);

        assert_eq!(success.kind, OptimizationOutcomeKind::BoundedSuccess);
        assert_eq!(budget.kind, OptimizationOutcomeKind::BudgetExhausted);
        assert_eq!(
            convergence.kind,
            OptimizationOutcomeKind::ConvergenceFailure
        );
        assert_eq!(
            infeasible.kind,
            OptimizationOutcomeKind::NoFeasibleCandidate
        );
        let kinds = [success.kind, budget.kind, convergence.kind, infeasible.kind];
        for (i, left) in kinds.iter().enumerate() {
            for right in &kinds[i + 1..] {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn optim_diagnostic_reports_binding_bound() {
        let stacked = infeasible_candidate();
        let diagnostics = stacked.diagnostics();
        assert!(!diagnostics.is_empty());
        // The binding diagnostic is the composite peak: widest excess over
        // the bound, localized at the stacked center.
        let breach = diagnostics
            .iter()
            .filter(|d| d.kind == ConstraintKind::CompositeBoost)
            .max_by(|left, right| {
                left.excess()
                    .partial_cmp(&right.excess())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("composite boost diagnostic");
        assert_eq!(breach.candidate_id, "stacked");
        assert!(
            (breach.bound - 6.0).abs() < 1e-12,
            "diagnostic must report the binding 6 dB bound, got {}",
            breach.bound
        );
        assert!(
            breach.observed > breach.bound,
            "observed {} must exceed the bound",
            breach.observed
        );
        assert!(
            breach
                .frequency_hz
                .is_some_and(|f| (f - 1000.0).abs() < 200.0),
            "peak breach must localize near the stacked center: {breach:?}"
        );
        assert!(breach.is_binding(1e-9));
        assert!(breach.excess() > 3.0);
    }

    #[test]
    fn optim_no_candidate_is_not_physical_impossibility() {
        let flat = feasible_candidate();
        let stacked = infeasible_candidate();
        let converged = evidence_for(Ok(("converged".to_string(), 0.5)), &flat.params);
        let exhausted = evidence_for(
            Ok(("maximum evaluations reached (nfev=50)".to_string(), 1.0)),
            &flat.params,
        );
        let failed = evidence_for(Err(("line search failed".to_string(), 2.0)), &flat.params);
        for outcome in [
            classify_outcome(&converged, &flat),
            classify_outcome(&exhausted, &flat),
            classify_outcome(&failed, &flat),
            classify_outcome(&converged, &stacked),
        ] {
            assert!(
                !outcome.implies_physical_impossibility(),
                "optimizer failure must never read as physical impossibility: {outcome:?}"
            );
        }
        // The infeasible outcome carries constraint evidence instead of a
        // physics verdict: diagnostics are present and non-empty.
        let infeasible = classify_outcome(&converged, &stacked);
        assert!(!infeasible.diagnostics.is_empty());
        assert!(!infeasible.detail.is_empty());
    }

    fn breaching_objective() -> ObjectiveData {
        // Tight composite boost envelope: a +6 dB peak at 1 kHz breaches.
        ObjectiveDataBuilder::new(
            Array1::from_vec(log_grid()),
            Array1::zeros(200),
            Array1::from_elem(200, 5.0),
            48_000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
        )
        .max_db(12.0)
        .min_db(0.0)
        .freq_range(20.0, 20_000.0)
        .max_boost_envelope(vec![(20.0, 1.0), (20_000.0, 1.0)])
        .build()
        .expect("valid test objective")
    }

    /// A3 regression: a candidate breaching its composite envelope must be
    /// refused at finalization, never emitted with the stale winner loss.
    #[test]
    fn roadmap_correction_finalize_refuses_composite_breach() {
        let data = breaching_objective();
        // Two stacked +6 dB peaks at 1 kHz: per-filter repair clamps each to
        // +1 dB, but the summed composite still breaches the +1 dB envelope.
        let x = vec![1000.0_f64.log10(), 2.0, 6.0, 1000.0_f64.log10(), 2.0, 6.0];
        let spec = ConstraintSpec::unconstrained();
        let error = finalize_candidate("breach", &x, &data, &spec).expect_err("breach refused");
        assert!(error.contains("composite"), "{error}");
    }

    /// Repair projects gains onto envelopes and re-verifies the loss at the
    /// repaired parameters; compliant candidates pass through untouched.
    #[test]
    fn roadmap_correction_finalize_repairs_and_reverifies() {
        let data = breaching_objective();
        // Small boost inside the envelope: no repair, loss re-verified.
        let x = vec![1000.0_f64.log10(), 2.0, 0.5];
        let finalized =
            finalize_candidate("compliant", &x, &data, &ConstraintSpec::unconstrained())
                .expect("compliant candidate finalizes");
        assert_eq!(finalized.params, x);
        let expected = super::super::compute::compute_fitness_penalties_ref(&x, &data);
        assert!(
            (finalized.loss - expected).abs() < 1e-12,
            "loss re-verified"
        );
        // Over-boost repaired onto the envelope with a fresh loss.
        let tight = ObjectiveDataBuilder::new(
            Array1::from_vec(log_grid()),
            Array1::zeros(200),
            Array1::from_elem(200, 5.0),
            48_000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
        )
        .max_db(12.0)
        .min_db(0.0)
        .freq_range(20.0, 20_000.0)
        .max_boost_envelope(vec![(20.0, 1.0), (20_000.0, 1.0)])
        .min_cut_envelope(vec![(20.0, -12.0), (20_000.0, -12.0)])
        .build()
        .expect("valid test objective");
        // Per-filter repair clamps the peak to the envelope; the composite
        // of one clamped +6 dB peak still breaches +1 dB, so this refuses.
        // Use a barely-over peak instead: +1.5 dB clamps to +1 dB and the
        // composite passes, proving repair-then-accept.
        let barely = vec![1000.0_f64.log10(), 2.0, 1.5];
        let finalized = finalize_candidate(
            "repaired",
            &barely,
            &tight,
            &ConstraintSpec::unconstrained(),
        )
        .expect("repaired candidate finalizes");
        assert!(
            finalized.constrained.gain_adjustments.len() == 1,
            "one gain repair recorded"
        );
        let expected =
            super::super::compute::compute_fitness_penalties_ref(&finalized.params, &tight);
        assert!(
            (finalized.loss - expected).abs() < 1e-12,
            "loss matches repaired params"
        );
    }

    /// Local-Q policy knots mirror bound-time conservatism at centers.
    #[test]
    fn roadmap_correction_policy_knots_mirror_bounds() {
        use super::super::params::FrequencyQPolicy;
        // Both sides declared: below-Schroeder caps low, guard band caps high.
        let policy = FrequencyQPolicy {
            schroeder_hz: Some(200.0),
            low_max_q: Some(8.0),
            high_start_hz: Some(1000.0),
            high_max_q: Some(2.0),
        };
        let knots = policy_local_q_knots(&policy, 12.0)
            .expect("valid policy knots")
            .expect("policy declares knots");
        assert!((envelope_bound_at(&knots, 100.0) - 8.0).abs() < 1e-9);
        assert!((envelope_bound_at(&knots, 500.0) - 2.0).abs() < 1e-9);
        assert!((envelope_bound_at(&knots, 5000.0) - 2.0).abs() < 1e-9);
        // Overlapping declarations step at the lower hinge: centers that can
        // reach the guard still cap high.
        let overlap = policy_local_q_knots(
            &FrequencyQPolicy {
                schroeder_hz: Some(1000.0),
                low_max_q: Some(8.0),
                high_start_hz: Some(200.0),
                high_max_q: Some(2.0),
            },
            12.0,
        )
        .expect("overlapping policy still encodes")
        .expect("overlap declares knots");
        assert!((envelope_bound_at(&overlap, 100.0) - 8.0).abs() < 1e-9);
        assert!((envelope_bound_at(&overlap, 500.0) - 2.0).abs() < 1e-9);
        // Single-sided policies step against the finite global cap.
        let low_only = policy_local_q_knots(
            &FrequencyQPolicy {
                schroeder_hz: Some(200.0),
                low_max_q: Some(8.0),
                high_start_hz: None,
                high_max_q: None,
            },
            12.0,
        )
        .expect("low-only policy encodes")
        .expect("low-only declares knots");
        assert!((envelope_bound_at(&low_only, 100.0) - 8.0).abs() < 1e-9);
        assert!((envelope_bound_at(&low_only, 5000.0) - 12.0).abs() < 1e-9);
        // No policy: no local knots, global cap only.
        assert!(
            policy_local_q_knots(
                &FrequencyQPolicy {
                    schroeder_hz: None,
                    low_max_q: None,
                    high_start_hz: None,
                    high_max_q: None,
                },
                12.0
            )
            .expect("empty policy is valid")
            .is_none()
        );
        // A single-sided policy under an infinite global cap cannot encode:
        // refuse instead of silently dropping the declared cap.
        assert!(
            policy_local_q_knots(
                &FrequencyQPolicy {
                    schroeder_hz: Some(200.0),
                    low_max_q: Some(8.0),
                    high_start_hz: None,
                    high_max_q: None,
                },
                f64::INFINITY
            )
            .is_err()
        );
    }
}
