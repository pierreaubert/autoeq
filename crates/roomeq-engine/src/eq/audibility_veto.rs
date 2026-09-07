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
use roomeq_model::{
    AppliedThreshold, AssessmentConfidence, AssessmentProvenance, AssessmentRecord,
    EnforcementState, FilterAudibilityConfig, FilterVetoVerdict, ReportOutcome, VetoDecision,
    VetoReason,
};

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
        Some(w) => {
            shape_db
                .iter()
                .zip(w.iter())
                .map(|(s, w)| s * w)
                .sum::<f64>()
                / weight_sum
        }
        None => shape_db.iter().sum::<f64>() / count as f64,
    };
    // Band levels at calibrated SPL.
    let levels: Vec<f64> = shape_db
        .iter()
        .map(|s| listening_phon + (s - mean))
        .collect();
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
        full += response;
    }
    let mut partial = full.clone();
    if let Some(removed) = responses.get(without_index) {
        partial -= removed;
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
            let peak_delta_db = response
                .iter()
                .fold(0.0_f64, |max, value| max.max(value.abs()));
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
                acceptance: AssessmentRecord::default(),
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
    if !evaluation.config.report_only && !evaluation.config.enforcement_authorized() {
        log::warn!(
            "audibility veto enforcement requested (report_only=false) without \
             allow_enforcement_with_experimental_proxy; staying advisory because \
             the loudness proxy is experimental and unvalidated"
        );
    }
    let verdicts = evaluate_audibility_veto(evaluation);
    enforce_veto_verdicts(filters, verdicts, evaluation.config.enforcement_authorized())
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
        *histogram
            .entry(format!("{:?}", verdict.reason))
            .or_insert(0) += 1;
    }
    log::info!(
        "  Audibility veto: {kept_count} kept, {removed} removed (enforce={enforce}); reasons: {histogram:?}",
    );
    (kept, verdicts)
}

/// Frozen-full-chain fingerprint: FNV-1a over the filter count plus the
/// composite dB response quantized to 1e-3 dB. Deterministic for identical
/// inputs, so twin runs (report-only vs enforced) share the reference id.
fn fingerprint_f0(composite_db: &Array1<f64>, filter_count: usize) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET;
    for byte in (filter_count as u64).to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for value in composite_db {
        for byte in ((value * 1000.0) as i64).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    format!("F0:{hash:016x}")
}

/// One filter removed by adjudication, with its stable pre-removal index
/// so callers can roll back by re-insertion.
#[derive(Debug, Clone)]
pub struct RemovedFilter {
    /// Stable index into the pre-adjudication filter set (matches the
    /// verdict `index`, which never shifts under removal).
    pub index: usize,
    /// The removed filter.
    pub filter: Biquad,
}

/// Inputs for Stage 1 cumulative adjudication.
#[derive(Debug, Clone)]
pub struct AdjudicationConfig {
    /// Calibrated evaluation level in phons (provenance only).
    pub listening_phon: f64,
    /// Per-removal inaudibility quantum in sones: a single removal is
    /// accepted only below this impact. Sourced from
    /// `elimination_loudness_delta_sones` (experimental proxy units).
    pub per_step_quantum_sones: f64,
    /// Cumulative cap in sones, when a pruning budget is configured.
    /// `None` falls back to one per-step quantum: total pruning stays
    /// within a single inaudibility quantum unless a budget is declared.
    pub cumulative_cap_sones: Option<f64>,
    /// No accepted removal set may move any single bin further than this
    /// from F0. Reuses the JND floor: a removal that shifts one region by
    /// an audible amount is rejected no matter how narrow it is.
    pub local_deviation_cap_db: f64,
    /// `true` actually removes accepted filters; `false` (report-only)
    /// walks the same greedy order hypothetically and keeps everything.
    pub enforce: bool,
    /// Model version recorded in acceptance provenance.
    pub model_version: String,
}

/// Outcome of adjudicating heuristic nominations against F0.
#[derive(Debug, Clone)]
pub struct VetoAdjudication {
    /// Kept filters in original relative order.
    pub kept: Vec<Biquad>,
    /// Removed filters with stable indices (empty in report-only mode).
    /// Re-inserting these at their indices reproduces the F0 composite.
    pub removed: Vec<RemovedFilter>,
    /// Frozen-full-chain reference identity, shared by twin runs.
    pub f0_reference_id: String,
    /// Cumulative loudness distance of the kept chain from F0 in sones
    /// (hypothetical in report-only mode).
    pub cumulative_loudness_delta_sones: f64,
    /// Worst single-bin distance of the kept chain from F0 in dB
    /// (hypothetical in report-only mode).
    pub max_local_deviation_db: f64,
    /// Whether removals were applied.
    pub enforced: bool,
}

/// Stored record of a post-pass adjudication: reference identity plus
/// everything needed to roll back. Carried on the optimization result so
/// reports and exports can cite the F0 reference and restore deletions.
/// The kept filters themselves live on the result; they are not repeated
/// here.
#[derive(Debug, Clone)]
pub struct VetoAdjudicationSummary {
    /// Frozen-full-chain reference identity.
    pub f0_reference_id: String,
    /// Removed filters with stable pre-removal indices. Re-inserting
    /// these into the kept set at their indices reproduces F0.
    pub removed: Vec<RemovedFilter>,
    /// Cumulative loudness distance of the kept chain from F0 in sones.
    pub cumulative_loudness_delta_sones: f64,
    /// Worst single-bin distance of the kept chain from F0 in dB.
    pub max_local_deviation_db: f64,
    /// Whether removals were applied (false in report-only mode).
    pub enforced: bool,
}

impl VetoAdjudication {
    /// Split off the storable summary; `kept` stays with the caller.
    pub fn summarize(&self) -> VetoAdjudicationSummary {
        VetoAdjudicationSummary {
            f0_reference_id: self.f0_reference_id.clone(),
            removed: self.removed.clone(),
            cumulative_loudness_delta_sones: self.cumulative_loudness_delta_sones,
            max_local_deviation_db: self.max_local_deviation_db,
            enforced: self.enforced,
        }
    }
}

fn acceptance_record(
    outcome: ReportOutcome,
    enforcement: EnforcementState,
    f0_reference_id: &str,
    listening_phon: f64,
    model_version: &str,
    quantum_sones: f64,
    local_cap_db: f64,
    budget_cap_sones: Option<f64>,
    reason: String,
) -> AssessmentRecord {
    let mut thresholds = vec![
        AppliedThreshold {
            name: String::from("elimination_loudness_delta_sones"),
            value: quantum_sones,
            unit: String::from("sones-experimental-proxy"),
        },
        AppliedThreshold {
            name: String::from("acceptance_local_deviation_db"),
            value: local_cap_db,
            unit: String::from("db"),
        },
    ];
    if let Some(cap) = budget_cap_sones {
        thresholds.push(AppliedThreshold {
            name: String::from("pruning_budget_max_cumulative_delta"),
            value: cap,
            unit: String::from("sones-experimental-proxy"),
        });
    }
    AssessmentRecord {
        outcome,
        // The loudness proxy has no measured listener/model validation, so
        // even accepted adjudications stay Low confidence (F12). The units
        // (`sones-experimental-proxy`) and heuristic model name say the same.
        confidence: AssessmentConfidence::Low,
        enforcement,
        provenance: AssessmentProvenance {
            model: String::from("heuristic-erb-proxy"),
            model_version: String::from(model_version),
            calibration: format!("nominal-{listening_phon}phon"),
            reference: String::from(f0_reference_id),
        },
        thresholds,
        reason,
    }
}

/// Adjudicate heuristic nominations one removal at a time against the
/// frozen full chain (Stage 1).
///
/// Nominations (`VetoDecision::Remove`) are candidates only. Starting from
/// the full set F0, each step removes the candidate with the smallest
/// loudness impact, recomputes interactions against the current chain,
/// and accepts the removal only if all three hold: the incremental impact
/// stays below the per-step quantum, the cumulative loudness distance from
/// F0 stays within cap, and no single bin moves further than the local
/// cap. The first failure stops the walk; unevaluated candidates keep
/// `Keep` with `NotEvaluated` enforcement.
///
/// Kept filters preserve original relative order; verdict `index` values
/// are pre-removal positions and never shift, so they are stable
/// identifiers for reports and rollback. In report-only mode the same
/// greedy order is walked hypothetically: accepted candidates become
/// `CandidateRemoval` (advisory) and everything is kept.
pub fn adjudicate_veto_removals(
    filters: Vec<Biquad>,
    verdicts: &mut [FilterVetoVerdict],
    freqs: &Array1<f64>,
    config: &AdjudicationConfig,
) -> VetoAdjudication {
    let responses: Vec<Array1<f64>> = filters
        .iter()
        .map(|filter| filter_db_response(filter, freqs))
        .collect();
    let erb = erb_positions(freqs);
    let weights = erb_weights(freqs);
    let weights_ref = weights.as_ref();
    let f0_composite: Array1<f64> = responses
        .iter()
        .fold(Array1::zeros(freqs.len()), |mut sum, response| {
            sum += response;
            sum
        });
    let f0_loudness =
        approximate_loudness_sones(&f0_composite, &erb, weights_ref, config.listening_phon);
    let f0_reference_id = fingerprint_f0(&f0_composite, filters.len());
    let no_adjudication = || VetoAdjudication {
        kept: filters.clone(),
        removed: Vec::new(),
        f0_reference_id: f0_reference_id.clone(),
        cumulative_loudness_delta_sones: 0.0,
        max_local_deviation_db: 0.0,
        enforced: false,
    };

    // Tracks which nominations received an acceptance record, so a second
    // adjudication pass over annotated verdicts cannot mistake old records
    // for unevaluated candidates.
    let mut evaluated = vec![false; filters.len()];

    if verdicts.len() != filters.len() {
        // Caller contract broken: never remove on ambiguous alignment.
        for verdict in verdicts.iter_mut() {
            verdict.acceptance = acceptance_record(
                ReportOutcome::Keep,
                EnforcementState::NotEvaluated,
                &f0_reference_id,
                config.listening_phon,
                &config.model_version,
                config.per_step_quantum_sones,
                config.local_deviation_cap_db,
                config.cumulative_cap_sones,
                String::from("verdict/filter count mismatch; no adjudication"),
            );
        }
        return no_adjudication();
    }

    // Heuristic passes need no adjudication: kept, advisory.
    for (index, verdict) in verdicts.iter_mut().enumerate() {
        if verdict.decision != VetoDecision::Remove {
            verdict.acceptance = acceptance_record(
                ReportOutcome::Keep,
                EnforcementState::Advisory,
                &f0_reference_id,
                config.listening_phon,
                &config.model_version,
                config.per_step_quantum_sones,
                config.local_deviation_cap_db,
                config.cumulative_cap_sones,
                String::from("heuristic pass; nothing to adjudicate"),
            );
            evaluated[index] = true;
        }
    }

    // Working set of remaining response indices, in original order.
    let mut remaining: Vec<usize> = (0..filters.len()).collect();
    let mut removed: Vec<RemovedFilter> = Vec::new();
    let cumulative_cap = config
        .cumulative_cap_sones
        .unwrap_or(config.per_step_quantum_sones);
    let mut cumulative_loudness = 0.0;
    let mut max_local = 0.0;
    let mut stopped = false;

    while !stopped {
        // Least-impact candidate among those still present.
        let current: Vec<Array1<f64>> =
            remaining.iter().map(|&i| responses[i].clone()).collect();
        let current_total: Array1<f64> =
            current
                .iter()
                .fold(Array1::zeros(freqs.len()), |mut sum, response| {
                    sum += response;
                    sum
                });
        // `candidate` is the stable original index (into `responses` and
        // `verdicts`); `position` is its transient slot in `current` for
        // the loudness call. Confusing the two removes nothing and spins.
        let mut best: Option<(usize, usize, f64)> = None;
        for (position, &original) in remaining.iter().enumerate() {
            if verdicts[original].decision != VetoDecision::Remove {
                continue;
            }
            let impact =
                loudness_delta_sones(&current, position, &erb, weights_ref, config.listening_phon);
            if !impact.is_finite() {
                continue;
            }
            if best.is_none_or(|(_, _, best_impact)| impact < best_impact) {
                best = Some((original, position, impact));
            }
        }
        let Some((candidate, _, impact)) = best else {
            break;
        };

        let removal_composite = &current_total - &responses[candidate];
        let candidate_cumulative_loudness =
            (approximate_loudness_sones(&removal_composite, &erb, weights_ref, config.listening_phon)
                - f0_loudness)
                .abs();
        let candidate_max_local = f0_composite
            .iter()
            .zip(removal_composite.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        let rejection = if impact >= config.per_step_quantum_sones {
            Some(format!(
                "incremental impact {impact:.4} sones at/above per-step quantum {:.4}",
                config.per_step_quantum_sones
            ))
        } else if candidate_cumulative_loudness > cumulative_cap {
            Some(format!(
                "cumulative loudness {candidate_cumulative_loudness:.4} sones would exceed cap {cumulative_cap:.4}"
            ))
        } else if candidate_max_local > config.local_deviation_cap_db {
            Some(format!(
                "local deviation {candidate_max_local:.2} dB would exceed cap {:.2}",
                config.local_deviation_cap_db
            ))
        } else {
            None
        };

        evaluated[candidate] = true;
        if let Some(reason) = rejection {
            // Evaluated and rejected: keep, advisory. The walk stops here;
            // later candidates are unevaluated, never assumed safe.
            verdicts[candidate].acceptance = acceptance_record(
                ReportOutcome::Keep,
                EnforcementState::Advisory,
                &f0_reference_id,
                config.listening_phon,
                &config.model_version,
                config.per_step_quantum_sones,
                config.local_deviation_cap_db,
                config.cumulative_cap_sones,
                format!("removal rejected: {reason}"),
            );
            stopped = true;
            continue;
        }

        cumulative_loudness = candidate_cumulative_loudness;
        max_local = candidate_max_local;
        remaining.retain(|&i| i != candidate);
        let (outcome, enforcement, verb) = if config.enforce {
            verdicts[candidate].enforced = true;
            (
                ReportOutcome::AcceptedRemoval,
                EnforcementState::Enforced,
                "removal accepted and applied",
            )
        } else {
            (
                ReportOutcome::CandidateRemoval,
                EnforcementState::Advisory,
                "removal accepted hypothetically; report-only keeps the filter",
            )
        };
        verdicts[candidate].acceptance = acceptance_record(
            outcome,
            enforcement,
            &f0_reference_id,
            config.listening_phon,
            &config.model_version,
            config.per_step_quantum_sones,
            config.local_deviation_cap_db,
            config.cumulative_cap_sones,
            format!(
                "{verb}: incremental impact {impact:.4} sones, cumulative {cumulative_loudness:.4}, local {max_local:.2} dB"
            ),
        );
        if config.enforce {
            removed.push(RemovedFilter {
                index: candidate,
                filter: filters[candidate].clone(),
            });
        }
        log::info!(
            "  Veto adjudication: filter #{candidate} ({center:.0} Hz, {gain:+.1} dB): {verb} \
             (incremental={impact:.4} sones, cumulative={cumulative_loudness:.4}, local={max_local:.2} dB)",
            center = verdicts[candidate].center_hz,
            gain = verdicts[candidate].gain_db,
        );
    }

    // Candidates never reached keep their nomination with NotEvaluated.
    for (index, verdict) in verdicts.iter_mut().enumerate() {
        if verdict.decision == VetoDecision::Remove && !evaluated[index] {
            verdict.acceptance = acceptance_record(
                ReportOutcome::Keep,
                EnforcementState::NotEvaluated,
                &f0_reference_id,
                config.listening_phon,
                &config.model_version,
                config.per_step_quantum_sones,
                config.local_deviation_cap_db,
                config.cumulative_cap_sones,
                String::from("not evaluated: adjudication stopped"),
            );
        }
    }

    let kept: Vec<Biquad> = if config.enforce {
        remaining.iter().map(|&i| filters[i].clone()).collect()
    } else {
        filters.clone()
    };
    // In report-only mode nothing was removed: hypothetical stats stay,
    // but the chain is unchanged, so report zero drift.
    let (cumulative_loudness, max_local) = if config.enforce {
        (cumulative_loudness, max_local)
    } else {
        (0.0, 0.0)
    };
    log::info!(
        "  Veto adjudication: {} kept, {} removed (enforce={}); F0={f0_reference_id} cumulative={cumulative_loudness:.4} sones local={max_local:.2} dB",
        kept.len(),
        removed.len(),
        config.enforce,
    );
    VetoAdjudication {
        kept,
        removed,
        f0_reference_id,
        cumulative_loudness_delta_sones: cumulative_loudness,
        max_local_deviation_db: max_local,
        enforced: config.enforce,
    }
}

/// Greedy backward elimination in veto (loudness-delta) units.
///
/// Repeatedly removes the filter whose removal changes total masked
/// loudness least, while that change stays below `threshold_sones` (same
/// greedy shape as [`super::consts::backward_eliminate`], different units).
/// The finalist is always retained (`len() <= 1` stops the walk): this
/// loudness-only pass must not empty the set on its own, because total
/// loudness alone cannot certify inaudibility (a narrow deep feature can
/// be locally audible with negligible total impact). The identity/
/// zero-filter solution is admissible, but only the Stage 1 post-pass
/// adjudication — which adds the cumulative-local-deviation guard — may
/// remove the last filter. Returns the kept filters; loss is recomputed
/// by the caller in raw units.
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
            let impact =
                loudness_delta_sones(&remaining, i, &erb, weights.as_ref(), listening_phon);
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

    fn apply(
        filters: Vec<Biquad>,
        report_only: bool,
        experimental_ack: bool,
    ) -> (Vec<Biquad>, Vec<FilterVetoVerdict>) {
        let freqs = grid();
        let evaluation = VetoEvaluation {
            filters: &filters,
            freqs: &freqs,
            listening_phon: 75.0,
            config: FilterAudibilityConfig {
                report_only,
                allow_enforcement_with_experimental_proxy: experimental_ack,
                ..FilterAudibilityConfig::default()
            },
            hf_guard_start_hz: 1600.0,
        };
        apply_audibility_veto(filters.clone(), &evaluation)
    }

    #[test]
    fn report_only_records_without_removing() {
        let filters = vec![peak(0.2, 500.0, 1.0), peak(3.0, 500.0, 1.0)];
        let (kept, verdicts) = apply(filters, true, false);
        assert_eq!(kept.len(), 2, "report-only must not remove");
        assert_eq!(verdicts[0].decision, VetoDecision::Remove);
        assert!(!verdicts[0].enforced);
    }

    #[test]
    fn enforcement_removes_with_reason_code() {
        let filters = vec![peak(0.2, 500.0, 1.0), peak(3.0, 500.0, 1.0)];
        let (kept, verdicts) = apply(filters, false, true);
        assert_eq!(kept.len(), 1);
        assert!(verdicts[0].enforced);
        assert_eq!(verdicts[0].reason, VetoReason::SubJnd);
        assert!(!verdicts[1].enforced);
    }

    #[test]
    fn enforcement_without_experimental_ack_stays_advisory() {
        // F12: the loudness proxy is unvalidated, so `report_only: false`
        // alone must not remove filters.
        let filters = vec![peak(0.2, 500.0, 1.0), peak(3.0, 500.0, 1.0)];
        let (kept, verdicts) = apply(filters, false, false);
        assert_eq!(kept.len(), 2, "unacknowledged enforcement must stay advisory");
        assert!(verdicts.iter().all(|verdict| !verdict.enforced));
        assert_eq!(verdicts[0].decision, VetoDecision::Remove);
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

    fn adjudicate_config(enforce: bool) -> AdjudicationConfig {
        AdjudicationConfig {
            listening_phon: 75.0,
            per_step_quantum_sones: 0.05,
            cumulative_cap_sones: None,
            local_deviation_cap_db: 1.0,
            enforce,
            model_version: String::from("test"),
        }
    }

    fn nominate(filters: &[Biquad]) -> (Array1<f64>, Vec<FilterVetoVerdict>) {
        let freqs = grid();
        let evaluation = VetoEvaluation {
            filters,
            freqs: &freqs,
            listening_phon: 75.0,
            config: config(),
            hf_guard_start_hz: 1600.0,
        };
        let verdicts = evaluate_audibility_veto(&evaluation);
        (freqs, verdicts)
    }

    fn composite_of(filters: &[Biquad], freqs: &Array1<f64>) -> Array1<f64> {
        filters
            .iter()
            .map(|filter| filter_db_response(filter, freqs))
            .fold(Array1::zeros(freqs.len()), |mut sum, response| {
                sum += &response;
                sum
            })
    }

    #[test]
    fn sub_jnd_ripple_accepts_identity_with_stable_f0() {
        // Single 0.2 dB ripple: nominated, then accepted down to the
        // identity solution. Twin report-only run shares the F0 id and
        // keeps the filter as a mere candidate.
        let filters = vec![peak(0.2, 500.0, 1.0)];
        let (freqs, mut verdicts) = nominate(&filters);
        assert_eq!(verdicts[0].decision, VetoDecision::Remove);
        let adjudication =
            adjudicate_veto_removals(filters.clone(), &mut verdicts, &freqs, &adjudicate_config(true));
        assert!(adjudication.kept.is_empty(), "identity must be reachable");
        assert_eq!(adjudication.removed.len(), 1);
        assert_eq!(adjudication.removed[0].index, 0);
        assert!(verdicts[0].enforced);
        let record = &verdicts[0].acceptance;
        assert_eq!(record.outcome, ReportOutcome::AcceptedRemoval);
        assert_eq!(record.enforcement, EnforcementState::Enforced);
        // F12: the proxy has no listener validation; accepted removals stay
        // Low confidence even when enforced.
        assert_eq!(record.confidence, AssessmentConfidence::Low);
        assert_eq!(record.provenance.reference, adjudication.f0_reference_id);
        assert_eq!(record.provenance.model, "heuristic-erb-proxy");
        assert!(
            adjudication.cumulative_loudness_delta_sones <= 0.05 + 1e-9,
            "cumulative {} exceeds one quantum",
            adjudication.cumulative_loudness_delta_sones
        );
        assert!(
            adjudication.max_local_deviation_db <= 1.0 + 1e-9,
            "local {} exceeds JND",
            adjudication.max_local_deviation_db
        );

        let (_, mut verdicts_ro) = nominate(&filters);
        let ro = adjudicate_veto_removals(
            filters.clone(),
            &mut verdicts_ro,
            &freqs,
            &adjudicate_config(false),
        );
        assert_eq!(ro.kept.len(), 1, "report-only removes nothing");
        assert!(ro.removed.is_empty());
        assert_eq!(ro.f0_reference_id, adjudication.f0_reference_id);
        assert_eq!(
            verdicts_ro[0].acceptance.outcome,
            ReportOutcome::CandidateRemoval
        );
        assert_eq!(
            verdicts_ro[0].acceptance.enforcement,
            EnforcementState::Advisory
        );
    }

    #[test]
    fn audible_narrow_resonance_survives_erb_nomination() {
        // Deep narrow notch: the width heuristic nominates it, but removing
        // it would move one region by 6 dB, so adjudication keeps it. Width
        // alone never authorizes removal.
        let filters = vec![peak(-6.0, 1000.0, 25.0)];
        let (freqs, mut verdicts) = nominate(&filters);
        assert_eq!(verdicts[0].decision, VetoDecision::Remove);
        assert_eq!(verdicts[0].reason, VetoReason::SubErbWidth);
        let adjudication =
            adjudicate_veto_removals(filters, &mut verdicts, &freqs, &adjudicate_config(true));
        assert_eq!(adjudication.kept.len(), 1, "resonance must be retained");
        assert!(adjudication.removed.is_empty());
        assert!(!verdicts[0].enforced);
        assert_eq!(verdicts[0].acceptance.outcome, ReportOutcome::Keep);
        assert!(
            verdicts[0].acceptance.reason.contains("local deviation"),
            "rejection must name the guard, got: {}",
            verdicts[0].acceptance.reason
        );
    }

    #[test]
    fn cancelling_pair_is_retained() {
        // +3 dB and -3 dB at the same center cancel out, but removing either
        // one alone shifts the response by 3 dB: one-at-a-time evaluation
        // keeps both. Joint removal is future work, not assumed safe.
        let filters = vec![peak(3.0, 500.0, 2.0), peak(-3.0, 500.0, 2.0)];
        let (freqs, mut verdicts) = nominate(&filters);
        let adjudication =
            adjudicate_veto_removals(filters.clone(), &mut verdicts, &freqs, &adjudicate_config(true));
        assert_eq!(adjudication.kept.len(), 2);
        assert!(adjudication.removed.is_empty());
        for verdict in &verdicts {
            assert_eq!(verdict.acceptance.outcome, ReportOutcome::Keep);
        }
    }

    #[test]
    fn overlapping_redundant_pair_simplifies_within_bounds() {
        // Two identical 0.4 dB cuts at one center: each removal is locally
        // sub-JND and cumulatively bounded, so both go and the chain ends
        // 0.8 dB from F0 at worst.
        let filters = vec![peak(-0.4, 500.0, 1.0), peak(-0.4, 500.0, 1.0)];
        let (freqs, mut verdicts) = nominate(&filters);
        let adjudication =
            adjudicate_veto_removals(filters, &mut verdicts, &freqs, &adjudicate_config(true));
        assert!(adjudication.kept.is_empty());
        assert_eq!(adjudication.removed.len(), 2);
        assert!(adjudication.max_local_deviation_db <= 1.0 + 1e-9);
        assert!(adjudication.cumulative_loudness_delta_sones <= 0.05 + 1e-9);
    }

    #[test]
    fn accumulated_skirt_overlap_stops_at_the_local_cap() {
        // Four overlapping 0.8 dB ripples: each is individually negligible,
        // but their skirts stack — after the first removal the next would
        // move one region by ~1.5 dB, so the local guard stops the walk
        // with the rest unevaluated rather than assumed safe.
        let cluster = [400.0, 500.0, 630.0, 800.0];
        let filters: Vec<Biquad> =
            cluster.iter().map(|&f| peak(0.8, f, 1.0)).collect();
        let (freqs, mut verdicts) = nominate(&filters);
        assert!(
            verdicts.iter().all(|v| v.decision == VetoDecision::Remove),
            "test needs four nominated ripples"
        );
        let adjudication = adjudicate_veto_removals(
            filters.clone(),
            &mut verdicts,
            &freqs,
            &adjudicate_config(true),
        );
        assert!(
            (1..=2).contains(&adjudication.removed.len()),
            "walk should stop after one or two removals, removed {}",
            adjudication.removed.len()
        );
        assert!(
            adjudication.max_local_deviation_db <= 1.0 + 1e-9,
            "local drift unbounded: {}",
            adjudication.max_local_deviation_db
        );
        assert!(
            verdicts.iter().any(|v| v.acceptance.reason.contains("local deviation")),
            "rejection must name the local guard"
        );
    }

    #[test]
    fn declared_tight_budget_binds_before_default() {
        // Six spread-out ripples total ~2e-5 sones of drift: the default
        // quantum accepts them all, while a declared 1e-5 cap stops the
        // walk partway — the budget, not the heuristics, is the difference.
        let freqs_spread = [100.0, 300.0, 700.0, 1500.0, 3000.0, 6000.0];
        let filters: Vec<Biquad> =
            freqs_spread.iter().map(|&f| peak(0.8, f, 2.0)).collect();
        let (freqs, mut verdicts) = nominate(&filters);
        let all = adjudicate_veto_removals(
            filters.clone(),
            &mut verdicts,
            &freqs,
            &adjudicate_config(true),
        );
        assert_eq!(all.removed.len(), filters.len());

        let (_, mut verdicts) = nominate(&filters);
        let mut budgeted = adjudicate_config(true);
        budgeted.cumulative_cap_sones = Some(0.00001);
        let adjudication =
            adjudicate_veto_removals(filters.clone(), &mut verdicts, &freqs, &budgeted);
        assert!(
            (1..filters.len()).contains(&adjudication.removed.len()),
            "tight budget should stop the walk partway, removed {}",
            adjudication.removed.len()
        );
        assert!(
            verdicts.iter().any(|v| v.acceptance.reason.contains("cumulative")),
            "rejection must name the cumulative guard"
        );
        assert!(
            verdicts
                .iter()
                .any(|v| v.acceptance.thresholds.iter().any(|t| t.name == "pruning_budget_max_cumulative_delta")),
            "budget threshold must be recorded"
        );
    }

    #[test]
    fn rollback_reproduces_f0() {
        // Re-inserting removed filters at their stable indices must
        // reproduce the frozen composite bit-for-bit (up to float order).
        let filters = vec![
            peak(0.2, 400.0, 1.0),
            peak(3.0, 900.0, 1.0),
            peak(-0.3, 2500.0, 1.5),
        ];
        let (freqs, mut verdicts) = nominate(&filters);
        let adjudication =
            adjudicate_veto_removals(filters.clone(), &mut verdicts, &freqs, &adjudicate_config(true));
        assert!(!adjudication.removed.is_empty());
        // Kept filters preserve original relative order, so they fill the
        // non-removed slots left to right.
        let mut restored: Vec<Option<Biquad>> = vec![None; filters.len()];
        let mut kept_iter = adjudication.kept.iter();
        for (index, slot) in restored.iter_mut().enumerate() {
            if adjudication.removed.iter().any(|r| r.index == index) {
                continue;
            }
            *slot = kept_iter.next().cloned();
        }
        for removed in &adjudication.removed {
            restored[removed.index] = Some(removed.filter.clone());
        }
        let restored_filters: Vec<Biquad> =
            restored.into_iter().map(|slot| slot.expect("every slot filled")).collect();
        let f0 = composite_of(&filters, &freqs);
        let back = composite_of(&restored_filters, &freqs);
        let drift = f0
            .iter()
            .zip(back.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(drift < 1e-9, "rollback drifted by {drift}");
        assert_eq!(
            fingerprint_f0(&back, restored_filters.len()),
            adjudication.f0_reference_id
        );
    }

    #[test]
    fn verdict_filter_mismatch_never_removes() {
        let filters = vec![peak(0.2, 500.0, 1.0), peak(0.2, 700.0, 1.0)];
        let (_, mut verdicts) = nominate(&filters[..1]);
        let freqs = grid();
        let adjudication = adjudicate_veto_removals(
            filters.clone(),
            &mut verdicts,
            &freqs,
            &adjudicate_config(true),
        );
        assert_eq!(adjudication.kept.len(), 2, "ambiguous input keeps all");
        assert!(!adjudication.enforced);
        assert_eq!(
            verdicts[0].acceptance.enforcement,
            EnforcementState::NotEvaluated
        );
    }

    #[test]
    fn empty_set_adjudicates_cleanly() {
        let freqs = grid();
        let mut verdicts = Vec::new();
        let adjudication =
            adjudicate_veto_removals(Vec::new(), &mut verdicts, &freqs, &adjudicate_config(true));
        assert!(adjudication.kept.is_empty());
        assert!(adjudication.removed.is_empty());
        assert!(adjudication.f0_reference_id.starts_with("F0:"));
        assert!(adjudication.enforced);
    }

    #[test]
    fn veto_units_elimination_retains_finalist_for_adjudication() {
        // Loudness-only elimination never empties the set on its own: the
        // last filter survives even when negligible, because total
        // loudness alone cannot certify inaudibility. Identity is reached
        // only through post-pass adjudication with its local-deviation
        // guard (see sub_jnd_ripple_accepts_identity_with_stable_f0).
        let filters = vec![peak(0.1, 400.0, 1.0)];
        let freqs = grid();
        let kept = backward_eliminate_veto_units(filters, &freqs, 75.0, 0.05);
        assert_eq!(kept.len(), 1);
    }
}
