//! Advisory-only splice/bus/XO headroom assessment (Phase 2).
//!
//! Pure functions over measured curves. They predict whether a splice can
//! meet its target within the [`HeadroomBudget`](roomeq_model::headroom::HeadroomBudget)
//! before any optimizer spends headroom. Phase 2 emits advisories only; no
//! verdict changes. Phase 3 consumers act on these verdicts.
//!
//! The splice computation mirrors the route objective normalization
//! ([`bass_management_objective_with_target`](crate::topology::bass_management_objective_with_target)):
//! the sub must reach the target shifted by the main-band offset, so a
//! predicted shortfall matches what the differential-evolution trim search
//! would need. All inputs must share one frequency grid; mismatched grids
//! return `None` (the caller interpolates explicitly) rather than zipping
//! by index.

// Rust guideline compliant 2026-02-21

use crate::Curve;
use crate::topology::same_frequency_grid;
use roomeq_model::headroom::{HeadroomBudget, HeadroomStrategy};

/// Per-splice keeps-up verdict, in dB.
#[derive(Debug, Clone, PartialEq)]
pub struct SpliceHeadroomVerdict {
    /// Logical source channel the verdict applies to.
    pub source_channel: String,
    /// Crossover the splice window was built around.
    pub crossover_hz: f64,
    /// Sub boost that would lift the weakest splice point to the
    /// main-band-anchored target. Negative means the sub is already hot
    /// and must be cut instead.
    pub required_sub_boost_db: f64,
    /// Unfunded part of the requirement: `max(0, required - pool)`.
    /// Positive predicts the optimizer/safety stalemate (boost requested,
    /// safety cuts it, splice plays cold at every correction strength).
    pub shortfall_db: f64,
    /// True when the joint pool funds the requirement.
    pub keeps_up: bool,
    /// Strategy the doctrine selects from this verdict.
    pub strategy: HeadroomStrategy,
}

/// Assess one splice from branch magnitudes against the budget.
///
/// `main` is the high-passed main branch and `sub` the low-passed sub-array
/// branch, both pre-trim (trim is what is being budgeted) on one shared
/// grid. `applied_sub_boost_db` is MSO-applied boost already spent from the
/// same pool (use the contributing outputs' maximum to stay conservative).
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use roomeq_engine::Curve;
/// use roomeq_engine::bass_management::headroom_assess::assess_splice_headroom;
/// use roomeq_model::{RoomConfig, headroom::HeadroomBudget};
///
/// let freq = Array1::linspace(20.0, 1_000.0, 981);
/// let flat = |level: f64| Curve {
///     spl: Array1::from_elem(freq.len(), level),
///     phase: None,
///     freq: freq.clone(),
///     ..Curve::default()
/// };
/// let budget = HeadroomBudget::from_legacy_config(&RoomConfig::default());
/// let verdict =
///     assess_splice_headroom("L", &flat(70.0), &flat(70.0), None, 80.0, &budget, 0.0)
///         .expect("shared grid");
/// assert!(verdict.keeps_up);
/// ```
pub fn assess_splice_headroom(
    source: &str,
    main: &Curve,
    sub: &Curve,
    target: Option<&Curve>,
    crossover_hz: f64,
    budget: &HeadroomBudget,
    applied_sub_boost_db: f64,
) -> Option<SpliceHeadroomVerdict> {
    if !same_frequency_grid(&main.freq, &sub.freq)
        || target.is_some_and(|t| {
            !same_frequency_grid(&main.freq, &t.freq) || t.spl.len() != main.freq.len()
        })
        || main.spl.len() != main.freq.len()
        || sub.spl.len() != main.freq.len()
        || !crossover_hz.is_finite()
        || crossover_hz <= 0.0
    {
        return None;
    }
    let in_window =
        |f: f64| f >= (crossover_hz / 2.0).max(20.0) && f <= (crossover_hz * 2.0).min(2000.0);
    let in_reference = |f: f64| f >= crossover_hz * 2.0 && f <= (crossover_hz * 8.0).min(2000.0);
    let mut window_need = Vec::new();
    let mut reference_offset = Vec::new();
    for (i, frequency) in main.freq.iter().enumerate() {
        let (main_level, sub_level) = (main.spl[i], sub.spl[i]);
        if !main_level.is_finite() || !sub_level.is_finite() {
            return None;
        }
        let target_level = match target {
            Some(t) if !t.spl[i].is_finite() => return None,
            Some(t) => t.spl[i],
            None => 0.0,
        };
        if in_window(*frequency) {
            window_need.push((target_level, sub_level));
        }
        if in_reference(*frequency) {
            reference_offset.push(main_level - target_level);
        }
    }
    if window_need.len() < 2 || reference_offset.is_empty() {
        return None;
    }
    // Anchor: the sub must reach the target shifted by the main-band
    // offset (without a target, the main-band level itself).
    let offset = reference_offset.iter().sum::<f64>() / reference_offset.len() as f64;
    let required = window_need
        .iter()
        .map(|(target_level, sub_level)| target_level + offset - sub_level)
        .fold(f64::NEG_INFINITY, f64::max);
    let pool = budget.joint_route_trim_up_db(applied_sub_boost_db);
    let shortfall = (required - pool).max(0.0);
    let strategy = if required <= 0.0 {
        HeadroomStrategy::Balanced
    } else if shortfall <= 0.0 {
        HeadroomStrategy::SubFlexes
    } else {
        HeadroomStrategy::MainProtects
    };
    Some(SpliceHeadroomVerdict {
        source_channel: source.into(),
        crossover_hz,
        required_sub_boost_db: required,
        shortfall_db: shortfall,
        keeps_up: shortfall <= 0.0,
        strategy,
    })
}

/// Worst-case redirected-bus peak against the ceiling, in dBFS.
///
/// `peaks_dbfs` are the contributing inputs' routed peaks at one physical
/// output. Summed coherently (linear-domain sum): a conservative bound, so
/// `feasible == false` always deserves attention while `feasible == true`
/// still needs the realized safety replay for the verdict.
///
/// # Examples
///
/// ```
/// use roomeq_engine::bass_management::headroom_assess::assess_bus_headroom;
/// use roomeq_model::{RoomConfig, headroom::HeadroomBudget};
///
/// let budget = HeadroomBudget::from_legacy_config(&RoomConfig::default());
/// let single = assess_bus_headroom("Sub1", &[-6.0], &budget);
/// assert!(single.feasible);
/// let five = assess_bus_headroom("Sub1", &[-3.0; 5], &budget);
/// assert!(!five.feasible);
/// ```
pub fn assess_bus_headroom(
    output_role: &str,
    peaks_dbfs: &[f64],
    budget: &HeadroomBudget,
) -> BusHeadroomVerdict {
    let linear_sum: f64 = peaks_dbfs
        .iter()
        .filter(|peak| peak.is_finite())
        .map(|peak| 10_f64.powf(peak / 20.0))
        .sum();
    let peak_sum_dbfs = if linear_sum > 0.0 {
        20.0 * linear_sum.log10()
    } else {
        f64::NEG_INFINITY
    };
    let shortfall = (peak_sum_dbfs - budget.output_ceiling_dbfs).max(0.0);
    BusHeadroomVerdict {
        output_role: output_role.into(),
        peak_sum_dbfs,
        shortfall_db: shortfall,
        feasible: shortfall <= 0.0,
    }
}

/// Worst-case bus verdict for one physical output.
#[derive(Debug, Clone, PartialEq)]
pub struct BusHeadroomVerdict {
    /// Physical output the contributing inputs redirect into.
    pub output_role: String,
    /// Coherent linear sum of routed peaks, in dBFS.
    pub peak_sum_dbfs: f64,
    /// Amount by which the worst-case sum exceeds the ceiling.
    pub shortfall_db: f64,
    /// True when even the worst-case sum fits under the ceiling.
    pub feasible: bool,
}

/// Splice-window roughness at a candidate crossover, in dB.
///
/// Returns the widest peak-to-trough spread of either branch inside
/// `[candidate/2, candidate*2]`: high values mark XOs sitting on modes,
/// nulls, or rolloff edges (measured genelec: ~20 dB at 40 Hz from the
/// 41 Hz room mode). Phase 3c consumes this as a soft XO penalty; Phase 2
/// only reports it.
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use roomeq_engine::Curve;
/// use roomeq_engine::bass_management::headroom_assess::xo_window_roughness_db;
///
/// let freq = Array1::linspace(20.0, 1_000.0, 981);
/// let flat = Curve {
///     spl: Array1::from_elem(freq.len(), 70.0),
///     phase: None,
///     freq: freq.clone(),
///     ..Curve::default()
/// };
/// let roughness = xo_window_roughness_db(&flat, &flat, 80.0).expect("window");
/// assert!(roughness < 1e-9);
/// ```
pub fn xo_window_roughness_db(main: &Curve, sub: &Curve, candidate_hz: f64) -> Option<f64> {
    if !same_frequency_grid(&main.freq, &sub.freq)
        || main.spl.len() != main.freq.len()
        || sub.spl.len() != main.freq.len()
        || !candidate_hz.is_finite()
        || candidate_hz <= 0.0
    {
        return None;
    }
    let mut levels = Vec::new();
    for (i, frequency) in main.freq.iter().enumerate() {
        if *frequency >= (candidate_hz / 2.0).max(20.0)
            && *frequency <= (candidate_hz * 2.0).min(2000.0)
        {
            if !main.spl[i].is_finite() || !sub.spl[i].is_finite() {
                return None;
            }
            levels.push(main.spl[i]);
            levels.push(sub.spl[i]);
        }
    }
    if levels.len() < 4 {
        return None;
    }
    let max = levels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = levels.iter().copied().fold(f64::INFINITY, f64::min);
    Some(max - min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use roomeq_model::RoomConfig;

    fn curve_with_levels(levels: impl Fn(f64) -> f64) -> Curve {
        let freq = Array1::linspace(20.0, 1_000.0, 981);
        Curve {
            spl: freq.mapv(levels),
            phase: None,
            freq,
            ..Curve::default()
        }
    }

    fn fixture_budget() -> HeadroomBudget {
        // Fixture-like pool (measured unknown/kef: 6.0 dB).
        let mut budget = HeadroomBudget::from_legacy_config(&RoomConfig::default());
        budget.max_route_trim_up_db = 6.0;
        budget
    }

    #[test]
    fn balanced_splice_needs_no_boost() {
        let main = curve_with_levels(|_| 70.0);
        let sub = curve_with_levels(|_| 70.0);
        let verdict = assess_splice_headroom("L", &main, &sub, None, 80.0, &fixture_budget(), 0.0)
            .expect("shared grid");
        assert!(verdict.required_sub_boost_db.abs() < 1e-9);
        assert!(verdict.keeps_up);
        assert_eq!(verdict.strategy, HeadroomStrategy::Balanced);
    }

    #[test]
    fn cold_sub_within_pool_selects_sub_flexes() {
        let main = curve_with_levels(|_| 70.0);
        let sub = curve_with_levels(|f| {
            if (40.0..=160.0).contains(&f) {
                67.0
            } else {
                70.0
            }
        });
        let verdict = assess_splice_headroom("L", &main, &sub, None, 80.0, &fixture_budget(), 0.0)
            .expect("shared grid");
        assert!((verdict.required_sub_boost_db - 3.0).abs() < 1e-9);
        assert!(verdict.keeps_up);
        assert_eq!(verdict.strategy, HeadroomStrategy::SubFlexes);
    }

    #[test]
    fn cold_sub_beyond_pool_predicts_stalemate() {
        // Unknown-like: splice needs ~+14 dB, pool holds 6, MSO spent 6.
        let main = curve_with_levels(|_| 70.0);
        let sub = curve_with_levels(|f| {
            if (40.0..=160.0).contains(&f) {
                56.0
            } else {
                70.0
            }
        });
        let verdict = assess_splice_headroom("L", &main, &sub, None, 80.0, &fixture_budget(), 6.0)
            .expect("shared grid");
        assert!((verdict.required_sub_boost_db - 14.0).abs() < 1e-9);
        assert!((verdict.shortfall_db - 14.0).abs() < 1e-9);
        assert!(!verdict.keeps_up);
        assert_eq!(verdict.strategy, HeadroomStrategy::MainProtects);
    }

    #[test]
    fn hot_sub_reports_negative_requirement() {
        let main = curve_with_levels(|_| 70.0);
        let sub = curve_with_levels(|_| 76.0);
        let verdict = assess_splice_headroom("L", &main, &sub, None, 80.0, &fixture_budget(), 0.0)
            .expect("shared grid");
        assert!(verdict.required_sub_boost_db < 0.0);
        assert!(verdict.keeps_up);
        assert_eq!(verdict.strategy, HeadroomStrategy::Balanced);
    }

    #[test]
    fn target_shape_shifts_requirement() {
        let main = curve_with_levels(|_| 70.0);
        let sub = curve_with_levels(|_| 70.0);
        // +10 dB bass shelf below 100 Hz: the splice must rise 10 dB.
        let target = curve_with_levels(|f| if f < 100.0 { 80.0 } else { 70.0 });
        let verdict = assess_splice_headroom(
            "L",
            &main,
            &sub,
            Some(&target),
            80.0,
            &fixture_budget(),
            0.0,
        )
        .expect("shared grid");
        // Reference band (160-640 Hz) sits on target: offset 0, so the
        // window must rise the full shelf.
        assert!((verdict.required_sub_boost_db - 10.0).abs() < 1e-9);
        assert!(!verdict.keeps_up);
    }

    #[test]
    fn mismatched_grids_and_bands_decline() {
        let main = curve_with_levels(|_| 70.0);
        let mut shifted = main.clone();
        shifted.freq = shifted.freq.mapv(|f| f * 1.001);
        assert!(
            assess_splice_headroom("L", &main, &shifted, None, 80.0, &fixture_budget(), 0.0)
                .is_none()
        );
        let mut nonfinite = main.clone();
        nonfinite.spl[100] = f64::NAN;
        assert!(
            assess_splice_headroom("L", &main, &nonfinite, None, 80.0, &fixture_budget(), 0.0)
                .is_none()
        );
        assert!(
            assess_splice_headroom("L", &main, &main, None, f64::NAN, &fixture_budget(), 0.0)
                .is_none()
        );
    }

    #[test]
    fn bus_bound_flags_multi_channel_redirect_overload() {
        let budget = fixture_budget();
        // Kef-like: five -3 dBFS inputs cohere to ~+11 dBFS over a 0 ceiling.
        let overloaded = assess_bus_headroom("Sub1", &[-3.0; 5], &budget);
        assert!(!overloaded.feasible);
        assert!(overloaded.shortfall_db > 10.0);
        let single = assess_bus_headroom("Sub1", &[-6.0], &budget);
        assert!(single.feasible);
        assert_eq!(single.shortfall_db, 0.0);
    }

    #[test]
    fn xo_roughness_marks_modal_candidates() {
        // Genelec-like: +20 dB mode at 41 Hz.
        let main = curve_with_levels(|f| {
            if (40.0..=42.0).contains(&f) {
                95.0
            } else {
                75.0
            }
        });
        let sub = curve_with_levels(|_| 75.0);
        let modal = xo_window_roughness_db(&main, &sub, 40.0).expect("window");
        assert!(modal > 19.0, "modal roughness: {modal}");
        let clean = xo_window_roughness_db(&main, &sub, 100.0).expect("window");
        assert!(clean < 1e-9, "clean roughness: {clean}");
    }
}
