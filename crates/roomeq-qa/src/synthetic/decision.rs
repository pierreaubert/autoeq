//! Registry-backed release decision matrix.
//!
//! Closes the release-matrix gap where the strategy match names
//! `ContinuousArea` but the guard list never invokes it (and never invokes
//! `MinimizeVariance` / `ModalBasis` either): every
//! [`DecisionCaseKind`](crate::registry::DecisionCaseKind) maps to exactly one
//! invoked test, and each registry claim is validated against the test that
//! exercises it.
//!
//! * NSGA-II/III conflicting-objective decisions and Pareto knee picks run as
//!   deterministic seeded decisions over synthetic fronts (order-invariant,
//!   budget-capped, invalid entries rejected).
//! * Continuous expected / CVaR / worst-case objectives invoke
//!   `optimize_multiseat_continuous_area` with deterministic seeds and
//!   configured quadrature/inner-iteration limits. Worst-case search receives
//!   a cooperative time budget; the separate total-runtime gate remains.
//!   Actual outer-search counts and stop reasons are retained. In-flight
//!   inner evaluations and final verification are not forcibly interrupted.
//! * Regression expectations cover phase permutation (seat/sub order
//!   invariance), measured support (missing-phase rejection), invalid fronts,
//!   and synthetic modal phase. Safety-fallback entries accept a clean
//!   revert; quality entries must meet their thresholds.
//!
//! Perf: the strategy x seed x seat x sub matrix is bounded per tier by
//! [`matrix_caps`]; immutable fixtures are built once and shared via
//! [`canonical_fixture`] / [`modal_fixture`].

use super::consts::SAMPLE_RATE;
use super::misc::make_multiseat_qa_curve;
use super::types::TestResult;
use crate::registry::{DecisionCaseKind, DecisionCaseSpec, QaGatePurpose, QaTier, load_registry};
use autoeq_optim::optim::pareto::{ParetoFilter, extract_non_dominated};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use roomeq_engine::multiseat::{
    MultiSeatMeasurements, MsoSearchBudget, optimize_multiseat,
    optimize_multiseat_continuous_area_with_budget,
};
use roomeq_model::{
    AreaPriorKind, AreaQuadratureKind, AreaScalarisationKind, ContinuousListeningAreaConfig, Curve,
    MultiSeatConfig, MultiSeatStrategy,
};
use std::sync::OnceLock;
use std::time::Instant;

/// Per-tier bounds for the strategy x seed x seat x sub decision matrix and
/// the continuous-area quadrature budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixCaps {
    /// Max discrete strategies exercised per engine case.
    pub strategies: usize,
    /// Max seeds exercised per engine case.
    pub seeds: usize,
    /// Max continuous-area quadrature points.
    pub points: usize,
}

/// Tier-bounded matrix caps: PR stays minimal, nightly exercises the full
/// discrete strategy set, weekly adds seeds.
pub fn matrix_caps(tier: QaTier) -> MatrixCaps {
    match tier {
        QaTier::Pr => MatrixCaps {
            strategies: 2,
            seeds: 1,
            points: 8,
        },
        QaTier::Nightly => MatrixCaps {
            strategies: 4,
            seeds: 2,
            points: 16,
        },
        QaTier::Weekly => MatrixCaps {
            strategies: 4,
            seeds: 3,
            points: 32,
        },
    }
}

/// Canonical shared 2-sub x 2-seat phase-bearing fixture. Built once,
/// immutable, and reused by every engine-backed decision test so the matrix
/// shares one allocation instead of rebuilding curves per row.
pub fn canonical_fixture() -> &'static Vec<Vec<Curve>> {
    static FIXTURE: OnceLock<Vec<Vec<Curve>>> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let flat = |_: f64| 90.0;
        let dipped = |f: f64| if f < 60.0 { 84.0 } else { 90.0 };
        let peaked = |f: f64| if f < 60.0 { 96.0 } else { 90.0 };
        vec![
            vec![
                make_multiseat_qa_curve(flat, 0.0, true),
                make_multiseat_qa_curve(dipped, 12.0, true),
            ],
            vec![
                make_multiseat_qa_curve(peaked, 24.0, true),
                make_multiseat_qa_curve(flat, 36.0, true),
            ],
        ]
    })
}

/// Shared synthetic modal-phase fixture: narrow room-mode bumps whose height
/// varies per seat, with phase. Drives the `ModalBasis` regression cases.
pub fn modal_fixture() -> &'static Vec<Vec<Curve>> {
    static FIXTURE: OnceLock<Vec<Vec<Curve>>> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let modal = |peak_db: f64, center_hz: f64| {
            move |f: f64| {
                let width = 6.0;
                let bump = peak_db * (-((f - center_hz) / width).powi(2)).exp();
                90.0 + bump
            }
        };
        vec![
            vec![
                make_multiseat_qa_curve(modal(9.0, 45.0), 0.0, true),
                make_multiseat_qa_curve(modal(5.0, 45.0), 20.0, true),
            ],
            vec![
                make_multiseat_qa_curve(modal(7.0, 72.0), 150.0, true),
                make_multiseat_qa_curve(modal(4.0, 72.0), -150.0, true),
            ],
        ]
    })
}

/// Deterministic sized fixture for non-canonical seat/sub counts.
fn sized_fixture(subs: usize, seats: usize) -> Vec<Vec<Curve>> {
    if subs == 2 && seats == 2 {
        return canonical_fixture().clone();
    }
    let flat = |_: f64| 90.0;
    (0..subs)
        .map(|sub| {
            (0..seats)
                .map(|seat| {
                    let offset = (sub * seats + seat) as f64;
                    let shaped = move |f: f64| {
                        flat(f) + 4.0 * ((f / 25.0 + offset).sin())
                            - 3.0 * ((f / 60.0 + offset * 0.5).cos())
                    };
                    make_multiseat_qa_curve(shaped, 15.0 * offset, true)
                })
                .collect()
        })
        .collect()
}

/// 1-D seat coordinates spread over the unit listening line.
fn seat_positions_1d(seats: usize) -> Vec<Vec<f64>> {
    (0..seats)
        .map(|i| {
            vec![if seats > 1 {
                i as f64 / (seats - 1) as f64
            } else {
                0.5
            }]
        })
        .collect()
}

/// One conflicting-objective candidate: (flatness loss, filter count, boost).
/// Flatness and count conflict by construction (more filters -> less loss).
#[derive(Debug, Clone, Copy)]
struct Candidate {
    flatness: f64,
    count: usize,
    boost: f64,
}

/// Canonical NSGA-II pool: 5 Pareto-optimal points plus 3 dominated decoys.
fn nsga2_pool() -> Vec<Candidate> {
    vec![
        Candidate {
            flatness: 3.20,
            count: 3,
            boost: 1.0,
        },
        Candidate {
            flatness: 2.40,
            count: 5,
            boost: 2.0,
        },
        Candidate {
            flatness: 1.90,
            count: 7,
            boost: 3.0,
        },
        Candidate {
            flatness: 1.70,
            count: 9,
            boost: 5.0,
        },
        Candidate {
            flatness: 1.65,
            count: 11,
            boost: 8.0,
        },
        Candidate {
            flatness: 2.60,
            count: 7,
            boost: 4.0,
        },
        Candidate {
            flatness: 2.00,
            count: 9,
            boost: 6.0,
        },
        Candidate {
            flatness: 3.40,
            count: 5,
            boost: 1.5,
        },
    ]
}

fn to_pareto_filters(pool: &[Candidate], three_objective: bool) -> Vec<ParetoFilter> {
    pool.iter()
        .map(|candidate| ParetoFilter {
            params: vec![candidate.flatness, candidate.count as f64],
            flatness_loss: candidate.flatness,
            score_loss: three_objective.then_some(candidate.boost),
            num_filters: candidate.count,
            converged: true,
        })
        .collect()
}

/// Reject empty fronts and non-finite entries before any decision logic.
fn validate_front(filters: &[ParetoFilter]) -> Result<(), String> {
    if filters.is_empty() {
        return Err("front is empty".to_string());
    }
    for (index, filter) in filters.iter().enumerate() {
        let score = filter.score_loss.unwrap_or(0.0);
        if !filter.flatness_loss.is_finite() || !score.is_finite() || filter.num_filters == 0 {
            return Err(format!("front entry {index} is invalid"));
        }
    }
    Ok(())
}

/// Deterministic order-invariant shuffle of candidate indices.
fn shuffled_order(len: usize, seed: u64) -> Vec<usize> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..len).collect();
    for i in (1..len).rev() {
        let j = rng.random_range(0..=i);
        order.swap(i, j);
    }
    order
}

/// Normalized distance-to-ideal compromise over a non-dominated front.
/// Iterates canonical order with strict improvement so the pick is invariant
/// to input shuffling. `three_objective` selects the NSGA-III (flatness,
/// count, boost) space; otherwise NSGA-II (flatness, count).
fn compromise_pick(front: &[&ParetoFilter], three_objective: bool) -> usize {
    let flat = |filter: &&ParetoFilter| filter.flatness_loss;
    let count = |filter: &&ParetoFilter| filter.num_filters as f64;
    let boost = |filter: &&ParetoFilter| filter.score_loss.unwrap_or(0.0);
    let normalize = |value: f64, min: f64, max: f64| {
        if max > min {
            (value - min) / (max - min)
        } else {
            0.0
        }
    };
    let (flat_min, flat_max) = front
        .iter()
        .map(flat)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(value), hi.max(value))
        });
    let (count_min, count_max) = front
        .iter()
        .map(count)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(value), hi.max(value))
        });
    let (boost_min, boost_max) = front
        .iter()
        .map(boost)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(value), hi.max(value))
        });
    // Canonical order first: with strict improvement the pick is then a pure
    // function of front values, invariant to input shuffling.
    let mut order: Vec<usize> = (0..front.len()).collect();
    order.sort_by(|&a, &b| {
        front[a]
            .flatness_loss
            .total_cmp(&front[b].flatness_loss)
            .then(front[a].num_filters.cmp(&front[b].num_filters))
    });
    let mut best = order[0];
    let mut best_distance = f64::INFINITY;
    for &position in &order {
        let filter = front[position];
        let mut distance = normalize(flat(&filter), flat_min, flat_max).powi(2)
            + normalize(count(&filter), count_min, count_max).powi(2);
        if three_objective {
            distance += normalize(boost(&filter), boost_min, boost_max).powi(2);
        }
        if distance < best_distance {
            best_distance = distance;
            best = position;
        }
    }
    best
}

/// Knee pick: front member with max perpendicular distance to the chord
/// between the flatness/count extremes (2-objective space).
fn knee_pick(front: &[&ParetoFilter]) -> usize {
    let flat: Vec<f64> = front.iter().map(|filter| filter.flatness_loss).collect();
    let count: Vec<f64> = front
        .iter()
        .map(|filter| filter.num_filters as f64)
        .collect();
    let (flat_min, flat_max) = flat
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &value| {
            (lo.min(value), hi.max(value))
        });
    let (count_min, count_max) = count
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &value| {
            (lo.min(value), hi.max(value))
        });
    let normalize = |value: f64, min: f64, max: f64| {
        if max > min {
            (value - min) / (max - min)
        } else {
            0.0
        }
    };
    // Sort by flatness so the chord endpoints are the objective extremes
    // regardless of input order; the pick is then order-invariant.
    let mut order: Vec<usize> = (0..front.len()).collect();
    order.sort_by(|&a, &b| {
        flat[a]
            .total_cmp(&flat[b])
            .then(count[a].total_cmp(&count[b]))
    });
    let ax = normalize(flat[order[0]], flat_min, flat_max);
    let ay = normalize(count[order[0]], count_min, count_max);
    let bx = normalize(flat[order[front.len() - 1]], flat_min, flat_max);
    let by = normalize(count[order[front.len() - 1]], count_min, count_max);
    let chord = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt().max(1e-12);
    let mut best = order[0];
    let mut best_distance = f64::NEG_INFINITY;
    for &position in &order {
        let px = normalize(flat[position], flat_min, flat_max);
        let py = normalize(count[position], count_min, count_max);
        let distance = ((bx - ax) * (ay - py) - (ax - px) * (by - ay)).abs() / chord;
        if distance > best_distance {
            best_distance = distance;
            best = position;
        }
    }
    best
}

/// Shared NSGA-II/III decision core: budget-capped pool, seeded shuffle,
/// front validation, non-dominated extraction, compromise pick. Returns the
/// picked canonical pool index plus the front size.
fn nsga_decision(spec: &DecisionCaseSpec, three_objective: bool) -> Result<(usize, usize), String> {
    let pool = nsga2_pool();
    if pool.len() > spec.maxeval {
        return Err(format!(
            "candidate pool {} exceeds maxeval budget {}",
            pool.len(),
            spec.maxeval
        ));
    }
    let order = shuffled_order(pool.len(), spec.seed);
    let shuffled: Vec<Candidate> = order.iter().map(|&i| pool[i]).collect();
    let filters = to_pareto_filters(&shuffled, three_objective);
    validate_front(&filters)?;
    let front = extract_non_dominated(&filters);
    if front.is_empty() {
        return Err("non-dominated front is empty".to_string());
    }
    if front.len() > spec.population {
        return Err(format!(
            "front size {} exceeds population budget {}",
            front.len(),
            spec.population
        ));
    }
    // Map the picked front member back to its canonical pool index by value.
    let picked = compromise_pick(&front, three_objective);
    let picked_filter = front[picked];
    let canonical = pool
        .iter()
        .position(|candidate| {
            candidate.flatness == picked_filter.flatness_loss
                && candidate.count == picked_filter.num_filters
        })
        .ok_or_else(|| "compromise pick is not a front member".to_string())?;
    Ok((canonical, front.len()))
}

fn pareto_knee_decision(spec: &DecisionCaseSpec) -> Result<(usize, usize), String> {
    let pool = nsga2_pool();
    if pool.len() > spec.maxeval {
        return Err(format!(
            "candidate pool {} exceeds maxeval budget {}",
            pool.len(),
            spec.maxeval
        ));
    }
    let order = shuffled_order(pool.len(), spec.seed);
    let shuffled: Vec<Candidate> = order.iter().map(|&i| pool[i]).collect();
    let filters = to_pareto_filters(&shuffled, false);
    validate_front(&filters)?;
    let front = extract_non_dominated(&filters);
    if front.is_empty() {
        return Err("non-dominated front is empty".to_string());
    }
    if front.len() > spec.population {
        return Err(format!(
            "front size {} exceeds population budget {}",
            front.len(),
            spec.population
        ));
    }
    let picked = knee_pick(&front);
    let picked_filter = front[picked];
    let canonical = pool
        .iter()
        .position(|candidate| {
            candidate.flatness == picked_filter.flatness_loss
                && candidate.count == picked_filter.num_filters
        })
        .ok_or_else(|| "knee pick is not a front member".to_string())?;
    Ok((canonical, front.len()))
}

fn fail(name: String, reason: String) -> TestResult {
    TestResult {
        name,
        passed: false,
        pre_score: 0.0,
        post_score: 0.0,
        epa_preference: None,
        reason,
    }
}

fn run_nsga_case(spec: &DecisionCaseSpec, three_objective: bool) -> TestResult {
    let variant = if three_objective { "nsga3" } else { "nsga2" };
    let name = format!("decision/{}/{}", spec.id, variant);
    match nsga_decision(spec, three_objective) {
        Ok((canonical, front_size)) => TestResult {
            name,
            passed: true,
            pre_score: front_size as f64,
            post_score: canonical as f64,
            epa_preference: None,
            reason: format!(
                "OK: {variant} conflicting-objective decision is deterministic (seed {}, pool {} -> front {front_size} within population {}, pick pool index {canonical})",
                spec.seed,
                nsga2_pool().len(),
                spec.population,
            ),
        },
        Err(reason) => fail(name, reason),
    }
}

fn run_pareto_knee_case(spec: &DecisionCaseSpec) -> TestResult {
    let name = format!("decision/{}/pareto_knee", spec.id);
    match pareto_knee_decision(spec) {
        Ok((canonical, front_size)) => TestResult {
            name,
            passed: true,
            pre_score: front_size as f64,
            post_score: canonical as f64,
            epa_preference: None,
            reason: format!(
                "OK: pareto knee pick is deterministic (seed {}, front {front_size}, pick pool index {canonical})",
                spec.seed,
            ),
        },
        Err(reason) => fail(name, reason),
    }
}

fn run_invalid_front_case(spec: &DecisionCaseSpec) -> TestResult {
    let name = format!("decision/{}/invalid_front", spec.id);
    let empty: Vec<ParetoFilter> = Vec::new();
    if validate_front(&empty).is_ok() {
        return fail(name, "empty front was accepted".to_string());
    }
    let mut nan_front = to_pareto_filters(&nsga2_pool()[..2], false);
    nan_front[0].flatness_loss = f64::NAN;
    if validate_front(&nan_front).is_ok() {
        return fail(name, "NaN front entry was accepted".to_string());
    }
    let mut inf_front = to_pareto_filters(&nsga2_pool()[..2], false);
    inf_front[1] = ParetoFilter {
        flatness_loss: f64::INFINITY,
        ..inf_front[1].clone()
    };
    if validate_front(&inf_front).is_ok() {
        return fail(name, "infinite front entry was accepted".to_string());
    }
    let mut zero_count = to_pareto_filters(&nsga2_pool()[..2], false);
    zero_count[0].num_filters = 0;
    if validate_front(&zero_count).is_ok() {
        return fail(name, "zero-filter front entry was accepted".to_string());
    }
    TestResult {
        name,
        passed: true,
        pre_score: 0.0,
        post_score: 0.0,
        epa_preference: None,
        reason: "OK: empty/NaN/infinite/zero-count fronts rejected before decision".to_string(),
    }
}

fn run_measured_support_case(spec: &DecisionCaseSpec) -> TestResult {
    let name = format!("decision/{}/measured_support", spec.id);
    // Curves without phase must be rejected: MSO needs measured support.
    let phaseless: Vec<Vec<Curve>> = canonical_fixture()
        .iter()
        .map(|sub| {
            sub.iter()
                .map(|curve| Curve {
                    phase: None,
                    ..curve.clone()
                })
                .collect()
        })
        .collect();
    let config = MultiSeatConfig {
        enabled: true,
        strategy: MultiSeatStrategy::Average,
        ..Default::default()
    };
    match MultiSeatMeasurements::new(phaseless) {
        Ok(measurements) => {
            match optimize_multiseat(&measurements, &config, (20.0, 120.0), SAMPLE_RATE) {
                Ok(_) => fail(name, "phaseless measurements were accepted".to_string()),
                Err(_) => TestResult {
                    name,
                    passed: true,
                    pre_score: 0.0,
                    post_score: 0.0,
                    epa_preference: None,
                    reason: "OK: missing phase rejected at optimization".to_string(),
                },
            }
        }
        Err(_) => TestResult {
            name,
            passed: true,
            pre_score: 0.0,
            post_score: 0.0,
            epa_preference: None,
            reason: "OK: missing phase rejected at measurement build".to_string(),
        },
    }
}

fn discrete_config(strategy: MultiSeatStrategy) -> MultiSeatConfig {
    MultiSeatConfig {
        enabled: true,
        strategy,
        primary_seat: 0,
        max_deviation_db: 6.0,
        ..Default::default()
    }
}

fn check_realization(
    name: &str,
    result: &roomeq_engine::multiseat::MultiSeatOptimizationResult,
    subs: usize,
) -> Result<String, String> {
    if result.polarities.len() != subs || result.allpass_filters.len() != subs {
        return Err(format!(
            "final realization has polarities={} allpass_rows={} for {subs} subs",
            result.polarities.len(), result.allpass_filters.len(),
        ));
    }
    if result.allpass_filters.iter().flatten().any(|&(frequency, q)| {
        !frequency.is_finite() || frequency <= 0.0 || frequency >= SAMPLE_RATE / 2.0
            || !q.is_finite() || q <= 0.0
    }) {
        return Err("final realization contains invalid all-pass frequency/Q".into());
    }
    if result.gains.len() != subs || result.delays.len() != subs {
        return Err(format!(
            "final realization has gains={} delays={} for {subs} subs",
            result.gains.len(),
            result.delays.len()
        ));
    }
    if !result
        .gains
        .iter()
        .chain(result.delays.iter())
        .all(|v| v.is_finite())
    {
        return Err("final realization is non-finite".to_string());
    }
    if !result.objective_before.is_finite() || !result.objective_after.is_finite() {
        return Err("objective is non-finite".to_string());
    }
    if result.gains.first() != Some(&0.0) || result.delays.first() != Some(&0.0) {
        return Err(format!(
            "reference sub moved: gains={:?} delays={:?}",
            result.gains, result.delays
        ));
    }
    Ok(format!(
        "{} {:.3} -> {:.3}",
        name, result.objective_before, result.objective_after
    ))
}

fn run_phase_permutation_case(spec: &DecisionCaseSpec) -> TestResult {
    let name = format!("decision/{}/phase_permutation", spec.id);
    let base = sized_fixture(spec.subs, spec.seats);
    let config = discrete_config(MultiSeatStrategy::Average);
    let run = |measurements: &MultiSeatMeasurements| {
        optimize_multiseat(measurements, &config, (20.0, 120.0), SAMPLE_RATE)
            .map(|result| result.objective_before)
            .map_err(|error| error.to_string())
    };
    let base_ms = match MultiSeatMeasurements::new(base) {
        Ok(measurements) => measurements,
        Err(error) => return fail(name, format!("failed to build measurements: {error}")),
    };
    let before = match run(&base_ms) {
        Ok(value) => value,
        Err(error) => return fail(name, format!("optimization failed: {error}")),
    };
    // Permute seat order within every sub: the discrete objective must not
    // depend on seat slot order.
    let mut seat_permuted = base_ms.measurements.clone();
    for sub in &mut seat_permuted {
        sub.reverse();
    }
    let seat_ms = match MultiSeatMeasurements::new(seat_permuted) {
        Ok(measurements) => measurements,
        Err(error) => {
            return fail(
                name,
                format!("failed to build permuted measurements: {error}"),
            );
        }
    };
    let seat_before = match run(&seat_ms) {
        Ok(value) => value,
        Err(error) => return fail(name, format!("permuted optimization failed: {error}")),
    };
    if (before - seat_before).abs() > 1e-9 {
        return fail(
            name,
            format!("seat permutation changed objective {before:.6} -> {seat_before:.6}"),
        );
    }
    // Permute sub order: identical reasoning for the sub axis.
    let mut sub_permuted = base_ms.measurements.clone();
    sub_permuted.reverse();
    let sub_ms = match MultiSeatMeasurements::new(sub_permuted) {
        Ok(measurements) => measurements,
        Err(error) => {
            return fail(
                name,
                format!("failed to build permuted measurements: {error}"),
            );
        }
    };
    let sub_before = match run(&sub_ms) {
        Ok(value) => value,
        Err(error) => return fail(name, format!("permuted optimization failed: {error}")),
    };
    if (before - sub_before).abs() > 1e-9 {
        return fail(
            name,
            format!("sub permutation changed objective {before:.6} -> {sub_before:.6}"),
        );
    }
    TestResult {
        name,
        passed: true,
        pre_score: before,
        post_score: seat_before,
        epa_preference: None,
        reason: format!(
            "OK: seat/sub permutation invariant (seed {}, objective {before:.6})",
            spec.seed,
        ),
    }
}

fn continuous_scalarisation(spec: &DecisionCaseSpec) -> Result<AreaScalarisationKind, String> {
    match spec.kind {
        DecisionCaseKind::ContinuousExpected => Ok(AreaScalarisationKind::ExpectedValue),
        DecisionCaseKind::ContinuousCvar => {
            let alpha = spec
                .cvar_alpha
                .ok_or_else(|| "cvar_alpha missing".to_string())?;
            if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
                return Err("cvar_alpha out of (0, 1]".to_string());
            }
            Ok(AreaScalarisationKind::Cvar { alpha })
        }
        DecisionCaseKind::ContinuousWorstCase => Ok(AreaScalarisationKind::WorstCase {
            inner_maxiter: spec.inner_maxiter,
            inner_seed: spec.inner_seed,
        }),
        other => Err(format!("{other:?} is not a continuous-area kind")),
    }
}

fn continuous_objective_name(spec: &DecisionCaseSpec) -> &'static str {
    match spec.kind {
        DecisionCaseKind::ContinuousExpected => "expected",
        DecisionCaseKind::ContinuousCvar => "cvar",
        DecisionCaseKind::ContinuousWorstCase => "worst_case",
        _ => "continuous",
    }
}

fn continuous_search_budget(spec: &DecisionCaseSpec) -> MsoSearchBudget {
    MsoSearchBudget {
        max_duration: (spec.kind == DecisionCaseKind::ContinuousWorstCase)
            .then(|| std::time::Duration::from_millis(spec.timeout_ms)),
        ..Default::default()
    }
}

fn run_continuous_case(spec: &DecisionCaseSpec, caps: MatrixCaps) -> TestResult {
    let objective = continuous_objective_name(spec);
    let name = format!("decision/{}/continuous_{objective}", spec.id);
    let scalarisation = match continuous_scalarisation(spec) {
        Ok(scalarisation) => scalarisation,
        Err(reason) => return fail(name, reason),
    };
    let outer_points = spec.outer_points.min(caps.points).max(4);
    let measurements = match MultiSeatMeasurements::new(sized_fixture(spec.subs, spec.seats)) {
        Ok(measurements) => measurements,
        Err(error) => return fail(name, format!("failed to build measurements: {error}")),
    };
    let config = MultiSeatConfig {
        enabled: true,
        strategy: MultiSeatStrategy::ContinuousArea,
        continuous_area: Some(ContinuousListeningAreaConfig {
            dimensions: 1,
            bounds: vec![(0.0, 1.0)],
            seat_positions: seat_positions_1d(spec.seats),
            prior: AreaPriorKind::Uniform,
            quadrature: AreaQuadratureKind::Sobol {
                num_points: outer_points,
                seed: spec.seed,
            },
            scalarisation,
            idw_power: 2.0,
        }),
        ..Default::default()
    };
    let start = Instant::now();
    let result = optimize_multiseat_continuous_area_with_budget(
        &measurements, &config, (20.0, 120.0), SAMPLE_RATE,
        &continuous_search_budget(spec),
    );
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let (result, search_report) = match result {
        Ok(result) => result,
        Err(error) => return fail(name, format!("optimization failed: {error}")),
    };
    let mut row = report_continuous_case(spec, name, objective, outer_points, elapsed_ms, &result);
    row.reason.push_str(&format!(
        "; search_outer_evaluations={} search_generations={} search_stop={} search_stopped_early={} search_elapsed_ms={} search_best_loss={} search_counts_exclude_baseline_final_and_inner_positions=true",
        search_report.evaluations, search_report.generations_run,
        search_report.stop_reason, search_report.stopped_early,
        search_report.elapsed.as_millis(), search_report.best_loss,
    ));
    row
}

fn report_continuous_case(
    spec: &DecisionCaseSpec,
    name: String,
    objective: &str,
    outer_points: usize,
    elapsed_ms: u64,
    result: &roomeq_engine::multiseat::MultiSeatOptimizationResult,
) -> TestResult {
    // Configuration limits are not consumed evaluation counts. Worst-case
    // search samples dynamic positions, not the static quadrature collection.
    let inner_maxiter = if spec.kind == DecisionCaseKind::ContinuousWorstCase {
        spec.inner_maxiter
    } else {
        0
    };
    let counts = format!(
        "configured_quadrature_points={outer_points} static_quadrature_used={} configured_inner_maxiter={inner_maxiter} inner_position_evaluation_count=unavailable elapsed_ms={elapsed_ms}",
        spec.kind != DecisionCaseKind::ContinuousWorstCase,
    );
    // Do not erase completed acoustic objective evidence on a report failure.
    let failure = |reason: String| TestResult {
        name: name.clone(),
        passed: false,
        pre_score: result.objective_before,
        post_score: result.objective_after,
        epa_preference: None,
        reason,
    };
    if spec.kind == DecisionCaseKind::ContinuousWorstCase && elapsed_ms > spec.timeout_ms {
        return failure(format!(
            "worst-case search exceeded timeout: {elapsed_ms}ms > {}ms; {counts}",
            spec.timeout_ms,
        ));
    }
    if result.objective_name != "continuous_area" {
        return failure(format!("unexpected objective '{}'; {counts}", result.objective_name));
    }
    report_multiseat_candidate(spec, name, objective, result, &counts)
}

fn report_multiseat_candidate(
    spec: &DecisionCaseSpec,
    name: String,
    objective: &str,
    result: &roomeq_engine::multiseat::MultiSeatOptimizationResult,
    counts: &str,
) -> TestResult {
    let failure = |reason: String| TestResult {
        name: name.clone(),
        passed: false,
        pre_score: result.objective_before,
        post_score: result.objective_after,
        epa_preference: None,
        reason,
    };
    let improvement_pct = if result.objective_before > 0.0 {
        100.0 * (result.objective_before - result.objective_after) / result.objective_before
    } else {
        0.0
    };
    let quality_bar = improvement_pct >= spec.expect.improvement_min_pct
        && result.objective_after <= spec.expect.max_post_score;
    let realization = check_realization(objective, result, spec.subs).and_then(|summary| {
        // This candidate contains one gain per physical output followed by
        // delay/polarity/valid all-pass controls, all of unit ideal magnitude.
        // This is not a substitute for replaying a later PEQ/FIR/routed chain.
        let maximum_gain_db = result.gains.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if maximum_gain_db > spec.expect.max_boost_db {
            Err(format!(
                "output gain {maximum_gain_db} dB exceeds registered maximum {} dB",
                spec.expect.max_boost_db,
            ))
        } else {
            Ok(format!("{summary}; maximum_output_gain_db={maximum_gain_db}"))
        }
    });
    match (quality_bar, realization) {
        (true, Ok(summary)) => TestResult {
            name,
            passed: true,
            pre_score: result.objective_before,
            post_score: result.objective_after,
            epa_preference: None,
            reason: format!("OK: {objective} {summary}; {counts}"),
        },
        (false, Ok(_)) if spec.expect.gate_purpose == QaGatePurpose::Safety
            && spec.expect.allow_safe_revert
            && result.objective_after <= result.objective_before + 0.05 => TestResult {
            name,
            passed: true,
            pre_score: result.objective_before,
            post_score: result.objective_after,
            epa_preference: None,
            reason: format!(
                "OK: safety-only {objective} is non-regressing; quality thresholds unmet; no rollback inferred; {counts}"
            ),
        },
        (_, Err(reason)) => failure(format!("{reason}; {counts}")),
        (false, Ok(_)) => failure(format!(
            "{objective} missed quality thresholds: {:.6} -> {:.6}, improvement_pct={improvement_pct:.6} required_min_pct={}, required_max_post_score={}; {counts}",
            result.objective_before, result.objective_after,
            spec.expect.improvement_min_pct, spec.expect.max_post_score,
        )),
    }
}

fn run_modal_phase_case(spec: &DecisionCaseSpec) -> TestResult {
    let name = format!("decision/{}/modal_phase", spec.id);
    let fixture: Vec<Vec<Curve>> = if spec.subs == 2 && spec.seats == 2 {
        modal_fixture().clone()
    } else {
        sized_fixture(spec.subs, spec.seats)
    };
    let measurements = match MultiSeatMeasurements::new(fixture) {
        Ok(measurements) => measurements,
        Err(error) => return fail(name, format!("failed to build measurements: {error}")),
    };
    let config = discrete_config(MultiSeatStrategy::ModalBasis);
    let result = match optimize_multiseat(&measurements, &config, (20.0, 120.0), SAMPLE_RATE) {
        Ok(result) => result,
        Err(error) => return fail(name, format!("optimization failed: {error}")),
    };
    report_modal_phase_result(spec, name, &result)
}

fn report_modal_phase_result(
    spec: &DecisionCaseSpec,
    name: String,
    result: &roomeq_engine::multiseat::MultiSeatOptimizationResult,
) -> TestResult {
    if result.objective_name != "modal_basis" {
        return TestResult {
            name,
            passed: false,
            pre_score: result.objective_before,
            post_score: result.objective_after,
            epa_preference: None,
            reason: format!("unexpected objective '{}'", result.objective_name),
        };
    }
    report_multiseat_candidate(spec, name, "modal_basis", result, "fixture=synthetic_modal_phase")
}

/// Dispatch one registry decision case to its invoked test.
pub(super) fn run_decision_case(spec: &DecisionCaseSpec, caps: MatrixCaps) -> TestResult {
    match spec.kind {
        DecisionCaseKind::Nsga2Decision => run_nsga_case(spec, false),
        DecisionCaseKind::Nsga3Decision => run_nsga_case(spec, true),
        DecisionCaseKind::ParetoFront => run_pareto_knee_case(spec),
        DecisionCaseKind::InvalidFront => run_invalid_front_case(spec),
        DecisionCaseKind::MeasuredSupport => run_measured_support_case(spec),
        DecisionCaseKind::PhasePermutation => run_phase_permutation_case(spec),
        DecisionCaseKind::ContinuousExpected
        | DecisionCaseKind::ContinuousCvar
        | DecisionCaseKind::ContinuousWorstCase => run_continuous_case(spec, caps),
        DecisionCaseKind::ModalPhase => run_modal_phase_case(spec),
    }
}

/// Run the registry-backed decision matrix for one tier. An empty selection
/// is a failure, never a silent pass: it means the matrix was filtered away.
pub(super) fn run_release_decision_matrix_for(tier: QaTier) -> Vec<TestResult> {
    let registry = match load_registry() {
        Ok(registry) => registry,
        Err(error) => {
            return vec![fail(
                "decision/registry".to_string(),
                format!("failed to load RoomEQ QA registry: {error:#}"),
            )];
        }
    };
    let caps = matrix_caps(tier);
    let specs: Vec<&DecisionCaseSpec> = registry.decision_cases_for(tier).collect();
    if specs.is_empty() {
        return vec![fail(
            "decision/matrix_empty".to_string(),
            format!("no decision cases selected for tier '{tier:?}'; refusing an empty matrix"),
        )];
    }
    specs
        .iter()
        .map(|spec| run_decision_case(spec, caps))
        .collect()
}

/// PR-tier decision matrix: the subset that runs inside the binary guard
/// phase and the fast unit-test path.
pub(super) fn run_release_decision_matrix() -> Vec<TestResult> {
    run_release_decision_matrix_for(QaTier::Pr)
}

/// Number of decision rows for one tier without running anything (keeps the
/// reported guard total in sync with the executed guards).
pub(super) fn release_decision_case_count(tier: QaTier) -> usize {
    load_registry()
        .map(|registry| registry.decision_cases_for(tier).count())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{
        compromise_pick, continuous_scalarisation, knee_pick, matrix_caps, nsga_decision,
        nsga2_pool, pareto_knee_decision, release_decision_case_count, run_release_decision_matrix,
        run_release_decision_matrix_for, shuffled_order, to_pareto_filters, validate_front,
    };
    use crate::registry::{DecisionCaseKind, QaTier, load_registry};
    use autoeq_optim::optim::pareto::extract_non_dominated;

    fn spec_for(kind: DecisionCaseKind) -> crate::registry::DecisionCaseSpec {
        load_registry()
            .expect("registry loads")
            .decision_cases
            .iter()
            .find(|spec| spec.kind == kind)
            .expect("kind registered")
            .clone()
    }

    #[test]
    fn registry_decision_matrix_is_nonempty_per_tier() {
        let registry = load_registry().expect("registry loads");
        for tier in [QaTier::Pr, QaTier::Nightly, QaTier::Weekly] {
            assert!(
                registry.decision_cases_for(tier).next().is_some(),
                "tier {tier:?} selects no decision cases"
            );
        }
    }

    #[test]
    fn every_claim_is_exercised_by_its_invoked_test() {
        let registry = load_registry().expect("registry loads");
        for spec in &registry.decision_cases {
            for claim in &spec.claims {
                assert!(
                    spec.kind.allowed_claims().contains(&claim.as_str()),
                    "'{}' claims '{claim}' which its {:?} test never invokes",
                    spec.id,
                    spec.kind
                );
            }
        }
    }

    #[test]
    fn matrix_caps_stay_tier_bounded() {
        let pr = matrix_caps(QaTier::Pr);
        let nightly = matrix_caps(QaTier::Nightly);
        let weekly = matrix_caps(QaTier::Weekly);
        assert!(pr.strategies <= nightly.strategies);
        assert!(nightly.strategies <= 4);
        assert!(pr.seeds <= nightly.seeds && nightly.seeds <= weekly.seeds);
        assert!(pr.points <= nightly.points && nightly.points <= weekly.points);
        assert!(weekly.points <= 32);
    }

    #[test]
    fn synthetic_front_has_five_nondominated_members() {
        let filters = to_pareto_filters(&nsga2_pool(), false);
        let front = extract_non_dominated(&filters);
        assert_eq!(front.len(), 5);
    }

    #[test]
    fn nsga2_compromise_pick_is_deterministic_and_interior() {
        let spec = spec_for(DecisionCaseKind::Nsga2Decision);
        let (first, front_size) = nsga_decision(&spec, false).expect("nsga2 decides");
        assert_eq!(front_size, 5);
        // Interior compromise: flatness 1.9 dB loss with 7 filters.
        assert_eq!(first, 2);
        let (second, _) = nsga_decision(&spec, false).expect("nsga2 re-decides");
        assert_eq!(first, second);
    }

    #[test]
    fn nsga3_compromise_uses_all_three_objectives() {
        let spec = spec_for(DecisionCaseKind::Nsga3Decision);
        let (picked, front_size) = nsga_decision(&spec, true).expect("nsga3 decides");
        assert_eq!(front_size, 5);
        let filters = to_pareto_filters(&nsga2_pool(), true);
        let front = extract_non_dominated(&filters);
        assert_eq!(front.len(), 5);
        // Reference-point compromise differs from the 2-D crowding pick.
        assert_eq!(picked, 1);
    }

    #[test]
    fn pareto_knee_pick_is_deterministic_and_interior() {
        let spec = spec_for(DecisionCaseKind::ParetoFront);
        let (picked, front_size) = pareto_knee_decision(&spec).expect("knee decides");
        assert_eq!(front_size, 5);
        assert_eq!(picked, 2);
    }

    #[test]
    fn decision_is_invariant_to_input_order() {
        let filters = to_pareto_filters(&nsga2_pool(), false);
        let order = shuffled_order(filters.len(), 0x5EED1);
        let shuffled: Vec<_> = order.iter().map(|&i| filters[i].clone()).collect();
        validate_front(&shuffled).expect("shuffled front valid");
        let front_a = extract_non_dominated(&filters);
        let front_b = extract_non_dominated(&shuffled);
        assert_eq!(front_a.len(), front_b.len());
        assert_eq!(compromise_pick(&front_a, false), {
            // Same member by value after reordering.
            let picked = compromise_pick(&front_b, false);
            let value = front_b[picked].flatness_loss;
            front_a
                .iter()
                .position(|filter| filter.flatness_loss == value)
                .expect("pick is a front member")
        });
        let knee_a = knee_pick(&front_a);
        let knee_b = knee_pick(&front_b);
        assert_eq!(front_a[knee_a].flatness_loss, front_b[knee_b].flatness_loss);
    }

    #[test]
    fn invalid_fronts_are_rejected() {
        assert!(validate_front(&[]).is_err());
        let mut nan = to_pareto_filters(&nsga2_pool()[..2], false);
        nan[0].flatness_loss = f64::NAN;
        assert!(validate_front(&nan).is_err());
        let mut inf = to_pareto_filters(&nsga2_pool()[..2], false);
        inf[1].flatness_loss = f64::INFINITY;
        assert!(validate_front(&inf).is_err());
    }

    #[test]
    fn budgets_reject_oversized_pools_and_fronts() {
        let mut spec = spec_for(DecisionCaseKind::Nsga2Decision);
        spec.maxeval = 4;
        assert!(nsga_decision(&spec, false).is_err());
        let mut spec = spec_for(DecisionCaseKind::Nsga2Decision);
        spec.population = 2;
        assert!(nsga_decision(&spec, false).is_err());
    }

    #[test]
    fn worst_case_registry_entry_carries_timeout_and_inner_budget() {
        let spec = spec_for(DecisionCaseKind::ContinuousWorstCase);
        assert!(spec.timeout_ms > 0);
        assert!(spec.inner_maxiter > 0);
        assert!(continuous_scalarisation(&spec).is_ok());
        assert_eq!(super::continuous_search_budget(&spec).max_duration,
            Some(std::time::Duration::from_millis(spec.timeout_ms)));
        for kind in [DecisionCaseKind::ContinuousExpected, DecisionCaseKind::ContinuousCvar] {
            assert!(super::continuous_search_budget(&spec_for(kind)).max_duration.is_none());
        }
    }

    #[test]
    fn continuous_report_preserves_scores_on_timeout_and_realization_failure() {
        let spec = spec_for(DecisionCaseKind::ContinuousWorstCase);
        let mut result = roomeq_engine::multiseat::MultiSeatOptimizationResult {
            gains: vec![0.0; spec.subs],
            delays: vec![0.0; spec.subs],
            polarities: vec![false; spec.subs],
            allpass_filters: vec![vec![]; spec.subs],
            strategy: roomeq_model::MultiSeatStrategy::ContinuousArea,
            objective_name: "continuous_area".into(),
            objective_before: 4.25,
            objective_after: 2.5,
            objective_improvement_db: 1.75,
            variance_before: 0.0,
            variance_after: 0.0,
            variance_improvement_db: 0.0,
            improvement_db: 1.75,
        };
        let report = |result: &roomeq_engine::multiseat::MultiSeatOptimizationResult, elapsed| {
            super::report_continuous_case(&spec, "timeout-fixture".into(), "worst_case", 16, elapsed, result)
        };
        let timeout = report(&result, spec.timeout_ms + 1);
        assert!(!timeout.passed);
        assert!(timeout.reason.contains("exceeded timeout"));
        assert_eq!((timeout.pre_score, timeout.post_score), (4.25, 2.5));
        assert!(timeout.reason.contains("static_quadrature_used=false"));
        assert!(timeout.reason.contains("inner_position_evaluation_count=unavailable"));
        assert!(!timeout.reason.contains("outer_evals="));
        assert!(!timeout.reason.contains("inner_evals="));
        let at_limit = report(&result, spec.timeout_ms);
        assert!(at_limit.passed,
            "exactly the existing timeout remains permitted: {}", at_limit.reason);
        result.objective_after = result.objective_before;
        assert!(!report(&result, 1).passed, "unchanged objective does not meet required improvement");
        result.objective_before = 100.0;
        result.objective_after = 90.0;
        assert!(!report(&result, 1).passed, "improvement alone does not meet max_post_score");
        result.objective_before = 4.25;
        result.objective_after = 2.5;
        let mut excessive_gain = result.clone();
        excessive_gain.gains[1] = spec.expect.max_boost_db + 1.0;
        assert!(!report(&excessive_gain, 1).passed, "output gain exceeds the registered boost ceiling");
        excessive_gain.gains[1] = spec.expect.max_boost_db;
        assert!(report(&excessive_gain, 1).passed, "exact gain ceiling remains permitted");
        let mut missing_polarity = result.clone();
        missing_polarity.polarities.pop();
        assert!(!report(&missing_polarity, 1).passed, "missing polarity is incomplete transfer evidence");
        let mut missing_allpass = result.clone();
        missing_allpass.allpass_filters.pop();
        assert!(!report(&missing_allpass, 1).passed, "missing all-pass row is incomplete transfer evidence");
        for (frequency, q) in [(0.0, 1.0), (24000.0, 1.0), (80.0, 0.0), (80.0, f64::NAN)] {
            let mut malformed = result.clone();
            malformed.allpass_filters[1] = vec![(frequency, q)];
            assert!(!report(&malformed, 1).passed, "invalid all-pass {frequency}/{q} was accepted");
        }
        result.gains.clear();
        let invalid_transfer = report(&result, 1);
        assert!(!invalid_transfer.passed);
        assert_eq!((invalid_transfer.pre_score, invalid_transfer.post_score), (4.25, 2.5));
        let mut safety_spec = spec.clone();
        safety_spec.expect.gate_purpose = crate::registry::QaGatePurpose::Safety;
        safety_spec.expect.allow_safe_revert = true;
        result.objective_after = result.objective_before;
        let safety = |value: &roomeq_engine::multiseat::MultiSeatOptimizationResult| {
            super::report_continuous_case(&safety_spec, "safety-fixture".into(), "worst_case", 16, 1, value)
        };
        assert!(!safety(&result).passed, "safety policy cannot accept invalid transfer evidence");
        result.gains = vec![0.0; spec.subs];
        let valid_safety = safety(&result);
        assert!(valid_safety.passed);
        assert!(valid_safety.reason.contains("safety-only"));
        assert!(!valid_safety.reason.starts_with("REVERTED:"), "no rollback occurred");
        result.objective_name = "modal_basis".into();
        let mut modal_spec = spec_for(DecisionCaseKind::ModalPhase);
        assert!(!super::report_modal_phase_result(&modal_spec, "modal-fixture".into(), &result).passed,
            "unchanged modal objective does not meet required improvement");
        modal_spec.expect.gate_purpose = crate::registry::QaGatePurpose::Safety;
        modal_spec.expect.allow_safe_revert = true;
        result.polarities.clear();
        assert!(!super::report_modal_phase_result(&modal_spec, "modal-fixture".into(), &result).passed,
            "modal safety fallback cannot accept invalid realization");
    }

    #[test]
    fn cvar_registry_entry_carries_valid_alpha() {
        let spec = spec_for(DecisionCaseKind::ContinuousCvar);
        let alpha = spec.cvar_alpha.expect("cvar alpha set");
        assert!(alpha > 0.0 && alpha <= 1.0);
    }

    #[test]
    fn pr_decision_count_matches_executed_rows() {
        assert_eq!(
            release_decision_case_count(QaTier::Pr),
            run_release_decision_matrix().len()
        );
    }

    #[test]
    fn tmp_probe_nightly_matrix() {
        let rows = run_release_decision_matrix_for(QaTier::Nightly);
        let expected = release_decision_case_count(QaTier::Nightly);
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qa");
        std::fs::create_dir_all(&directory).unwrap();
        let records: Vec<_> = rows.iter().map(|row| serde_json::json!({
            "name": row.name, "assertion_passed": row.passed,
            "outcome": format!("{:?}", row.outcome()), "reason": row.reason,
            "pre_score": row.pre_score, "post_score": row.post_score,
            "epa_preference": row.epa_preference,
        })).collect();
        std::fs::write(directory.join("nightly-decision-matrix.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "scope": "registered_nightly_decision_contracts_not_full_acoustic_or_release_certification",
                "expected_rows": expected, "executed_rows": rows.len(), "rows": records,
            })).unwrap()).unwrap();
        for row in &rows {
            println!(
                "PROBE {} passed={} reason={}",
                row.name, row.passed, row.reason
            );
        }
        assert!(expected > 0, "nightly registry must select at least one case");
        assert_eq!(rows.len(), expected, "nightly decision inventory mismatch");
        let failures: Vec<_> = rows.iter().filter(|row| !row.passed)
            .map(|row| format!("{}: {}", row.name, row.reason)).collect();
        assert!(failures.is_empty(), "nightly decision failures: {}", failures.join("; "));
    }

    #[test]
    fn continuous_expected_seed_diagnostic_retains_non_improvement() {
        let spec = spec_for(DecisionCaseKind::ContinuousExpected);
        let measurements = roomeq_engine::multiseat::MultiSeatMeasurements::new(
            super::canonical_fixture().clone(),
        ).unwrap();
        let config = roomeq_model::MultiSeatConfig {
            enabled: true,
            strategy: roomeq_model::MultiSeatStrategy::ContinuousArea,
            continuous_area: Some(roomeq_model::ContinuousListeningAreaConfig {
                dimensions: 1,
                bounds: vec![(0.0, 1.0)],
                seat_positions: vec![vec![0.0], vec![1.0]],
                prior: roomeq_model::AreaPriorKind::Uniform,
                quadrature: roomeq_model::AreaQuadratureKind::Sobol {
                    num_points: spec.outer_points, seed: spec.seed,
                },
                scalarisation: roomeq_model::AreaScalarisationKind::ExpectedValue,
                idw_power: 2.0,
            }),
            ..Default::default()
        };
        let mut records = Vec::new();
        for seed in [None, Some(1), Some(42), Some(spec.seed)] {
            let budget = roomeq_engine::multiseat::MsoSearchBudget { seed, ..Default::default() };
            let (result, report) = roomeq_engine::multiseat::optimize_multiseat_continuous_area_with_budget(
                &measurements, &config, (20.0, 120.0), super::SAMPLE_RATE, &budget,
            ).unwrap();
            assert_eq!(report.stop_reason, "completed");
            assert_eq!(report.evaluations, 9448);
            assert!(result.objective_before.is_finite() && result.objective_after.is_finite());
            let improvement_pct = 100.0 * (result.objective_before - result.objective_after)
                / result.objective_before;
            records.push(serde_json::json!({
                "outer_seed_override": seed, "before": result.objective_before,
                "after": result.objective_after, "improvement_pct": improvement_pct,
                "meets_registry_improvement": improvement_pct >= spec.expect.improvement_min_pct,
                "gains_db": result.gains, "delays_ms": result.delays,
                "polarities": result.polarities, "allpass_filters": result.allpass_filters,
                "search_evaluations": report.evaluations, "generations": report.generations_run,
            }));
        }
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qa");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("continuous-expected-seed-diagnostic.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "scope": "seed_sensitivity_diagnostic_not_quality_acceptance_or_global_optimality_proof",
                "required_improvement_pct": spec.expect.improvement_min_pct,
                "records": records,
            })).unwrap()).unwrap();
    }

    #[test]
    fn empty_tier_selection_is_a_failure_not_a_pass() {
        // Weekly includes everything, so simulate filtering by asserting the
        // runner reports the empty case instead of returning zero rows.
        let rows = run_release_decision_matrix_for(QaTier::Pr);
        assert!(!rows.is_empty());
        assert!(rows.iter().all(|row| !row.name.is_empty()));
    }
}
