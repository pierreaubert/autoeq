//! Centralised constraint installation.
//!
//! The three core PEQ inequality constraints (ceiling, min-gain, spacing) are
//! built here in [`build_constraint_set`]. Backends call
//! [`install_constraints`] which decides — based on
//! [`ConstraintCapabilities`] — whether to register them natively or fold
//! them into the objective via penalty weights.
//!
//! Crossover monotonicity for multi-driver `DriversFlat` is exposed
//! separately as [`build_crossover_monotonicity_constraint`]. The pure-Rust
//! COBYLA and ISRES backends register it; the AutoEQ DE path does not.

use super::backend::{ConstraintCapabilities, ConstraintInstallation, NativeConstraint};
use super::{ObjectiveData, PenaltyMode};
use crate::constraints::{
    CeilingConstraintData, CrossoverMonotonicityConstraintData, MinGainConstraintData,
    SpacingConstraintData, viol_ceiling_from_spl, viol_min_gain_from_xs, viol_spacing_from_xs,
};
use crate::x2peq::x2spl;

/// Install constraints into `obj` for a backend with the given capabilities.
///
/// Returns `Native(constraints)` when the backend supports nonlinear
/// inequalities natively (penalties are disabled in `obj`); returns
/// `Penalty` when penalty weights are written into `obj` so that
/// `compute_fitness_penalties_ref` folds the violations into the objective.
pub fn install_constraints(
    caps: ConstraintCapabilities,
    obj: &mut ObjectiveData,
) -> ConstraintInstallation {
    if caps.nonlinear_ineq {
        obj.configure_penalties(PenaltyMode::Disabled);
        ConstraintInstallation::Native(build_constraint_set(obj))
    } else {
        obj.configure_penalties(caps.fallback_penalty_mode);
        ConstraintInstallation::Penalty
    }
}

/// Build the three core PEQ constraints, each guarded by the same enable
/// conditions retained from the pre-registry backend implementations.
pub fn build_constraint_set(obj: &ObjectiveData) -> Vec<NativeConstraint> {
    let mut out: Vec<NativeConstraint> = Vec::new();

    if obj.max_db > 0.0 {
        out.push(build_ceiling_constraint(obj));
    }
    if obj.min_db > 0.0 {
        out.push(build_min_gain_constraint(obj));
    }
    if obj.min_spacing_oct > 0.0 {
        out.push(build_spacing_constraint(obj));
    }

    out
}

/// Crossover monotonicity for `DriversFlat` multi-driver layouts.
///
/// Returns `None` unless `loss_type == DriversFlat` with at least one
/// crossover (i.e. ≥ 2 drivers). COBYLA and ISRES opt in explicitly rather
/// than every backend picking it up by accident.
pub fn build_crossover_monotonicity_constraint(obj: &ObjectiveData) -> Option<NativeConstraint> {
    use crate::LossType;
    if obj.loss_type != LossType::DriversFlat {
        return None;
    }
    let drivers = obj.drivers_data.as_ref()?;
    if drivers.drivers.len() < 2 {
        return None;
    }

    // 0.15 log10 ≈ 40% frequency separation. This is the established
    // behavior-preserving value retained by the pure-Rust constrained
    // backends, not a new tuning choice.
    let data = CrossoverMonotonicityConstraintData {
        n_drivers: drivers.drivers.len(),
        min_log_separation: 0.15,
    };
    Some(NativeConstraint {
        label: "crossover_monotonicity",
        fun: Box::new(move |x| {
            let n_xovers = data.n_drivers - 1;
            if n_xovers <= 1 {
                return 0.0;
            }
            let xover_start = 2 * data.n_drivers;
            let mut max_violation = f64::NEG_INFINITY;
            for i in 0..(n_xovers - 1) {
                let log_xover_i = x[xover_start + i];
                let log_xover_i_plus_1 = x[xover_start + i + 1];
                let violation = log_xover_i + data.min_log_separation - log_xover_i_plus_1;
                if violation > max_violation {
                    max_violation = violation;
                }
            }
            // Match the original `constraint_crossover_monotonicity`
            // (constraints/crossover_monotonicity.rs:57): it returns the
            // signed maximum violation, which can be negative when the
            // constraint is comfortably satisfied. Native constrained
            // backends treat fc(x) <= 0 as feasible, so do not clamp to 0.
            max_violation
        }),
        tol: 1e-6,
    })
}

fn build_ceiling_constraint(obj: &ObjectiveData) -> NativeConstraint {
    let data = CeilingConstraintData {
        freqs: obj.freqs.clone(),
        srate: obj.srate,
        max_db: obj.max_db,
        peq_model: obj.peq_model,
    };
    NativeConstraint {
        label: "ceiling",
        fun: Box::new(move |x| {
            let peq_spl = x2spl(&data.freqs, x, data.srate, data.peq_model);
            viol_ceiling_from_spl(&peq_spl, data.max_db, data.peq_model)
        }),
        tol: 1e-6,
    }
}

fn build_min_gain_constraint(obj: &ObjectiveData) -> NativeConstraint {
    let data = MinGainConstraintData {
        min_db: obj.min_db,
        peq_model: obj.peq_model,
    };
    NativeConstraint {
        label: "min_gain",
        fun: Box::new(move |x| viol_min_gain_from_xs(x, data.peq_model, data.min_db)),
        tol: 1e-6,
    }
}

fn build_spacing_constraint(obj: &ObjectiveData) -> NativeConstraint {
    let data = SpacingConstraintData {
        min_spacing_oct: obj.min_spacing_oct,
        peq_model: obj.peq_model,
    };
    NativeConstraint {
        label: "spacing",
        fun: Box::new(move |x| viol_spacing_from_xs(x, data.peq_model, data.min_spacing_oct)),
        tol: 1e-6,
    }
}
