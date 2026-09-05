use super::default::default_multiseat_max_allpass_per_sub;
use super::default::default_multiseat_max_points_per_axis;
use super::default::default_multiseat_max_quadrature_points;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Stage-specific search contract for multi-seat optimization.
///
/// `MultiSeatConfig` exposes the polarity/all-pass toggles and the
/// continuous-area prior but historically carried no outer evaluation/time
/// budget or seed: the engine used fixed internal values. This block lets
/// callers pin those down per run. Every field is optional or capped with a
/// default that preserves the historical behavior:
///
/// - `seed`, `evaluation_budget`, `time_budget_ms` are `None` by default,
///   meaning "engine legacy behavior" (fixed internal seed/budgets).
///   Consumers (`roomeq-workflow`, `roomeq-engine`) must treat an explicit
///   value as authoritative and `None` as "keep doing what you do today".
/// - The `max_*` caps bound resource use. Validation rejects quadrature and
///   all-pass configurations above the effective caps; the defaults are
///   generous so previously valid configs keep passing.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct MultiSeatSearchConfig {
    /// PRNG seed for the multi-seat search (quadrature sampling, stochastic
    /// stages). `None` = engine legacy fixed seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Outer evaluation budget (objective evaluations). `None` = engine
    /// legacy budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_budget: Option<usize>,
    /// Wall-clock budget in milliseconds. `None` = no model-level limit
    /// (engine legacy behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_budget_ms: Option<u64>,
    /// Maximum Sobol / Latin-hypercube quadrature points, and maximum total
    /// Gauss-Legendre points (`points_per_axis^dimensions`).
    #[serde(default = "default_multiseat_max_quadrature_points")]
    pub max_quadrature_points: usize,
    /// Maximum Gauss-Legendre nodes per axis.
    #[serde(default = "default_multiseat_max_points_per_axis")]
    pub max_points_per_axis: usize,
    /// Maximum per-sub all-pass filters (`allpass_filters_per_sub` must not
    /// exceed this).
    #[serde(default = "default_multiseat_max_allpass_per_sub")]
    pub max_allpass_per_sub: usize,
}

impl Default for MultiSeatSearchConfig {
    fn default() -> Self {
        Self {
            seed: None,
            evaluation_budget: None,
            time_budget_ms: None,
            max_quadrature_points: default_multiseat_max_quadrature_points(),
            max_points_per_axis: default_multiseat_max_points_per_axis(),
            max_allpass_per_sub: default_multiseat_max_allpass_per_sub(),
        }
    }
}

impl MultiSeatSearchConfig {
    /// Structural validation. Returns one error string per violation; empty
    /// means the contract is usable. Zero budgets and zero caps are rejected
    /// (use `None` for "no explicit budget").
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.evaluation_budget == Some(0) {
            errors.push(
                "multi_seat.search.evaluation_budget must be > 0 when set".to_string(),
            );
        }
        if self.time_budget_ms == Some(0) {
            errors.push("multi_seat.search.time_budget_ms must be > 0 when set".to_string());
        }
        if self.max_quadrature_points == 0 {
            errors.push(
                "multi_seat.search.max_quadrature_points must be > 0".to_string(),
            );
        }
        if self.max_points_per_axis == 0 {
            errors.push("multi_seat.search.max_points_per_axis must be > 0".to_string());
        }
        return errors;
    }
}
