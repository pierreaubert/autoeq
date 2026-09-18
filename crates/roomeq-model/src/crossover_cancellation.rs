//! Evidence and policy for coherent main/sub crossover cancellation.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const DEFAULT_MAX_CROSSOVER_CANCELLATION_DB: f64 = 3.0;
pub const CROSSOVER_CANCELLATION_TOLERANCE_DB: f64 = 0.05;

/// Frozen measured cancellation spectrum of the configured, unoptimized route.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CrossoverCancellationBaseline {
    pub crossover_hz: f64,
    pub frequencies_hz: Vec<f64>,
    pub cancellation_db: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CrossoverCancellationContext {
    pub limit_db: f64,
    pub sources: BTreeMap<String, CrossoverCancellationBaseline>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CrossoverCancellationEvidence {
    pub source_channel: String,
    pub baseline_db: Option<f64>,
    pub baseline_worst_frequency_hz: Option<f64>,
    pub final_db: f64,
    pub final_worst_frequency_hz: f64,
    pub comparison_band_hz: [f64; 2],
    pub limit_db: f64,
    pub improvement_db: Option<f64>,
    pub accepted: bool,
    pub reason: String,
}

/// Above-limit residuals must improve by more than interpolation tolerance.
pub fn crossover_cancellation_accepted(candidate: f64, baseline: Option<f64>, limit: f64) -> bool {
    candidate.is_finite()
        && candidate >= 0.0
        && limit.is_finite()
        && limit >= 0.0
        && (candidate <= limit + CROSSOVER_CANCELLATION_TOLERANCE_DB
            || baseline.is_some_and(|before| {
                before.is_finite() && before - candidate > CROSSOVER_CANCELLATION_TOLERANCE_DB
            }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn acceptance_requires_improvement_above_configured_limit() {
        assert!(crossover_cancellation_accepted(4.0, Some(10.0), 3.0));
        for after in [10.0, 11.0, 9.99] {
            assert!(!crossover_cancellation_accepted(after, Some(10.0), 3.0));
        }
        assert!(!crossover_cancellation_accepted(4.0, None, 3.0));
        assert!(crossover_cancellation_accepted(4.0, None, 5.0));
        assert!(crossover_cancellation_accepted(3.049, None, 3.0));
        assert!(!crossover_cancellation_accepted(3.051, None, 3.0));
        assert!(!crossover_cancellation_accepted(f64::NAN, Some(10.0), 3.0));
    }

    #[test]
    fn cancellation_config_defaults_round_trips_and_validates() {
        let mut json = serde_json::to_value(crate::OptimizerConfig::default()).unwrap();
        json.as_object_mut()
            .unwrap()
            .remove("max_crossover_cancellation_db");
        let legacy: crate::OptimizerConfig = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(legacy.max_crossover_cancellation_db, 3.0);
        json["max_crossover_cancellation_db"] = serde_json::json!(6.0);
        let custom: crate::OptimizerConfig = serde_json::from_value(json).unwrap();
        assert_eq!(custom.max_crossover_cancellation_db, 6.0);
        assert!(
            !serde_json::to_value(custom)
                .unwrap()
                .as_object()
                .unwrap()
                .contains_key("crossover_cancellation_baseline")
        );
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            let mut config = crate::RoomConfig::default();
            config.optimizer.max_crossover_cancellation_db = invalid;
            assert!(
                config
                    .validate_structure()
                    .unwrap_err()
                    .contains("max_crossover_cancellation_db")
            );
        }
    }
}
