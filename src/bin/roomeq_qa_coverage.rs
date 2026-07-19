//! Thin compatibility launcher for crate-owned RoomEQ coverage QA.

use autoeq::roomeq::{CallbackAction, RoomOptimizationProgress};
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::RoomConfig;
use std::path::Path;
use std::sync::Arc;

struct LegacyOptimizer;

impl roomeq_qa::RoomOptimizer for LegacyOptimizer {
    fn optimize_room(
        &self,
        config: &RoomConfig,
        sample_rate: f64,
        output_dir: Option<&Path>,
    ) -> anyhow::Result<RoomOptimizationResult> {
        let callback = Box::new(|_: &RoomOptimizationProgress| CallbackAction::Continue);
        autoeq::roomeq::optimize_room(config, sample_rate, Some(callback), output_dir)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    }
}

fn main() -> anyhow::Result<()> {
    if roomeq_qa::coverage::run(Arc::new(LegacyOptimizer))? {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::LegacyOptimizer;
    use roomeq_qa::coverage::{DEFAULT_MAXEVAL, ProcessingMethod, run_regression_case};

    #[test]
    fn grouped_topology_modes_accept_final_realization() {
        for method in [
            ProcessingMethod::Iir,
            ProcessingMethod::Fir,
            ProcessingMethod::Mixed,
            ProcessingMethod::MixedPhase,
        ] {
            if let Err(error) = run_regression_case(
                &LegacyOptimizer,
                "small_stereo_2_2_group",
                method,
                DEFAULT_MAXEVAL,
            ) {
                panic!("{} failed: {error}", method.name());
            }
        }
    }

    #[test]
    fn mso_realization_counts_as_coverage_when_flatness_is_unchanged() {
        if let Err(error) = run_regression_case(
            &LegacyOptimizer,
            "small_stereo_2_2_mso",
            ProcessingMethod::Iir,
            DEFAULT_MAXEVAL,
        ) {
            panic!("MSO coverage failed: {error}");
        }
    }
}
