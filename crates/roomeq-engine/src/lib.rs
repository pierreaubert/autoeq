//! Experimental RoomEQ execution boundary.
//!
//! A run requires both an optimizer and an explicit DSP graph builder. The
//! production RoomEQ pipeline remains in the root `autoeq` crate until its
//! orchestration is migrated; this crate must not synthesize placeholder
//! success results.

#![forbid(unsafe_code)]

use autoeq_optim::{OptimizationProblem, OptimizationResult};
use roomeq_model::{ConfigValidationReport, DspGraph, RoomConfig, ValidationStage};

/// A narrow execution boundary. The engine owns orchestration; concrete
/// optimizers and exporters remain replaceable and independently testable.
pub trait RoomOptimizer {
    fn optimize(&self, problem: &OptimizationProblem) -> Result<OptimizationResult, String>;
}

/// Builds the realized DSP graph from a validated configuration and completed
/// optimization. Requiring this boundary prevents successful runs with an
/// empty placeholder graph.
pub trait RoomGraphBuilder {
    fn build_graph(
        &self,
        config: &RoomConfig,
        optimization: &OptimizationResult,
    ) -> Result<DspGraph, String>;
}

/// Input to one deterministic engine run.
pub struct EngineRequest<'a> {
    pub config: &'a RoomConfig,
    pub problem: OptimizationProblem,
}

#[derive(Debug, Clone)]
pub struct EngineResult {
    pub optimization: OptimizationResult,
    pub graph: DspGraph,
    pub validation: ConfigValidationReport,
}

/// Stateless coordinator useful both in production and unit tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct RoomEngine;

impl RoomEngine {
    pub fn validation_report(request: &EngineRequest<'_>) -> ConfigValidationReport {
        let mut report = request.config.validation_report();
        let mut errors = report.stage(ValidationStage::Structural).errors.clone();
        if let Err(error) = request.problem.validate() {
            errors.push(error);
        }
        report.record(ValidationStage::Structural, errors, Vec::new());
        report
    }

    pub fn validate(request: &EngineRequest<'_>) -> Result<(), String> {
        let report = Self::validation_report(request);
        report.errors().next().cloned().map_or(Ok(()), Err)
    }

    pub fn run<O: RoomOptimizer, G: RoomGraphBuilder>(
        &self,
        request: EngineRequest<'_>,
        optimizer: &O,
        graph_builder: &G,
    ) -> Result<EngineResult, String> {
        let mut validation = Self::validation_report(&request);
        if let Some(error) = validation.errors().next().cloned() {
            return Err(error);
        }
        let optimization = optimizer.optimize(&request.problem)?;
        let graph = graph_builder.build_graph(request.config, &optimization)?;
        match graph.validate() {
            Ok(()) => validation.record(ValidationStage::ExportTarget, Vec::new(), Vec::new()),
            Err(error) => {
                validation.record(
                    ValidationStage::ExportTarget,
                    vec![error.clone()],
                    Vec::new(),
                );
                return Err(error);
            }
        }
        Ok(EngineResult {
            optimization,
            graph,
            validation,
        })
    }
}

/// Adapter for simple closures in tests and small integrations.
impl<F> RoomOptimizer for F
where
    F: Fn(&OptimizationProblem) -> Result<OptimizationResult, String>,
{
    fn optimize(&self, problem: &OptimizationProblem) -> Result<OptimizationResult, String> {
        self(problem)
    }
}

/// Adapter for graph-building closures in tests and integrations.
impl<F> RoomGraphBuilder for F
where
    F: Fn(&RoomConfig, &OptimizationResult) -> Result<DspGraph, String>,
{
    fn build_graph(
        &self,
        config: &RoomConfig,
        optimization: &OptimizationResult,
    ) -> Result<DspGraph, String> {
        self(config, optimization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::RoomConfig;

    #[test]
    fn validates_before_calling_optimizer() {
        let mut config = RoomConfig::default();
        config.speakers.insert(
            "L".into(),
            roomeq_model::SpeakerConfig::Single(roomeq_model::MeasurementSource::Single(
                roomeq_model::MeasurementSingle {
                    measurement: roomeq_model::MeasurementRef::Path("left.csv".into()),
                    speaker_name: None,
                },
            )),
        );
        let request = EngineRequest {
            config: &config,
            problem: OptimizationProblem::new(vec![0.0], vec![1.0], |_: &[f64]| 0.0),
        };
        let optimizer = |_p: &OptimizationProblem| {
            Ok(OptimizationResult {
                parameters: vec![0.5],
                objective: 0.0,
                status: "ok".into(),
            })
        };
        let graph_builder = |_config: &RoomConfig, _optimization: &OptimizationResult| {
            let mut graph = DspGraph::new("1");
            graph.add_channel("L", Vec::new());
            Ok(graph)
        };
        let result = RoomEngine.run(request, &optimizer, &graph_builder).unwrap();
        assert_eq!(result.optimization.parameters, vec![0.5]);
        assert!(
            !result.graph.channels.is_empty(),
            "a successful engine run must not return an empty DSP graph"
        );
        assert_eq!(
            result
                .validation
                .stage(ValidationStage::ExportTarget)
                .status,
            roomeq_model::ValidationStageStatus::Passed
        );
        assert!(
            !result.validation.production_ready(),
            "the extracted engine must not claim resource/acoustic stages that it did not run"
        );
    }

    #[test]
    fn rejects_invalid_room_config_before_optimizer() {
        let config = RoomConfig::default();
        let request = EngineRequest {
            config: &config,
            problem: OptimizationProblem::new(vec![0.0], vec![1.0], |_: &[f64]| 0.0),
        };
        let optimizer = |_p: &OptimizationProblem| panic!("optimizer must not be called");
        let graph_builder = |_config: &RoomConfig, _optimization: &OptimizationResult| {
            panic!("graph builder must not be called")
        };
        let error = RoomEngine
            .run(request, &optimizer, &graph_builder)
            .unwrap_err();
        assert!(error.contains("at least one speaker"));
    }

    #[test]
    fn rejects_empty_graph_after_optimization() {
        let mut config = RoomConfig::default();
        config.speakers.insert(
            "L".into(),
            roomeq_model::SpeakerConfig::Single(roomeq_model::MeasurementSource::Single(
                roomeq_model::MeasurementSingle {
                    measurement: roomeq_model::MeasurementRef::Path("left.csv".into()),
                    speaker_name: None,
                },
            )),
        );
        let request = EngineRequest {
            config: &config,
            problem: OptimizationProblem::new(vec![0.0], vec![1.0], |_: &[f64]| 0.0),
        };
        let optimizer = |_p: &OptimizationProblem| {
            Ok(OptimizationResult {
                parameters: vec![0.5],
                objective: 0.0,
                status: "ok".into(),
            })
        };
        let graph_builder =
            |_config: &RoomConfig, _optimization: &OptimizationResult| Ok(DspGraph::new("1"));

        let error = RoomEngine
            .run(request, &optimizer, &graph_builder)
            .expect_err("empty graph must be rejected");
        assert!(error.contains("at least one channel"));
    }
}
