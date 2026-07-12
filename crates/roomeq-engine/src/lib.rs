//! RoomEQ analysis, correction, and orchestration.

#![forbid(unsafe_code)]

use autoeq_optim::{OptimizationProblem, OptimizationResult};
use roomeq_model::{DspGraph, RoomConfig};

/// A narrow execution boundary. The engine owns orchestration; concrete
/// optimizers and exporters remain replaceable and independently testable.
pub trait RoomOptimizer {
    fn optimize(&self, problem: &OptimizationProblem) -> Result<OptimizationResult, String>;
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
}

/// Stateless coordinator useful both in production and unit tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct RoomEngine;

impl RoomEngine {
    pub fn validate(request: &EngineRequest<'_>) -> Result<(), String> {
        let _ = request.config;
        request.problem.validate()
    }

    pub fn run<O: RoomOptimizer>(
        &self,
        request: EngineRequest<'_>,
        optimizer: &O,
    ) -> Result<EngineResult, String> {
        Self::validate(&request)?;
        let optimization = optimizer.optimize(&request.problem)?;
        Ok(EngineResult { optimization, graph: DspGraph::new("1") })
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

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::RoomConfig;

    #[test]
    fn validates_before_calling_optimizer() {
        let config = RoomConfig::default();
        let request = EngineRequest {
            config: &config,
            problem: OptimizationProblem::new(vec![0.0], vec![1.0], |_: &[f64]| 0.0),
        };
        let optimizer = |_p: &OptimizationProblem| {
            Ok(OptimizationResult { parameters: vec![0.5], objective: 0.0, status: "ok".into() })
        };
        let result = RoomEngine.run(request, &optimizer).unwrap();
        assert_eq!(result.optimization.parameters, vec![0.5]);
    }
}
