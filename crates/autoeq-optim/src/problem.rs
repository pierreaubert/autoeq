use std::sync::Arc;

/// Scalar objective evaluated by an optimizer backend.
pub trait ObjectiveEvaluator: Send + Sync {
    fn evaluate(&self, parameters: &[f64]) -> f64;
}

impl<F> ObjectiveEvaluator for F
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    fn evaluate(&self, parameters: &[f64]) -> f64 {
        self(parameters)
    }
}

/// Backend-independent optimization problem.
#[derive(Clone)]
pub struct OptimizationProblem {
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub objective: Arc<dyn ObjectiveEvaluator>,
    pub integrality: Option<Vec<bool>>,
}

impl OptimizationProblem {
    pub fn new(
        lower_bounds: Vec<f64>,
        upper_bounds: Vec<f64>,
        objective: impl ObjectiveEvaluator + 'static,
    ) -> Self {
        Self {
            lower_bounds,
            upper_bounds,
            objective: Arc::new(objective),
            integrality: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.lower_bounds.len() != self.upper_bounds.len() {
            return Err("lower and upper bounds must have equal lengths".to_string());
        }
        if self
            .lower_bounds
            .iter()
            .zip(&self.upper_bounds)
            .any(|(lower, upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        {
            return Err("optimization bounds must be finite and ordered".to_string());
        }
        if let Some(integrality) = &self.integrality
            && integrality.len() != self.lower_bounds.len()
        {
            return Err("integrality length must match bounds length".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationResult {
    pub parameters: Vec<f64>,
    pub objective: f64,
    pub status: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_neutral_problem_shape() {
        let problem =
            OptimizationProblem::new(vec![0.0, 1.0], vec![1.0, 2.0], |x: &[f64]| x.iter().sum());
        assert!(problem.validate().is_ok());
        assert_eq!(problem.objective.evaluate(&[2.0, 3.0]), 5.0);
    }

    #[test]
    fn rejects_mismatched_bounds_and_integrality() {
        let mut problem = OptimizationProblem::new(vec![0.0], vec![1.0, 2.0], |_: &[f64]| 0.0);
        assert!(problem.validate().is_err());
        problem.upper_bounds = vec![1.0];
        problem.integrality = Some(vec![true, false]);
        assert!(problem.validate().is_err());
    }
}
