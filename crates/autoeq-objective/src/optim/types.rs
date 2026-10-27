use super::objective_data::ObjectiveData;

/// Data for multi-objective optimization across multiple measurements
#[derive(Debug, Clone)]
pub struct MultiObjectiveData {
    /// One ObjectiveData per measurement curve. For
    /// `MultiMeasurementStrategy::MinimaxUncertainty`, this holds B
    /// bootstrap-resampled objectives generated once at setup time.
    pub objectives: Vec<ObjectiveData>,
    /// Strategy for combining per-measurement losses
    pub strategy: crate::roomeq::MultiMeasurementStrategy,
    /// Normalized weights (len == objectives.len()), used by WeightedSum
    pub weights: Vec<f64>,
    /// Lambda for VariancePenalized strategy
    pub variance_lambda: f64,
    /// CVaR tail fraction in (0, 1], used by `MinimaxUncertainty` when its
    /// scalarisation is `Cvar`. `None` means pure worst-case (max).
    pub uncertainty_cvar_alpha: Option<f64>,
}
