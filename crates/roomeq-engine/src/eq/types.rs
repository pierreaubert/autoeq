use crate::PeqModel;

/// Prepared data for single-channel EQ optimization.
/// Contains all pre-processed data that is independent of filter count.
pub(super) struct PreparedSingleChannelEq {
    pub(super) objective_data: autoeq_optim::optim::ObjectiveData,
    pub(super) args_template: autoeq_optim::OptimParams,
    pub(super) peq_model: PeqModel,
    pub(super) sample_rate: f64,
}
