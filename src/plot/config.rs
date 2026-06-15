//! Plotting configuration decoupled from CLI arguments.

/// Domain configuration for plot generation.
#[derive(Debug, Clone)]
pub struct PlotConfig {
    pub speaker_name: Option<String>,
    pub num_filters: usize,
    pub sample_rate: f64,
    pub peq_model: crate::PeqModel,
    pub min_freq: f64,
    pub max_freq: f64,
}

impl From<&crate::cli::Args> for PlotConfig {
    fn from(args: &crate::cli::Args) -> Self {
        Self {
            speaker_name: args.speaker.clone(),
            num_filters: args.num_filters,
            sample_rate: args.sample_rate,
            peq_model: args.effective_peq_model(),
            min_freq: args.min_freq,
            max_freq: args.max_freq,
        }
    }
}
