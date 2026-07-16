use std::path::PathBuf;

/// Frequency grid used to normalize workflow inputs and build report curves.
///
/// Omitted bounds follow [`crate::OptimParams::min_freq`] and
/// [`crate::OptimParams::max_freq`]. This keeps narrow-band RoomEQ workflows
/// from spending most of their report points outside the optimization band.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualizationGridConfig {
    /// Number of logarithmically spaced frequency points.
    pub points: usize,
    /// Optional lower grid bound in Hz.
    pub min_freq: Option<f64>,
    /// Optional upper grid bound in Hz.
    pub max_freq: Option<f64>,
}

impl Default for VisualizationGridConfig {
    fn default() -> Self {
        Self {
            points: 200,
            min_freq: None,
            max_freq: None,
        }
    }
}

impl VisualizationGridConfig {
    /// Resolve and validate the logarithmic frequency grid for an optimizer.
    pub fn create_frequency_grid(
        &self,
        params: &crate::OptimParams,
    ) -> crate::Result<ndarray::Array1<f64>> {
        let min_freq = self.min_freq.unwrap_or(params.min_freq);
        let max_freq = self.max_freq.unwrap_or(params.max_freq);
        if self.points < 2 {
            return Err(crate::AutoeqError::InvalidConfiguration {
                message: "visualization grid requires at least 2 points".to_string(),
            });
        }
        if !min_freq.is_finite() || !max_freq.is_finite() || min_freq <= 0.0 {
            return Err(crate::AutoeqError::InvalidConfiguration {
                message: "visualization grid bounds must be finite and positive".to_string(),
            });
        }
        if min_freq >= max_freq {
            return Err(crate::AutoeqError::InvalidConfiguration {
                message: format!(
                    "visualization grid minimum ({min_freq} Hz) must be below maximum ({max_freq} Hz)"
                ),
            });
        }
        let nyquist = params.sample_rate / 2.0;
        if !params.sample_rate.is_finite() || params.sample_rate <= 0.0 || max_freq >= nyquist {
            return Err(crate::AutoeqError::InvalidConfiguration {
                message: format!(
                    "visualization grid maximum ({max_freq} Hz) must be below Nyquist ({nyquist} Hz)"
                ),
            });
        }

        Ok(crate::read::create_log_frequency_grid(
            self.points,
            min_freq,
            max_freq,
        ))
    }
}

/// Domain configuration for selecting an input measurement.
///
/// This is the CLI-independent counterpart of the API/file selection fields
/// from [`crate::cli::Args`].
#[derive(Debug, Clone)]
pub struct InputConfig {
    pub speaker: Option<String>,
    pub version: Option<String>,
    pub measurement: Option<String>,
    pub curve_name: String,
    pub curve_path: Option<PathBuf>,
}

/// Domain configuration for selecting a target curve.
#[derive(Debug, Clone)]
pub struct TargetConfig {
    pub target_path: Option<PathBuf>,
    pub curve_name: String,
}

impl From<&crate::cli::Args> for InputConfig {
    fn from(args: &crate::cli::Args) -> Self {
        Self {
            speaker: args.speaker.clone(),
            version: args.version.clone(),
            measurement: args.measurement.clone(),
            curve_name: args.curve_name.clone(),
            curve_path: args.curve.clone(),
        }
    }
}

impl From<&crate::cli::Args> for TargetConfig {
    fn from(args: &crate::cli::Args) -> Self {
        Self {
            target_path: args.target.clone(),
            curve_name: args.curve_name.clone(),
        }
    }
}
