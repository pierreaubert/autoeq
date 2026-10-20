#[derive(Debug, Clone)]
pub(in super::super) struct GeneratedFir {
    pub coeffs: Vec<f64>,
    pub filename: String,
    pub mixed_phase_report: Option<roomeq_engine::mixed_phase::MixedPhaseCorrectionReport>,
}

impl GeneratedFir {
    pub(in super::super) fn convolution_plugin(
        &self,
        optimizer: &roomeq_model::OptimizerConfig,
        sample_rate: f64,
    ) -> roomeq_model::PluginConfigWrapper {
        if let Some(report) = &self.mixed_phase_report {
            return roomeq_engine::output::create_mixed_phase_convolution_plugin(
                &self.filename,
                report,
            );
        }
        let mut plugin = roomeq_engine::output::create_convolution_plugin(&self.filename);
        let fir = optimizer.fir.clone().unwrap_or_default();
        if fir.pre_ringing.is_none() {
            let samples = match fir.phase.to_ascii_lowercase().as_str() {
                "linear" | "kirkeby" => Some(fir.taps / 2),
                "minimum" => Some(0),
                _ => None,
            };
            if let Some(samples) = samples {
                plugin.parameters["correction_design_delay_ms"] =
                    serde_json::json!(samples as f64 * 1000.0 / sample_rate);
            }
        }
        plugin
    }
}
