#[derive(Debug, Clone)]
pub(in super::super) struct GeneratedFir {
    pub coeffs: Vec<f64>,
    pub filename: String,
    pub mixed_phase_report: Option<roomeq_engine::mixed_phase::MixedPhaseCorrectionReport>,
}
