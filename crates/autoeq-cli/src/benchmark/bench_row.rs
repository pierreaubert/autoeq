#[derive(Debug)]
pub(super) struct BenchRow {
    pub(super) speaker: String,
    pub(super) flat_cea2034_lw: Option<f64>,
    pub(super) flat_eir: Option<f64>,
    pub(super) score_cea2034_mh_rga: Option<f64>,
    pub(super) score_cea2034_mh_pso: Option<f64>,
    pub(super) score_cea2034_autoeq_de: Option<f64>,
    pub(super) score_cea2034_autoeq_cmaes: Option<f64>,
    pub(super) metadata_pref: Option<f64>,
}
