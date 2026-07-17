#[derive(Debug, Clone)]
pub struct ResolvedGroupCrossover {
    pub crossover_type: String,
    pub frequency_hz: Option<f64>,
    pub configured_hz: Option<f64>,
    pub frequency_range: Option<(f64, f64)>,
    pub missing_config_key: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ResolvedGroupRouteSettings {
    pub main_delay_ms: f64,
    pub bass_route_delay_ms: f64,
    pub polarity_inverted: bool,
    pub trim_db: f64,
}
