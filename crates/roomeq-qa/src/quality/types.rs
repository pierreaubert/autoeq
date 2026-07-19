use super::metric_scorecard::MetricScorecard;

pub(super) struct TestResult {
    pub(super) label: String,
    pub(super) pre_score: f64,
    pub(super) scorecard: MetricScorecard,
    pub(super) pass: bool,
    pub(super) reason: String,
}
