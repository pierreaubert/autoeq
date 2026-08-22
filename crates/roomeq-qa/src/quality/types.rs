use super::metric_scorecard::MetricScorecard;

pub(super) struct TestResult {
    pub(super) label: String,
    pub(super) pre_score: f64,
    pub(super) scorecard: MetricScorecard,
    pub(super) pass: bool,
    pub(super) reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QaOutcome {
    Passed,
    Reverted,
    Failed,
}

impl TestResult {
    pub(super) fn outcome(&self) -> QaOutcome {
        if self.scorecard.correction_reverted || self.reason.to_ascii_lowercase().contains("revert")
        {
            QaOutcome::Reverted
        } else if self.pass {
            QaOutcome::Passed
        } else {
            QaOutcome::Failed
        }
    }
}
