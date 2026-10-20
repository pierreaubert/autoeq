use anyhow::Result;
use std::path::Path;

pub(super) fn delivered_quality_improved(
    qa_passed: bool,
    accepted: bool,
    reverted: bool,
    improvements: impl Iterator<Item = f64>,
) -> bool {
    let mut improvements = improvements.peekable();
    qa_passed
        && accepted
        && !reverted
        && improvements.peek().is_some()
        && improvements.all(|value| value.is_finite() && value > 0.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum QaOutcome {
    Passed,
    Reverted,
    Failed,
}

impl QaOutcome {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Passed => "PASS",
            Self::Reverted => "REVERTED",
            Self::Failed => "FAIL",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[allow(dead_code)]
pub(super) struct TestResult {
    pub(super) name: String,
    pub(super) scenario: String,
    pub(super) solver: String,
    pub(super) method: String,
    pub(super) pre_score: f64,
    pub(super) post_score: f64,
    pub(super) epa_preference: Option<f64>,
    pub(super) passed: bool,
    pub(super) outcome: QaOutcome,
    pub(super) error: Option<String>,
    pub(super) duration_ms: u64,
    /// Final runtime verdict, independent of whether the QA safety case passed.
    pub(super) correction_decision: Option<roomeq_model::CorrectionDecision>,
    /// True only with accepted, unreverted improvement at every delivered seat/input.
    pub(super) quality_improved: bool,
    /// Retain applied, skipped, reverted and unassessed stage evidence separately.
    pub(super) stage_outcomes: Vec<roomeq_model::StageOutcome>,
    /// Delivered input/seat metrics used to establish the improvement claim.
    pub(super) final_seats: Vec<roomeq_model::FinalSeatEvaluation>,
}

impl TestResult {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn success(
        name: &str,
        scenario: &str,
        solver: &str,
        method: &str,
        pre: f64,
        post: f64,
        epa_preference: Option<f64>,
        validation_failures: Vec<String>,
        reverted: bool,
        dur: u64,
    ) -> Self {
        let passed = validation_failures.is_empty();
        let outcome = if !passed {
            QaOutcome::Failed
        } else if reverted {
            QaOutcome::Reverted
        } else {
            QaOutcome::Passed
        };
        let error = if validation_failures.is_empty() {
            None
        } else {
            Some(validation_failures.join("; "))
        };
        Self {
            name: name.to_string(),
            scenario: scenario.to_string(),
            solver: solver.to_string(),
            method: method.to_string(),
            pre_score: pre,
            post_score: post,
            epa_preference,
            passed,
            outcome,
            error,
            duration_ms: dur,
            correction_decision: None,
            quality_improved: false,
            stage_outcomes: Vec::new(),
            final_seats: Vec::new(),
        }
    }

    pub(super) fn failure(
        name: &str,
        scenario: &str,
        solver: &str,
        method: &str,
        err: String,
        dur: u64,
    ) -> Self {
        Self {
            name: name.to_string(),
            scenario: scenario.to_string(),
            solver: solver.to_string(),
            method: method.to_string(),
            pre_score: 0.0,
            post_score: 0.0,
            epa_preference: None,
            passed: false,
            outcome: QaOutcome::Failed,
            error: Some(err),
            duration_ms: dur,
            correction_decision: None,
            quality_improved: false,
            stage_outcomes: Vec::new(),
            final_seats: Vec::new(),
        }
    }

    pub(super) fn junit_xml(&self) -> String {
        let mut xml = format!(
            r#"    <testcase name="{}" classname="roomeq.{}" time="{}">"#,
            self.name,
            self.scenario,
            self.duration_ms as f64 / 1000.0
        );

        if self.outcome == QaOutcome::Reverted {
            xml.push_str("\n      <skipped message=\"runtime correction reverted\"/>");
        } else if !self.passed {
            let msg = if let Some(ref err) = self.error {
                err.clone()
            } else {
                format!(
                    "post_score {:.4} >= pre_score {:.4}",
                    self.post_score, self.pre_score
                )
            };
            let escaped = msg
                .replace('"', "&quot;")
                .replace('<', "&lt;")
                .replace('>', "&gt;");
            xml.push_str(&format!(
                "\n      <failure message=\"{}\" type=\"AssertionError\"/>",
                escaped
            ));
        }

        xml.push_str("\n    </testcase>");
        xml
    }
}

pub(super) fn write_junit_xml(results: &[TestResult], output: &Path) -> Result<()> {
    let mut xml = String::from(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    xml.push_str("\n<testsuite name=\"roomeq-qa-full\" ");
    xml.push_str(&format!("tests=\"{}\" ", results.len()));
    xml.push_str(&format!(
        "failures=\"{}\" ",
        results.iter().filter(|r| !r.passed).count()
    ));
    xml.push_str(&format!(
        "errors=\"{}\" ",
        results.iter().filter(|r| r.error.is_some()).count()
    ));
    xml.push_str(&format!(
        "time=\"{}\" ",
        results.iter().map(|r| r.duration_ms).sum::<u64>() as f64 / 1000.0
    ));
    xml.push_str(">\n");

    for result in results {
        xml.push_str(&result.junit_xml());
        xml.push('\n');
    }

    xml.push_str("</testsuite>\n");

    std::fs::write(output, xml)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivered_quality_requires_every_seat_and_cannot_be_established_by_safe_revert() {
        assert!(delivered_quality_improved(
            true,
            true,
            false,
            [0.1, 2.0].into_iter()
        ));
        for values in [
            vec![],
            vec![1.0, 0.0],
            vec![1.0, -0.1],
            vec![f64::NAN],
            vec![f64::INFINITY],
        ] {
            assert!(!delivered_quality_improved(
                true,
                true,
                false,
                values.into_iter()
            ));
        }
        for (passed, accepted, reverted) in [
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            assert!(!delivered_quality_improved(
                passed,
                accepted,
                reverted,
                [1.0].into_iter()
            ));
        }
    }

    #[test]
    fn json_preserves_safe_reversion_and_skipped_assessment_without_claiming_improvement() {
        let mut result = TestResult::success(
            "safety",
            "room",
            "fem",
            "iir",
            1.0,
            1.0,
            None,
            vec![],
            true,
            1,
        );
        result.correction_decision = Some(roomeq_model::CorrectionDecision::IdentityFallback);
        result.stage_outcomes.push(roomeq_model::StageOutcome {
            stage: "final_correlated_bass_acceptance".into(),
            status: roomeq_model::StageStatus::Skipped,
            checks: vec![],
            advisories: vec!["correlated_bass_unassessed:missing_timing".into()],
        });
        let json = serde_json::to_value(result).unwrap();
        assert_eq!(json["passed"], true);
        assert_eq!(json["outcome"], "reverted");
        assert_eq!(json["quality_improved"], false);
        assert_eq!(json["correction_decision"], "identity_fallback");
        assert_eq!(json["stage_outcomes"][0]["status"], "skipped");
        assert_eq!(
            json["stage_outcomes"][0]["advisories"][0],
            "correlated_bass_unassessed:missing_timing"
        );
    }
}
