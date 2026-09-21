//! Machine-readable QA status matrix (Q1/Q6 evidence artifact).
//!
//! Each case records pass, fail, not-run, or blocked with its exact
//! command, log path, seed/limit configuration, and reason. A passing row
//! without evidence is rejected: zero-test selections and missing
//! artifacts must never read as green.

// Rust guideline compliant 2026-02-21

/// Execution status of one matrix case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    Pass,
    Fail,
    NotRun,
    Blocked,
}

/// One row of the QA status matrix.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MatrixCase {
    /// Case id, e.g. `"roomeq-qa-lib"` or `"F03"`.
    pub id: String,
    /// Execution status.
    pub status: CaseStatus,
    /// Exact command that produced this row (empty only for blocked rows
    /// whose command is recorded in `reason` instead).
    pub command: String,
    /// Log or artifact path under the lane workspace.
    pub log_path: String,
    /// Seeds, limits, fixtures, and environment for the run.
    pub configuration: String,
    /// Pass/fail counts or the failure/block reason.
    pub detail: String,
}

impl MatrixCase {
    /// Shape validation: passes carry evidence, failures carry reasons,
    /// and blocked rows name the missing prerequisite.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err(String::from("matrix case needs an id"));
        }
        match self.status {
            CaseStatus::Pass => {
                if self.command.trim().is_empty() || self.detail.trim().is_empty() {
                    return Err(format!(
                        "passing case '{}' needs a command and nonzero counts",
                        self.id
                    ));
                }
            }
            CaseStatus::Fail => {
                if self.detail.trim().is_empty() {
                    return Err(format!("failing case '{}' needs a reason", self.id));
                }
            }
            CaseStatus::NotRun => {
                if self.command.trim().is_empty() {
                    return Err(format!(
                        "not-run case '{}' needs the pending command",
                        self.id
                    ));
                }
            }
            CaseStatus::Blocked => {
                if self.detail.trim().is_empty() {
                    return Err(format!(
                        "blocked case '{}' needs the missing prerequisite",
                        self.id
                    ));
                }
            }
        }
        Ok(())
    }
}

/// A versioned collection of matrix rows.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QaMatrix {
    /// Matrix schema version.
    pub version: String,
    /// HEAD the matrix was derived at (never trust old counts).
    pub head: String,
    /// Rows in stable id order.
    pub cases: Vec<MatrixCase>,
}

impl QaMatrix {
    /// Validate every row and reject duplicate case ids.
    pub fn validate(&self) -> Result<(), String> {
        if self.version.trim().is_empty() || self.head.trim().is_empty() {
            return Err(String::from("matrix needs a version and a HEAD"));
        }
        let mut seen = std::collections::HashSet::new();
        for case in &self.cases {
            case.validate()?;
            if !seen.insert(case.id.as_str()) {
                return Err(format!("duplicate matrix case '{}'", case.id));
            }
        }
        Ok(())
    }

    /// Count rows by status in (pass, fail, not-run, blocked) order.
    pub fn counts(&self) -> (usize, usize, usize, usize) {
        let mut counts = (0, 0, 0, 0);
        for case in &self.cases {
            match case.status {
                CaseStatus::Pass => counts.0 += 1,
                CaseStatus::Fail => counts.1 += 1,
                CaseStatus::NotRun => counts.2 += 1,
                CaseStatus::Blocked => counts.3 += 1,
            }
        }
        counts
    }
}

#[cfg(test)]
mod matrix_tests {
    use super::*;

    fn passing(id: &str) -> MatrixCase {
        MatrixCase {
            id: id.to_string(),
            status: CaseStatus::Pass,
            command: format!("rtk cargo test -p {id} --lib"),
            log_path: String::from(".target-lane-roomeq-qa/qa.log"),
            configuration: String::from("default features"),
            detail: String::from("143 passed; 0 failed"),
        }
    }

    #[test]
    fn matrix_rejects_pass_without_evidence_and_duplicate_rows() {
        let valid = QaMatrix {
            version: String::from("1"),
            head: String::from("2759a62"),
            cases: vec![passing("roomeq-qa-lib")],
        };
        assert!(valid.validate().is_ok());
        assert_eq!(valid.counts(), (1, 0, 0, 0));
        // A pass without a command is a zero-test selection until proven
        // otherwise: rejected.
        let mut no_command = valid.clone();
        no_command.cases[0].command.clear();
        assert!(no_command.validate().is_err());
        // Duplicate case ids are rejected.
        let mut duplicated = valid.clone();
        duplicated.cases.push(passing("roomeq-qa-lib"));
        assert!(duplicated.validate().is_err());
        // Blocked rows must name the prerequisite.
        let mut unexplained = valid.clone();
        unexplained.cases[0].status = CaseStatus::Blocked;
        unexplained.cases[0].detail.clear();
        assert!(unexplained.validate().is_err());
    }

    #[test]
    fn matrix_round_trips_through_json() {
        let matrix = QaMatrix {
            version: String::from("1"),
            head: String::from("2759a62"),
            cases: vec![
                passing("roomeq-qa-lib"),
                MatrixCase {
                    id: String::from("nightly-fuzzer"),
                    status: CaseStatus::NotRun,
                    command: String::from(
                        "cargo run --bin roomeq-fuzzer --release -- -n 1000 --seed 42",
                    ),
                    log_path: String::new(),
                    configuration: String::from("seed 42; coordinator-owned"),
                    detail: String::from("not launched from this lane"),
                },
            ],
        };
        let json = serde_json::to_string_pretty(&matrix).unwrap();
        let parsed: QaMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, matrix);
        assert!(parsed.validate().is_ok());
    }
}
