/// Result of configuration validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the configuration is valid
    pub is_valid: bool,
    /// Critical errors that prevent optimization
    pub errors: Vec<String>,
    /// Non-critical warnings that may affect results
    pub warnings: Vec<String>,
    /// Named-stage evidence for public compatibility adapters. Internal rule
    /// accumulators leave this absent until assembled by the canonical pipeline.
    pub staged_report: Option<crate::roomeq::types::ConfigValidationReport>,
}

impl ValidationResult {
    /// Create a valid result with no errors or warnings
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            staged_report: None,
        }
    }

    /// Add an error (marks result as invalid)
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
    }

    /// Add a warning (does not affect validity)
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Merge another validation result into this one
    #[allow(dead_code)]
    pub fn merge(&mut self, other: ValidationResult) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        self.is_valid = self.is_valid && other.is_valid;
        if other.staged_report.is_some() {
            self.staged_report = other.staged_report;
        }
    }

    /// Print validation results to stderr
    pub fn print_results(&self) {
        for warning in &self.warnings {
            eprintln!("Warning: {}", warning);
        }
        for error in &self.errors {
            eprintln!("Error: {}", error);
        }
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::valid()
    }
}

/// Shared context passed to each independent validation rule.
///
/// This type is the decomposition scaffolding for `validate_optimizer_config`.
/// Each rule receives the optimizer config and a mutable result, allowing rules
/// to be unit-tested in isolation.
#[derive(Debug)]
pub struct ValidationContext<'a> {
    pub opt: &'a crate::roomeq::types::OptimizerConfig,
    pub result: &'a mut ValidationResult,
}

impl<'a> ValidationContext<'a> {
    pub fn new(
        opt: &'a crate::roomeq::types::OptimizerConfig,
        result: &'a mut ValidationResult,
    ) -> Self {
        Self { opt, result }
    }

    pub fn add_error(&mut self, error: impl Into<String>) {
        self.result.add_error(error.into());
    }

    pub fn add_warning(&mut self, warning: impl Into<String>) {
        self.result.add_warning(warning.into());
    }
}
