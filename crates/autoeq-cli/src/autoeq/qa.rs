use anyhow::{Result, bail};

/// Structure to hold QA analysis results
pub(crate) struct QaAnalysisResult {
    pub(super) converge_ok: bool,
    pub(super) spacing_ok: bool,
    pub(super) improvement_ok: bool,
    pub(super) improvement_threshold: f64,
    pub(super) pre_value: f64,
    pub(super) post_value: f64,
}

impl QaAnalysisResult {
    fn passes(&self) -> bool {
        self.converge_ok && self.spacing_ok && self.improvement_ok
    }
}

/// Perform QA analysis similar to qa_check.sh
pub(crate) fn perform_qa_analysis(
    converged: bool,
    spacing_ok: bool,
    pre_score: Option<f64>,
    post_score: Option<f64>,
    threshold: f64,
) -> QaAnalysisResult {
    let pre_value = pre_score.unwrap_or(f64::NAN);
    let post_value = post_score.unwrap_or(f64::NAN);

    // Check convergence
    let converge_ok = converged;

    // Check spacing (already computed)
    let spacing_check_ok = spacing_ok;

    // Check improvement: post > pre + threshold
    let improvement_threshold = pre_value + threshold;
    let improvement_ok = pre_value.is_finite()
        && post_value.is_finite()
        && threshold.is_finite()
        && improvement_threshold.is_finite()
        && post_value > improvement_threshold;

    QaAnalysisResult {
        converge_ok,
        spacing_ok: spacing_check_ok,
        improvement_ok,
        improvement_threshold,
        pre_value,
        post_value,
    }
}

/// Display QA analysis results similar to qa_check.sh
pub(super) fn display_qa_analysis(result: &QaAnalysisResult) {
    println!("Parsed values:");
    println!(
        "  Converge: {} ({})",
        if result.converge_ok { "true" } else { "false" },
        if result.converge_ok { "✓" } else { "✗" }
    );
    println!(
        "  Spacing:  {} ({})",
        if result.spacing_ok { "ok" } else { "ko" },
        if result.spacing_ok { "✓" } else { "✗" }
    );
    println!("  Pre:      {:.3}", result.pre_value);
    println!("  Post:     {:.3}", result.post_value);
    println!(
        "  Improvement: {:.3} > {:.3} + {:.1} = {:.3} ({})",
        result.post_value,
        result.pre_value,
        result.improvement_threshold - result.pre_value,
        result.improvement_threshold,
        if result.improvement_ok { "✓" } else { "✗" }
    );
    println!();

    // Final result
    if result.passes() {
        println!("OK");
    } else {
        println!("FAIL");
    }
}

/// Return a failing process result for any failed QA criterion.
pub(crate) fn require_qa_pass(result: &QaAnalysisResult) -> Result<()> {
    if result.passes() {
        Ok(())
    } else {
        bail!(
            "AutoEQ QA failed: convergence={}, spacing={}, improvement={}",
            result.converge_ok,
            result.spacing_ok,
            result.improvement_ok
        )
    }
}
