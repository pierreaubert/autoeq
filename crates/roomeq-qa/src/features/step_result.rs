use super::consts::SELF_REGRESSION_TOLERANCE;
use super::consts::SLOPE_TOLERANCE;
use super::consts::STEP_REGRESSION_TOLERANCE;
use super::types::StepResult;

pub(super) fn validate_pass(
    pass_name: &str,
    results: &[StepResult],
    enforce_flat_slope: bool,
) -> Vec<String> {
    let mut errors = Vec::new();

    // Track whether we've crossed a loss-change boundary. Once crossed,
    // flat-score step-over-step comparisons are invalid for all subsequent steps.
    let mut loss_changed = false;

    let baseline_epa = results.first().and_then(|s| s.epa_preference);

    for (i, step) in results.iter().enumerate() {
        if step.changes_loss {
            loss_changed = true;
        }

        // Convergence: every step must produce a finite loss
        if !step.post_score.is_finite() {
            errors.push(format!(
                "  {} step '{}': post_score is not finite — optimizer failed to converge",
                pass_name, step.name
            ));
            continue;
        }
        if step.correction_reverted {
            errors.push(format!(
                "  {} step '{}': REVERTED by runtime correction acceptance",
                pass_name, step.name
            ));
        }

        if loss_changed {
            // Once the objective changes, flat scores are no longer directly
            // comparable. Perceptual evidence is mandatory rather than a
            // check that silently disappears when EPA is absent.
            match (baseline_epa, step.epa_preference) {
                (Some(baseline), Some(current)) if current < baseline * 0.95 => {
                    errors.push(format!(
                        "  {} step '{}': EPA preference {:.3} < baseline {:.3} * 0.95 — perceptual regression",
                        pass_name, step.name, current, baseline
                    ));
                }
                (None, _) => errors.push(format!(
                    "  {} step '{}': baseline EPA preference is missing",
                    pass_name, step.name
                )),
                (_, None) => errors.push(format!(
                    "  {} step '{}': EPA preference is missing after a loss-changing feature",
                    pass_name, step.name
                )),
                _ => {}
            }
        } else {
            // No loss change yet — flat-score checks are valid.

            // Per-step sanity: post_score should not be much worse than own pre_score
            if step.post_score > step.pre_score * SELF_REGRESSION_TOLERANCE {
                errors.push(format!(
                    "  {} step '{}': post_score {:.4} > pre_score {:.4} * {:.2} — optimization made things worse",
                    pass_name, step.name, step.post_score, step.pre_score, SELF_REGRESSION_TOLERANCE
                ));
            }

            // Step-over-step regression check
            if i > 0 {
                let prev = &results[i - 1];
                if step.post_score > prev.post_score * STEP_REGRESSION_TOLERANCE {
                    errors.push(format!(
                        "  {} step '{}': post_score {:.4} > prev {:.4} * {:.2} — excessive regression",
                        pass_name,
                        step.name,
                        step.post_score,
                        prev.post_score,
                        STEP_REGRESSION_TOLERANCE
                    ));
                }
            }

            // Slope invariant
            if enforce_flat_slope && step.worst_slope > SLOPE_TOLERANCE {
                errors.push(format!(
                    "  {} step '{}': slope {:.2} dB/oct > {:.1} tolerance — positive tilt detected",
                    pass_name, step.name, step.worst_slope, SLOPE_TOLERANCE
                ));
            }
        }
    }

    // End-of-pass: baseline must improve; runtime reversion is a QA outcome,
    // not an exemption from the quality gate.
    if let Some(baseline) = results.first()
        && baseline.post_score >= baseline.pre_score
    {
        errors.push(format!(
            "  {} step '{}': post_score {:.4} >= pre_score {:.4} — EQ did not improve over raw",
            pass_name, baseline.name, baseline.post_score, baseline.pre_score
        ));
    }

    errors
}

pub(super) fn print_pass_results(results: &[StepResult]) {
    let baseline_epa = results.first().and_then(|s| s.epa_preference);

    for (i, step) in results.iter().enumerate() {
        let outcome = if step.correction_reverted {
            "REVERTED"
        } else if step.post_score < step.pre_score {
            "PASS"
        } else {
            "FAIL"
        };
        let epa = match step.epa_preference {
            Some(v) => format!("epa={:.3}", v),
            None => "epa=n/a".to_string(),
        };
        let epa_str = format!("[{outcome}] {epa}");

        if i == 0 {
            println!(
                "  Step {}: {:30} post={:.4}  slope={:.2}  {}",
                i, step.name, step.post_score, step.worst_slope, epa_str
            );
        } else {
            let prev = &results[i - 1];
            let pct = if prev.post_score > 0.0 {
                (step.post_score - prev.post_score) / prev.post_score * 100.0
            } else {
                0.0
            };

            let epa_vs_baseline = match (baseline_epa, step.epa_preference) {
                (Some(b), Some(c)) if b > 0.0 => {
                    format!("  epa vs baseline: {:+.1}%", (c - b) / b * 100.0)
                }
                _ => String::new(),
            };

            println!(
                "  Step {}: {:30} post={:.4}  slope={:.2}  (vs prev: {:+.1}%)  {}{}",
                i, step.name, step.post_score, step.worst_slope, pct, epa_str, epa_vs_baseline
            );
        }
    }

    let reverted = results
        .iter()
        .filter(|step| step.correction_reverted)
        .count();
    println!("    Outcomes: REVERTED={reverted}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline(pre_score: f64, post_score: f64, correction_reverted: bool) -> StepResult {
        StepResult {
            name: "Baseline",
            pre_score,
            post_score,
            worst_slope: 0.0,
            changes_loss: false,
            epa_preference: None,
            correction_reverted,
        }
    }

    #[test]
    fn explicitly_reverted_identity_baseline_is_a_reverted_qa_failure() {
        let errors = validate_pass("Pass A", &[baseline(10.0, 10.0, true)], true);
        assert!(
            errors.iter().any(|error| error.contains("did not improve")),
            "expected no-improvement error, got: {errors:?}"
        );
    }

    #[test]
    fn ordinary_identity_baseline_still_fails() {
        let errors = validate_pass("Pass A", &[baseline(10.0, 10.0, false)], true);
        assert!(
            errors.iter().any(|error| error.contains("did not improve")),
            "expected no-improvement error, got: {errors:?}"
        );
    }

    #[test]
    fn explicitly_reverted_regression_still_fails() {
        let errors = validate_pass("Pass A", &[baseline(10.0, 10.1, true)], true);
        assert!(
            errors.iter().any(|error| error.contains("did not improve")),
            "expected no-improvement error, got: {errors:?}"
        );
    }
}
