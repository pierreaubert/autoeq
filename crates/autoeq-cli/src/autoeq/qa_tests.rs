#[cfg(test)]
mod tests {
    use crate::autoeq_command::qa::{
        QaAnalysisResult, display_qa_analysis, perform_qa_analysis, require_qa_pass,
    };

    #[test]
    fn test_perform_qa_analysis_all_pass() {
        let result = perform_qa_analysis(
            true,      // converged
            true,      // spacing_ok
            Some(5.0), // pre_score
            Some(6.0), // post_score (improved)
            0.5,       // threshold
        );

        assert!(result.converge_ok);
        assert!(result.spacing_ok);
        assert!(result.improvement_ok);
        assert!(require_qa_pass(&result).is_ok());
    }

    #[test]
    fn test_perform_qa_analysis_no_improvement() {
        let result = perform_qa_analysis(
            true,
            true,
            Some(5.0),
            Some(5.2), // Not enough improvement
            0.5,
        );

        assert!(result.converge_ok);
        assert!(result.spacing_ok);
        assert!(!result.improvement_ok);
        assert!(require_qa_pass(&result).is_err());
    }

    #[test]
    fn test_perform_qa_analysis_no_convergence_returns_error() {
        let result = perform_qa_analysis(false, true, Some(5.0), Some(6.0), 0.5);

        assert!(!result.converge_ok);
        assert!(result.spacing_ok);
        assert!(result.improvement_ok);
        assert!(require_qa_pass(&result).is_err());
    }

    #[test]
    fn test_perform_qa_analysis_with_nan() {
        let result = perform_qa_analysis(
            true,
            true,
            None, // pre_score is NaN
            Some(4.0),
            0.5,
        );

        // Should handle NaN gracefully
        assert!(result.pre_value.is_nan());
        assert!(!result.improvement_ok); // Won't pass with NaN
        assert!(require_qa_pass(&result).is_err());
    }

    #[test]
    fn shared_qa_rejects_nonfinite_scores_thresholds_and_threshold_overflow() {
        let cases = [
            (0.0, f64::INFINITY, 0.0),
            (0.0, f64::NEG_INFINITY, 0.0),
            (f64::INFINITY, 1.0, 0.0),
            (f64::NEG_INFINITY, 1.0, 0.0),
            (0.0, 1.0, f64::INFINITY),
            (0.0, 1.0, f64::NEG_INFINITY),
            (f64::NAN, 1.0, 0.0),
            (0.0, f64::NAN, 0.0),
            (0.0, 1.0, f64::NAN),
            (f64::MAX, f64::MAX, f64::MAX),
            (-f64::MAX, 0.0, -f64::MAX),
        ];
        for (pre, post, threshold) in cases {
            let result = perform_qa_analysis(true, true, Some(pre), Some(post), threshold);
            assert!(result.converge_ok && result.spacing_ok);
            assert!(
                !result.improvement_ok,
                "pre={pre:?} post={post:?} threshold={threshold:?}"
            );
            assert!(require_qa_pass(&result).is_err());
        }
    }

    #[test]
    fn shared_qa_preserves_exact_finite_numeric_threshold_comparison() {
        // Negative explicit thresholds retain their original comparison semantics.
        for (pre, post, threshold, expected) in [
            (5.0, 4.0, -2.0, true),
            (5.0, 3.0, -2.0, false),
            (5.0, 6.0, 1.0, false),
            (5.0, 6.0, 0.5, true),
            (-1.0, 0.0, -0.0, true),
            (0.0, -0.0, 0.0, false),
        ] {
            let result = perform_qa_analysis(true, true, Some(pre), Some(post), threshold);
            assert_eq!(result.improvement_threshold, pre + threshold);
            assert_eq!(result.improvement_ok, expected);
            assert_eq!(require_qa_pass(&result).is_ok(), expected);
        }
    }

    #[test]
    fn test_display_qa_analysis() {
        let result = QaAnalysisResult {
            converge_ok: true,
            spacing_ok: false,
            improvement_ok: true,
            improvement_threshold: 4.5,
            pre_value: 5.0,
            post_value: 4.0,
        };

        // Should not panic
        display_qa_analysis(&result);
        assert!(require_qa_pass(&result).is_err());
    }

    #[test]
    fn autoeq_command_bare_qa_uses_strict_zero_threshold() {
        use clap::Parser;
        let args = autoeq::cli::Args::parse_from(["autoeq", "--qa"]);
        let threshold = args.qa.expect("bare QA enabled");
        assert_eq!(threshold, 0.0);
        assert!(
            require_qa_pass(&perform_qa_analysis(
                true,
                true,
                Some(5.0),
                Some(6.0),
                threshold
            ))
            .is_ok()
        );
        assert!(
            require_qa_pass(&perform_qa_analysis(
                true,
                true,
                Some(5.0),
                Some(5.0),
                threshold
            ))
            .is_err()
        );
        assert!(
            require_qa_pass(&perform_qa_analysis(
                false,
                true,
                Some(5.0),
                Some(6.0),
                threshold
            ))
            .is_err()
        );
    }
}
