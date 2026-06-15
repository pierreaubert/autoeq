#[cfg(test)]
mod tests {
    use crate::postscore::{PostOptMetrics, compute_peq_cached, compute_post_optimization_metrics};
    use autoeq::Curve;
    use autoeq::cli::Args;
    use autoeq::PeqModel;
    use clap::Parser;
    use ndarray::Array1;

    #[tokio::test]
    async fn test_compute_post_optimization_headphone() {
        let args = Args::try_parse_from([
            "autoeq-test",
            "--loss",
            "headphone-flat",
            "--sample-rate",
            "48000",
        ])
        .unwrap();

        let freqs = Array1::from_vec(vec![100.0, 500.0, 1000.0, 5000.0, 10000.0]);
        let target_spl = Array1::from_vec(vec![0.0; 5]);
        let input_spl = Array1::from_vec(vec![2.0, 1.5, 1.0, 1.2, 0.8]);

        let target_curve = Curve {
            freq: freqs.clone(),
            spl: target_spl.clone(),
            phase: None,
            ..Default::default()
        };

        let input_curve = Curve {
            freq: freqs.clone(),
            spl: input_spl.clone(),
            phase: None,
            ..Default::default()
        };

        let objective_data = autoeq::optim::ObjectiveDataBuilder::headphone_flat(
            freqs.clone(),
            target_spl.clone(),
            &input_spl - &target_spl,
            48000.0,
            PeqModel::Pk,
        )
        .min_spacing_oct(0.1)
        .max_db(10.0)
        .min_db(-10.0)
        .freq_range(20.0, 20000.0)
        .smoothing(false, 3)
        .build()
        .expect("valid test objective data");

        let opt_params = vec![500.0, 2.0, -2.0]; // Example PEQ params

        let result = compute_post_optimization_metrics(
            &args,
            &objective_data,
            false,
            &opt_params,
            &freqs,
            &target_curve,
            &input_curve,
            &None,
            None,
            None,
        )
        .await;

        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert!(metrics.headphone_loss.is_some());
    }

    #[test]
    fn test_print_optimization_scores_headphone() {
        let args = Args::try_parse_from(["autoeq-test", "--loss", "headphone-flat"]).unwrap();

        let metrics = PostOptMetrics {
            cea2034_metrics: None,
            headphone_loss: Some(5.5),
            pre_cea2034: None,
            pre_headphone_loss: Some(8.2),
        };

        // Should not panic
        crate::postscore::print_optimization_scores(&args, &metrics, Some(0.5), Some(0.3));
    }

    /// Cache key must include actual param values, not just length.
    /// Two different param vectors of the same length must not collide.
    #[test]
    fn test_compute_peq_cached_no_collision() {
        let freq = Array1::from_vec(vec![100.0, 500.0, 1000.0, 5000.0, 10000.0]);
        let params_a = vec![100.0, 1.0, 3.0];
        let params_b = vec![1000.0, 1.0, -3.0];

        let result_a = compute_peq_cached(&freq, &params_a, 48000.0, PeqModel::Pk);
        let result_b = compute_peq_cached(&freq, &params_b, 48000.0, PeqModel::Pk);

        assert_ne!(
            result_a, result_b,
            "Cache collision: different params of same length returned identical PEQ responses"
        );
    }
}
