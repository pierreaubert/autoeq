#[cfg(test)]
mod tests {
    use crate::autoeq_command::runopt::perform_optimization_with_backend;
    use autoeq::OptimParams;
    use autoeq::PeqModel;
    use autoeq::cli::Args;
    use autoeq::loss::LossType;
    use autoeq::optim::{
        MockOptimizerBackend, ObjectiveData, ObjectiveDataBuilder, OptimizerBackend,
    };
    use clap::Parser;
    use ndarray::Array1;

    const GLOBAL_STATUS: &str = "AutoEQ DE converged (nfev=100)";
    const LOCAL_STATUS: &str = "AutoEQ COBYLA converged (nfev=50)";

    fn test_params(refine: bool) -> OptimParams {
        let mut argv = vec![
            "autoeq-test".to_string(),
            "--loss".to_string(),
            "speaker-flat".to_string(),
            "--num-filters".to_string(),
            "2".to_string(),
        ];
        if refine {
            argv.push("--refine".to_string());
        }
        let args = Args::try_parse_from(argv).unwrap();
        OptimParams::from(&args)
    }

    fn test_objective_data() -> ObjectiveData {
        let freqs = Array1::from_vec(vec![100.0, 500.0, 1000.0, 5000.0, 10000.0]);
        let deviation = Array1::from_vec(vec![2.0, 1.5, 1.0, 1.2, 0.8]);
        let target = Array1::zeros(freqs.len());

        ObjectiveDataBuilder::new(freqs, target, deviation, 48000.0, PeqModel::Pk, LossType::SpeakerFlat)
            .min_spacing_oct(0.1)
            .max_db(10.0)
            .min_db(-10.0)
            .freq_range(20.0, 20000.0)
            .smoothing(false, 3)
            .build()
            .expect("valid test objective data")
    }

    /// Backend that mutates the parameter vector during refinement so the
    /// global-snapshot rollback is observable (the stock mock backend never
    /// touches `x`, which would make restore a no-op).
    struct MutatingBackend {
        global_loss: f64,
        local_loss: f64,
    }

    impl OptimizerBackend for MutatingBackend {
        fn optimize_filters(
            &self,
            _x: &mut [f64],
            _lower_bounds: &[f64],
            _upper_bounds: &[f64],
            _objective: ObjectiveData,
            _params: &OptimParams,
        ) -> Result<(String, f64), (String, f64)> {
            Ok((GLOBAL_STATUS.to_string(), self.global_loss))
        }

        fn optimize_filters_with_callback(
            &self,
            x: &mut [f64],
            lower_bounds: &[f64],
            upper_bounds: &[f64],
            objective: ObjectiveData,
            params: &OptimParams,
            _callback: autoeq::optim::OptimProgressCallback,
        ) -> Result<(String, f64), (String, f64)> {
            self.optimize_filters(x, lower_bounds, upper_bounds, objective, params)
        }

        fn optimize_filters_with_algo_override(
            &self,
            x: &mut [f64],
            _lower_bounds: &[f64],
            upper_bounds: &[f64],
            _objective: ObjectiveData,
            _params: &OptimParams,
            _algo_override: Option<&str>,
        ) -> Result<(String, f64), (String, f64)> {
            // Simulate a regressing local step that drags every parameter
            // to its upper bound (still in-bounds, so the rejection comes
            // purely from the worse objective value).
            for (value, upper) in x.iter_mut().zip(upper_bounds.iter()) {
                *value = *upper;
            }
            Ok((LOCAL_STATUS.to_string(), self.local_loss))
        }
    }

    #[test]
    fn worse_refinement_keeps_global_result() {
        let params = test_params(true);
        let objective = test_objective_data();
        let backend = MockOptimizerBackend::ok(GLOBAL_STATUS, 1.0)
            .with_refine_result(Ok((LOCAL_STATUS.to_string(), 2.0)));

        let result = perform_optimization_with_backend(&params, &objective, None, &backend)
            .expect("a worse refinement must keep the usable global result");

        assert_eq!(result.post_objective, Some(1.0));
        assert_eq!(result.optimizer_evidence.len(), 2);
        assert!(
            result.optimizer_evidence[0].selected_for_output,
            "global pass supplies the output when refinement regresses"
        );
        assert!(
            !result.optimizer_evidence[1].selected_for_output,
            "regressing refinement must not be selected for output"
        );
    }

    #[test]
    fn failed_refinement_keeps_global_result() {
        let params = test_params(true);
        let objective = test_objective_data();
        let backend = MockOptimizerBackend::ok(GLOBAL_STATUS, 1.0).with_refine_result(Err((
            "AutoEQ COBYLA: backend failure".to_string(),
            f64::INFINITY,
        )));

        let result = perform_optimization_with_backend(&params, &objective, None, &backend)
            .expect("a failed refinement must not discard the usable global result");

        assert_eq!(result.post_objective, Some(1.0));
        assert!(result.optimizer_evidence[0].selected_for_output);
        assert!(!result.optimizer_evidence[1].selected_for_output);
    }

    #[test]
    fn regressing_refinement_restores_global_parameters() {
        let params = test_params(true);
        let objective = test_objective_data();
        let backend = MutatingBackend {
            global_loss: 1.0,
            local_loss: 2.0,
        };

        let result = perform_optimization_with_backend(&params, &objective, None, &backend)
            .expect("regressing refinement must roll back");

        let (lower, upper) = autoeq::workflow::setup_bounds(&params);
        let expected = autoeq::workflow::initial_guess(&params, &lower, &upper);
        assert_eq!(
            result.params, expected,
            "rejected refinement must restore the global snapshot"
        );
        assert_eq!(result.post_objective, Some(1.0));
    }

    #[test]
    fn improving_refinement_is_accepted() {
        let params = test_params(true);
        let objective = test_objective_data();
        let backend = MockOptimizerBackend::ok(GLOBAL_STATUS, 2.0)
            .with_refine_result(Ok((LOCAL_STATUS.to_string(), 1.0)));

        let result = perform_optimization_with_backend(&params, &objective, None, &backend)
            .expect("improving refinement should be accepted");

        assert_eq!(result.post_objective, Some(1.0));
        assert!(!result.optimizer_evidence[0].selected_for_output);
        assert!(result.optimizer_evidence[1].selected_for_output);
    }
}
