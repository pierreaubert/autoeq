#[cfg(test)]
mod tests {
    use crate::autoeq_command::runopt::{
        perform_optimization_with_backend, perform_optimization_with_backend_and_candidate,
        perform_optimization_with_candidate, perform_optimization_with_exact_checkpoint,
    };
    use autoeq::OptimParams;
    use autoeq::PeqModel;
    use autoeq::cli::Args;
    use autoeq::loss::LossType;
    use autoeq::optim::OptimizerBackendCompletion;
    use autoeq::optim::backend::{BackendSearchEvidence, BackendSearchStopCause};
    use autoeq::optim::{
        MockOptimizerBackend, ObjectiveData, ObjectiveDataBuilder, OptimizerBackend,
    };
    use clap::Parser;
    use ndarray::Array1;

    const GLOBAL_STATUS: &str = "AutoEQ DE converged (nfev=100)";
    const LOCAL_STATUS: &str = "AutoEQ COBYLA converged (nfev=50)";

    struct DiagnosticBackend;

    impl OptimizerBackend for DiagnosticBackend {
        fn optimize_filters(
            &self,
            _x: &mut [f64],
            _lower: &[f64],
            _upper: &[f64],
            _objective: ObjectiveData,
            _params: &OptimParams,
        ) -> Result<(String, f64), (String, f64)> {
            Ok(("Metaheuristics(rga)".to_string(), 1.0))
        }

        fn optimize_filters_with_completion_evidence(
            &self,
            x: &mut [f64],
            lower: &[f64],
            upper: &[f64],
            objective: ObjectiveData,
            params: &OptimParams,
        ) -> (
            Result<(String, f64), (String, f64)>,
            Option<autoeq::optim::de::DECompletion>,
            Option<BackendSearchEvidence>,
        ) {
            (
                self.optimize_filters(x, lower, upper, objective, params),
                None,
                Some(BackendSearchEvidence {
                    completion: OptimizerBackendCompletion::EvaluationLimit,
                    stop_cause: BackendSearchStopCause::GenerationLimit,
                    evaluations: 72,
                    denied_evaluations: 0,
                    generations: 9,
                    generation_limit: 10,
                    task_callbacks: 10,
                    population_mean: Some(1.5),
                    population_stddev: Some(0.25),
                }),
            )
        }

        fn optimize_filters_with_callback(
            &self,
            x: &mut [f64],
            lower: &[f64],
            upper: &[f64],
            objective: ObjectiveData,
            params: &OptimParams,
            _callback: autoeq::optim::OptimProgressCallback,
        ) -> Result<(String, f64), (String, f64)> {
            self.optimize_filters(x, lower, upper, objective, params)
        }

        fn optimize_filters_with_algo_override(
            &self,
            x: &mut [f64],
            lower: &[f64],
            upper: &[f64],
            objective: ObjectiveData,
            params: &OptimParams,
            _algo: Option<&str>,
        ) -> Result<(String, f64), (String, f64)> {
            self.optimize_filters(x, lower, upper, objective, params)
        }
    }

    #[test]
    fn ordinary_cli_route_exports_rga_search_counters_without_claiming_convergence() {
        let mut params = test_params(false);
        params.algo = "mh:rga".to_string();
        let result = perform_optimization_with_backend(
            &params,
            &test_objective_data(),
            None,
            &DiagnosticBackend,
        )
        .expect("diagnostic backend returns a finite candidate");
        let evidence = &result.optimizer_evidence[0];
        assert_eq!(evidence.backend_evaluation_count, Some(72));
        assert_eq!(
            evidence.backend_stop_cause,
            Some(BackendSearchStopCause::GenerationLimit)
        );
        assert_eq!(evidence.generation_count, Some(9));
        assert_eq!(evidence.generation_limit, Some(10));
        assert_eq!(evidence.task_callback_count, Some(10));
        assert_eq!(evidence.population_fitness_mean, Some(1.5));
        assert_eq!(evidence.population_fitness_stddev, Some(0.25));
        assert_eq!(
            evidence.termination,
            autoeq::optim::OptimizerTermination::NonConverged
        );
        assert!(!result.converged);
    }

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

        ObjectiveDataBuilder::new(
            freqs,
            target,
            deviation,
            48000.0,
            PeqModel::Pk,
            LossType::SpeakerFlat,
        )
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

    struct CapturingBackend {
        initial: std::sync::Arc<std::sync::Mutex<Option<Vec<f64>>>>,
    }

    impl OptimizerBackend for CapturingBackend {
        fn optimize_filters(
            &self,
            x: &mut [f64],
            _lower_bounds: &[f64],
            _upper_bounds: &[f64],
            _objective: ObjectiveData,
            _params: &OptimParams,
        ) -> Result<(String, f64), (String, f64)> {
            *self
                .initial
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(x.to_vec());
            Ok((GLOBAL_STATUS.to_string(), 1.0))
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
            lower_bounds: &[f64],
            upper_bounds: &[f64],
            objective: ObjectiveData,
            params: &OptimParams,
            _algo_override: Option<&str>,
        ) -> Result<(String, f64), (String, f64)> {
            self.optimize_filters(x, lower_bounds, upper_bounds, objective, params)
        }
    }

    #[test]
    fn explicit_candidate_reaches_the_optimizer_as_its_initial_vector() {
        let params = test_params(false);
        let objective = test_objective_data();
        let (lower, upper) = autoeq::workflow::setup_bounds(&params);
        let mut candidate = autoeq::workflow::initial_guess(&params, &lower, &upper);
        candidate[2] = 1.25;
        let captured = std::sync::Arc::new(std::sync::Mutex::new(None));
        let backend = CapturingBackend {
            initial: std::sync::Arc::clone(&captured),
        };

        let result = perform_optimization_with_backend_and_candidate(
            &params,
            &objective,
            None,
            Some(&candidate),
            &backend,
        )
        .expect("valid explicit candidate should reach the optimizer");

        assert_eq!(
            captured
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .as_deref(),
            Some(candidate.as_slice()),
            "the optimizer must receive the saved vector, not a newly generated guess"
        );
        assert_eq!(result.params, candidate);
    }

    #[test]
    fn warm_start_rejects_backend_that_ignores_initial_candidate() {
        let mut params = test_params(false);
        params.algo = "mh:de".into();
        let objective = test_objective_data();
        let (lower, upper) = autoeq::workflow::setup_bounds(&params);
        let candidate = autoeq::workflow::initial_guess(&params, &lower, &upper);

        let error = match perform_optimization_with_candidate(&params, &objective, None, &candidate)
        {
            Ok(_) => panic!("MH must not silently ignore a requested warm start"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("does not use the supplied initial candidate")
        );
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
    fn best_effort_global_reports_not_converged() {
        let params = test_params(false);
        let objective = test_objective_data();
        let backend = MockOptimizerBackend::ok(
            "AutoEQ DE: maximum evaluations reached (not converged, nfev=42)",
            1.0,
        );

        let result = perform_optimization_with_backend(&params, &objective, None, &backend)
            .expect("a best-effort global result is usable");

        assert_eq!(result.post_objective, Some(1.0));
        assert!(
            !result.converged,
            "best-effort transport success must not report converged=true"
        );
        assert_eq!(result.optimizer_evidence.len(), 1);
        assert!(result.optimizer_evidence[0].best_effort);
    }

    #[test]
    fn legacy_convergence_text_does_not_claim_typed_convergence() {
        let params = test_params(false);
        let objective = test_objective_data();
        let backend = MockOptimizerBackend::ok(GLOBAL_STATUS, 1.0);

        let result = perform_optimization_with_backend(&params, &objective, None, &backend)
            .expect("a finite legacy result should remain usable");

        assert!(!result.converged);
        assert_eq!(result.post_objective, Some(1.0));
        assert_eq!(result.optimizer_evidence.len(), 1);
        assert_eq!(
            result.optimizer_evidence[0].termination,
            autoeq::optim::OptimizerTermination::NonConverged,
            "legacy status text is diagnostic, even when it says converged"
        );
        assert!(result.optimizer_evidence[0].best_effort);
    }

    #[test]
    fn accepted_best_effort_refinement_reports_not_converged() {
        let params = test_params(true);
        let objective = test_objective_data();
        let backend = MockOptimizerBackend::ok(GLOBAL_STATUS, 2.0).with_refine_result(Ok((
            "AutoEQ COBYLA: maximum evaluations reached (not converged, nfev=30)".to_string(),
            1.0,
        )));

        let result = perform_optimization_with_backend(&params, &objective, None, &backend)
            .expect("improving best-effort refinement should be accepted");

        assert_eq!(result.post_objective, Some(1.0));
        assert!(
            !result.converged,
            "accepted best-effort refinement must not report converged=true"
        );
        assert!(result.optimizer_evidence[1].selected_for_output);
    }

    #[test]
    fn optimization_result_carries_apo_roundtrip_gap() {
        let params = test_params(false);
        let objective = test_objective_data();
        let backend = MockOptimizerBackend::ok(GLOBAL_STATUS, 1.0);

        let result = perform_optimization_with_backend(&params, &objective, None, &backend)
            .expect("optimization should succeed");

        let gap = result
            .apo_roundtrip_gap
            .expect("PEQ runs must report the APO round-trip gap");
        assert!(
            gap.is_finite() && gap >= 0.0,
            "round-trip gap must be a finite non-negative drift, got {gap:?}"
        );
    }

    #[test]
    fn optimization_result_retains_the_selected_boxes_and_objective_envelopes() {
        let params = test_params(false);
        let mut objective = test_objective_data();
        let boost = vec![(20.0, 8.0), (20_000.0, 3.0)];
        let cut = vec![(20.0, -8.0), (20_000.0, -3.0)];
        objective.max_boost_envelope = Some(boost.clone());
        objective.min_cut_envelope = Some(cut.clone());
        let (mut lower, mut upper) = autoeq::workflow::setup_bounds(&params);
        lower[0] -= 0.125;
        upper[0] += 0.125;

        let result = perform_optimization_with_backend(
            &params,
            &objective,
            Some((lower.clone(), upper.clone())),
            &MockOptimizerBackend::ok(GLOBAL_STATUS, 1.0),
        )
        .expect("valid selected bounds should reach the optimizer");

        assert_eq!(result.effective_envelope.lower_bounds, lower);
        assert_eq!(result.effective_envelope.upper_bounds, upper);
        assert_eq!(
            result.effective_envelope.constraints.global_max_q,
            params.max_q
        );
        assert_eq!(result.effective_envelope.max_boost_envelope, Some(boost));
        assert_eq!(result.effective_envelope.min_cut_envelope, Some(cut));
        assert_eq!(
            result.effective_envelope.composite_frequencies_hz,
            objective.freqs.as_slice().unwrap()
        );
        assert_eq!(
            result.effective_envelope.composite_band_hz,
            [objective.min_freq, objective.max_freq]
        );
    }

    #[test]
    fn optimization_rejects_malformed_effective_envelope_before_search() {
        let params = test_params(false);
        let objective = test_objective_data();
        let (mut lower, upper) = autoeq::workflow::setup_bounds(&params);
        lower[0] = f64::NAN;
        let error = perform_optimization_with_backend(
            &params,
            &objective,
            Some((lower, upper)),
            &MockOptimizerBackend::ok(GLOBAL_STATUS, 1.0),
        )
        .err()
        .expect("non-finite selected bounds must fail before optimizer dispatch");
        assert!(
            error
                .to_string()
                .contains("bounds at parameter 0 are invalid")
        );

        let mut malformed_objective = test_objective_data();
        malformed_objective.max_boost_envelope = Some(vec![(100.0, f64::INFINITY)]);
        let error = perform_optimization_with_backend(
            &params,
            &malformed_objective,
            None,
            &MockOptimizerBackend::ok(GLOBAL_STATUS, 1.0),
        )
        .err()
        .expect("invalid objective envelope knots must fail before optimizer dispatch");
        assert!(error.to_string().contains("non-finite bound"));
    }

    #[test]
    fn cli_progress_callback_saves_only_a_valid_improved_candidate() {
        let params = test_params(false);
        let objective = test_objective_data();
        let (lower, upper) = autoeq::workflow::setup_bounds(&params);
        let constraint_spec = autoeq::optim::OwnedConstraintSpec::from_params(&params)
            .expect("test constraints should be valid");
        let candidate = autoeq::workflow::initial_guess(&params, &lower, &upper);
        let finalized = autoeq::optim::finalize_candidate(
            "test-progress",
            &candidate,
            &objective,
            &constraint_spec.as_spec(),
        )
        .expect("test initial candidate should be feasible");
        let directory = tempfile::tempdir().expect("temporary checkpoint directory");
        let path = directory.path().join("optimizer_state.json");
        let identity = super::super::CliCheckpointIdentity {
            measurement: "measurement".into(),
            config: "configuration".into(),
            normalization: "normalized-measurement".into(),
            sample_rate: params.sample_rate,
            lower_bounds: lower,
            upper_bounds: upper,
            algorithm: "autoeq:de".into(),
            algorithm_version: autoeq::optim::OPTIMIZER_IMPLEMENTATION_VERSION.into(),
            budget: 100,
            seed: params.seed,
        };
        let best_loss = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut callback = super::super::cli_checkpoint_callback(
            path.clone(),
            identity.clone(),
            objective.clone(),
            constraint_spec,
            best_loss,
        );
        let update = autoeq::optim::setup::ProgressUpdate {
            iteration: 7,
            max_iterations: 100,
            loss: finalized.loss,
            score: None,
            convergence: 0.0,
            params: finalized.params,
            biquads: Vec::new(),
            filter_response: Vec::new(),
        };

        callback(&update).expect("valid progress candidate should be saved");
        let saved = autoeq::workflow::resume::load_optimizer_state(&path)
            .expect("checkpoint should load")
            .expect("checkpoint should exist");
        assert_eq!(saved.iteration, 7);
        assert_eq!(
            saved.lower_bounds.as_deref(),
            Some(identity.lower_bounds.as_slice())
        );
        saved
            .check_warm_start_compatible(&identity.as_identity())
            .expect("saved progress candidate should match current identity");

        let mut tied_update = update;
        tied_update.iteration = 9;
        callback(&tied_update).expect("a tied candidate should be ignored safely");
        let saved_again = autoeq::workflow::resume::load_optimizer_state(&path)
            .expect("checkpoint should still load")
            .expect("checkpoint should remain present");
        assert_eq!(
            saved_again.iteration, 7,
            "tied candidate must not replace best"
        );
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

    #[test]
    fn exact_resume_identity_rejection_precedes_any_objective_score() {
        let mut params = test_params(false);
        params.algo = "autoeq:de".to_owned();
        params.population = 8;
        params.maxeval = 1_000;
        params.seed = Some(12_345);
        params.no_parallel = true;
        params.parallel_threads = 1;
        params.refine = false;

        let source_objective = test_objective_data();
        let checkpoint = std::sync::Arc::new(std::sync::Mutex::new(None));
        let checkpoint_for_callback = std::sync::Arc::clone(&checkpoint);
        let save_and_stop: autoeq::optim::de::DECheckpointSaveCallback = Box::new(move |state| {
            if state.generation >= 1 && state.terminal.is_none() {
                *checkpoint_for_callback
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(state.clone());
                return Err("test interruption after generation barrier".to_owned());
            }
            Ok(())
        });
        let _ = autoeq::optim::setup::perform_optimization_with_run_descriptor_and_exact_checkpoint(
            &params,
            &source_objective,
            autoeq::optim::setup::ExactDECheckpointOptions {
                checkpoint: None,
                run_identity: "saved-measurement-v1".to_owned(),
                save_callback: save_and_stop,
            },
            Box::new(|_| autoeq::de::CallbackAction::Continue),
        );
        let checkpoint = checkpoint
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .expect("test solver reached and saved a generation barrier");

        let requested_objective = test_objective_data();
        let error = match perform_optimization_with_exact_checkpoint(
            &params,
            &requested_objective,
            autoeq::optim::setup::ExactDECheckpointOptions {
                checkpoint: Some(checkpoint),
                run_identity: "changed-measurement-v1".to_owned(),
                save_callback: Box::new(|_| Ok(())),
            },
        ) {
            Ok(_) => panic!("changed identity must reject saved exact state"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("identity"));
        assert!(
            requested_objective.prepared.get().is_none(),
            "CLI exact-resume path must reject stale identity before scoring"
        );
    }

    #[test]
    fn exact_resume_rejects_same_outer_identity_with_stale_math_build_or_source_before_scoring() {
        let mut params = test_params(false);
        params.algo = "autoeq:de".to_owned();
        params.population = 8;
        params.maxeval = 1_000;
        params.seed = Some(12_345);
        params.no_parallel = true;
        params.parallel_threads = 1;
        params.refine = false;

        let source_objective = test_objective_data();
        let captured_checkpoint = std::sync::Arc::new(std::sync::Mutex::new(None));
        let checkpoint_for_callback = std::sync::Arc::clone(&captured_checkpoint);
        let stop_after_barrier: autoeq::optim::de::DECheckpointSaveCallback =
            Box::new(move |checkpoint| {
                if checkpoint.generation >= 1 && checkpoint.terminal.is_none() {
                    *checkpoint_for_callback
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        Some(checkpoint.clone());
                    return Err("intentional first-run stop".to_owned());
                }
                Ok(())
            });
        let _ = autoeq::optim::setup::perform_optimization_with_run_descriptor_and_exact_checkpoint(
            &params,
            &source_objective,
            autoeq::optim::setup::ExactDECheckpointOptions {
                checkpoint: None,
                run_identity: "same-run-identity".to_owned(),
                save_callback: stop_after_barrier,
            },
            Box::new(|_| autoeq::de::CallbackAction::Continue),
        );
        let checkpoint = captured_checkpoint
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .expect("first run saved a complete DE barrier");

        let temporary = tempfile::tempdir().expect("temporary exact-state directory");
        for (identity_field, expected_error) in [
            ("build", "executable build mismatch"),
            ("source", "solver source mismatch"),
        ] {
            let mut stale_checkpoint = checkpoint.clone();
            if identity_field == "build" {
                stale_checkpoint.build_identity.push_str("-stale");
            } else {
                stale_checkpoint.solver_source_identity.push_str("-stale");
            }
            let state = autoeq::workflow::exact_resume::ExactOptimizerState::from_checkpoint(
                stale_checkpoint,
                "same-run-identity",
            )
            .expect("outer state keeps the same run identity");
            let path = temporary.path().join(format!("{identity_field}.json"));
            autoeq::workflow::exact_resume::save_exact_optimizer_state(&state, &path)
                .expect("persist stale math identity fixture");
            let loaded = autoeq::workflow::exact_resume::load_exact_optimizer_state(&path)
                .expect("load persisted exact state")
                .expect("exact state exists");
            loaded
                .check_compatible("same-run-identity")
                .expect("outer measurement/config identity matches");

            let requested_objective = test_objective_data();
            let save_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let save_called_by_callback = std::sync::Arc::clone(&save_called);
            let error = match perform_optimization_with_exact_checkpoint(
                &params,
                &requested_objective,
                autoeq::optim::setup::ExactDECheckpointOptions {
                    checkpoint: Some(loaded.checkpoint),
                    run_identity: "same-run-identity".to_owned(),
                    save_callback: Box::new(move |_| {
                        save_called_by_callback.store(true, std::sync::atomic::Ordering::SeqCst);
                        Ok(())
                    }),
                },
            ) {
                Ok(_) => panic!("stale math identity must reject exact resume"),
                Err(error) => error,
            };
            assert!(error.to_string().contains(expected_error), "{error}");
            assert!(
                requested_objective.prepared.get().is_none(),
                "math checkpoint rejection must happen before objective scoring"
            );
            assert!(
                !save_called.load(std::sync::atomic::Ordering::SeqCst),
                "math checkpoint rejection must happen before persisting another barrier"
            );
        }
    }

    #[test]
    fn exact_resume_rejects_refinement_before_scoring_or_checkpoint_save() {
        let mut params = test_params(false);
        params.algo = "autoeq:de".to_owned();
        params.population = 8;
        params.maxeval = 1_000;
        params.seed = Some(12_345);
        params.no_parallel = true;
        params.parallel_threads = 1;
        params.refine = true;
        let objective = test_objective_data();
        let save_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let save_called_by_callback = std::sync::Arc::clone(&save_called);

        let error = match perform_optimization_with_exact_checkpoint(
            &params,
            &objective,
            autoeq::optim::setup::ExactDECheckpointOptions {
                checkpoint: None,
                run_identity: "exact-run".to_owned(),
                save_callback: Box::new(move |_| {
                    save_called_by_callback.store(true, std::sync::atomic::Ordering::SeqCst);
                    Ok(())
                }),
            },
        ) {
            Ok(_) => panic!("exact continuation must reject local refinement"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("local-refinement"));
        assert!(
            objective.prepared.get().is_none(),
            "refinement rejection must happen before objective scoring"
        );
        assert!(
            !save_called.load(std::sync::atomic::Ordering::SeqCst),
            "refinement rejection must happen before checkpoint persistence"
        );
    }
}
