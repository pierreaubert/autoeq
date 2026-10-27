//! Compare whole standalone candidates under the requested Hybrid safety policy.

use super::*;
use roomeq_engine::PipelineControl;

pub(super) struct Observers {
    observer: Arc<Mutex<Option<Box<dyn PipelineObserver>>>>,
    stopped: Arc<AtomicBool>,
}

impl Observers {
    pub(super) fn new(observer: Option<Box<dyn PipelineObserver>>) -> Self {
        Self {
            observer: Arc::new(Mutex::new(observer)),
            stopped: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(super) fn stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }

    pub(super) fn observer(&self) -> Option<Box<dyn PipelineObserver>> {
        if self
            .observer
            .lock()
            .expect("observer lock poisoned")
            .is_none()
        {
            return None;
        }
        let observer = Arc::clone(&self.observer);
        let stopped = Arc::clone(&self.stopped);
        Some(Box::new(move |event: &PipelineEvent| {
            if stopped.load(Ordering::Relaxed) {
                return PipelineControl::Stop;
            }
            let control = observer
                .lock()
                .expect("observer lock poisoned")
                .as_mut()
                .map_or(PipelineControl::Continue, |observer| {
                    observer.on_event(event)
                });
            if control == PipelineControl::Stop {
                stopped.store(true, Ordering::Relaxed);
            }
            control
        }))
    }
}

// The tolerance removes numerical ties, not native allowed regression or a
// presumed listener threshold. It matches finalization's deterministic ranking.
const TIE_DB: f64 = 1e-9;

pub(super) fn relocate_resources(
    result: &mut RoomOptimizationResult,
    prefix: &Path,
    output_dir: &Path,
    store: &dyn autoeq_artifacts::ArtifactStore,
    fs: f64,
) -> Result<()> {
    let references =
        roomeq_export::checked_convolution_resource_references(&result.to_dsp_chain_output())
            .map_err(|error| AutoeqError::InvalidMeasurement {
                message: error.to_string(),
            })?;
    let original = result
        .metadata
        .final_convolution_sha256
        .clone()
        .unwrap_or_default();
    let replacements: HashMap<_, _> = references
        .into_iter()
        .map(|reference| {
            let path = Path::new(&reference);
            let relocated = if path.is_absolute() {
                reference.clone()
            } else {
                prefix.join(path).to_string_lossy().into_owned()
            };
            (reference, relocated)
        })
        .collect();
    for chain in result.channels.values_mut() {
        for plugin in chain.plugins.iter_mut().chain(
            chain
                .drivers
                .iter_mut()
                .flatten()
                .flat_map(|driver| driver.plugins.iter_mut()),
        ) {
            if plugin.plugin_type == "convolution" {
                let reference = plugin.parameters["ir_file"].as_str().ok_or_else(|| {
                    AutoeqError::InvalidMeasurement {
                        message: "part FIR reference is not a string".into(),
                    }
                })?;
                let relocated =
                    replacements
                        .get(reference)
                        .ok_or_else(|| AutoeqError::InvalidMeasurement {
                            message: "part FIR resource was not inventoried".into(),
                        })?;
                plugin.parameters["ir_file"] = serde_json::Value::String(relocated.clone());
            }
        }
    }
    crate::export::bind_final_convolution_artifacts(result, output_dir, store, fs)?;
    let rebound = result.metadata.final_convolution_sha256.as_ref().unwrap();
    for (reference, relocated) in &replacements {
        let expected = original
            .get(reference)
            .and_then(Option::as_ref)
            .ok_or_else(|| AutoeqError::InvalidMeasurement {
                message: format!("part FIR '{reference}' was not byte-bound"),
            })?;
        if rebound.get(relocated).and_then(Option::as_ref) != Some(expected) {
            return Err(AutoeqError::InvalidMeasurement {
                message: format!("relocated part FIR '{relocated}' does not match assessed bytes"),
            });
        }
    }
    // The verified byte identity above is a resource-relocation witness, not
    // permission to replace the assessed waveform with a different kernel.
    for decision in &mut result.metadata.provisional_decisions {
        for reference in &mut decision.evidence_refs {
            if let Some(path) = reference.strip_prefix("phase-fir:")
                && let Some(relocated) = replacements.get(path)
            {
                *reference = format!("phase-fir:{relocated}");
            }
        }
    }
    result.metadata.stage_outcomes.push(StageOutcome {
        stage: "hybrid_part_resource_relocation".into(),
        status: StageStatus::Applied,
        advisories: vec![serde_json::json!({"references":replacements,
            "scope":"same assessed WAV bytes and retained coefficients; output-relative resource paths"}).to_string()],
        checks: vec![StageCheck::pass("assessed_fir_bytes_preserved", StageCheckKind::Quality)],
    });
    let processing = crate::final_ledger::processing_snapshot(result)
        .map_err(|message| AutoeqError::OptimizationFailed { message })?;
    crate::final_ledger::finalize_result_ledger(result, &processing, fs)
        .map_err(|message| AutoeqError::OptimizationFailed { message })?;
    Ok(())
}

fn score(result: &RoomOptimizationResult) -> Option<(f64, serde_json::Value)> {
    let report = result.metadata.correction_acceptance.as_ref()?;
    if !matches!(
        report.outcome,
        roomeq_model::RoomEqOutcome::Accepted
            | roomeq_model::RoomEqOutcome::Unchanged
            | roomeq_model::RoomEqOutcome::SafeFallback
    ) {
        return None;
    }
    seat_replay::configured_selection_metrics(report).ok()?;
    let training = &report.acoustic_quality.as_ref()?.training;
    let comparison = training.configured_baseline.as_ref()?;
    let mut basis = comparison.clone();
    let mut post = 0.0;
    for residual in &mut basis.residuals {
        post += residual.score.post_weighted_rms_db?;
        residual.score.post_weighted_rms_db = None;
        residual.score.improvement_db = None;
        // This is an assessed candidate level match, not reference identity.
        residual.score.post_reference_gain_db = 0.0;
        // Delivered membership and support describe the candidate, not its
        // fixed reference. Every candidate already passed native replay.
        residual.delivered_physical_outputs.clear();
        residual.delivered_support.clear();
        // EPA receipts include post preference and the delivered level match.
        // They are candidate assessments, not configured baseline identity.
        residual.level_matched_epa = None;
        residual.epa_unavailable = None;
        residual.epa_spectral_flatness = None;
    }
    Some((
        post / training.curve_count as f64,
        serde_json::to_value(basis).ok()?,
    ))
}

#[cfg(test)]
fn select(
    result: &mut RoomOptimizationResult,
    evaluate: impl FnMut(ProcessingMode) -> Result<RoomOptimizationResult>,
) -> bool {
    select_using_scores(result, evaluate, score)
}

pub(super) fn select_with_reference(
    result: &mut RoomOptimizationResult,
    evaluate: impl FnMut(ProcessingMode) -> Result<RoomOptimizationResult>,
    reference: &seat_replay::ConfiguredComparison<'_>,
) -> bool {
    select_using_scores(result, evaluate, |candidate| {
        // Ranking does not rewrite a candidate's independently assessed native
        // acceptance. Re-evaluate only a clone on one frozen comparison plane.
        let mut ranking = candidate.clone();
        reference.attach(&mut ranking);
        score(&ranking)
    })
}

fn select_using_scores(
    result: &mut RoomOptimizationResult,
    mut evaluate: impl FnMut(ProcessingMode) -> Result<RoomOptimizationResult>,
    mut candidate_score: impl FnMut(&RoomOptimizationResult) -> Option<(f64, serde_json::Value)>,
) -> bool {
    let Some((hybrid_score, basis)) = candidate_score(result) else {
        result.metadata.stage_outcomes.push(StageOutcome {
            stage: "hybrid_best_part_selection".into(),
            status: StageStatus::Degraded,
            advisories: vec!["configured_reference_training_score_unavailable".into()],
            checks: vec![StageCheck {
                id: "hybrid_best_part_reference".into(),
                kind: StageCheckKind::Quality,
                passed: false,
                observed: None,
                limit: None,
                diagnostic: Some("complete configured-reference comparison unavailable".into()),
            }],
        });
        return false;
    };
    // RMS is nonnegative. A graph already at its numeric lower bound cannot
    // lose to either standalone part within the ranking tolerance. This also
    // avoids fitting two redundant parts for an exact identity/flat target.
    if (0.0..=TIE_DB).contains(&hybrid_score) {
        result.metadata.stage_outcomes.push(StageOutcome {
            stage: "hybrid_best_part_selection".into(),
            status: StageStatus::Applied,
            advisories: vec!["selected=hybrid".into(), "certified_nonnegative_rms_lower_bound".into()],
            checks: vec![StageCheck {
                id: "hybrid_best_part_lower_bound".into(),
                kind: StageCheckKind::Quality,
                passed: true,
                observed: Some(hybrid_score),
                limit: Some(TIE_DB),
                diagnostic: Some("all standalone target RMS means are at least zero; no standalone optimization was run".into()),
            }],
        });
        return true;
    }
    let mut complete = true;
    let mut best_score = hybrid_score;
    let mut best = None;
    let mut checks = Vec::new();
    for (name, mode) in [
        ("iir", ProcessingMode::LowLatency),
        ("fir", ProcessingMode::PhaseLinear),
    ] {
        let evaluated = evaluate(mode);
        let mut check =
            StageCheck::pass(format!("hybrid_best_part_{name}"), StageCheckKind::Quality);
        match evaluated {
            Ok(candidate) => match candidate_score(&candidate) {
                Some((post, candidate_basis)) if candidate_basis == basis => {
                    check.observed = Some(hybrid_score - post);
                    check.limit = Some(TIE_DB);
                    check.passed = hybrid_score <= post + TIE_DB;
                    check.diagnostic = Some(serde_json::json!({"score_basis":"configured_reference_training_mean",
                        "hybrid_post_mean_db":hybrid_score, "part_post_mean_db":post,
                        "signed_regression_db":hybrid_score-post, "original_hybrid_safety_budgets":true}).to_string());
                    if post < best_score - TIE_DB {
                        best_score = post;
                        best = Some((name, candidate));
                    }
                }
                Some((_, candidate_basis)) => {
                    complete = false;
                    check.passed = false;
                    check.diagnostic = Some(
                        serde_json::json!({
                            "reason": "standalone configured-reference identity differs",
                            "hybrid_reference": basis,
                            "part_reference": candidate_basis,
                        })
                        .to_string(),
                    );
                }
                None => {
                    complete = false;
                    check.passed = false;
                    check.diagnostic = Some(format!(
                        "standalone configured-reference comparison unavailable: decision={:?}; violations={:?}; comparison_error={:?}",
                        candidate
                            .metadata
                            .correction_acceptance
                            .as_ref()
                            .map(|report| &report.decision),
                        candidate
                            .metadata
                            .correction_acceptance
                            .as_ref()
                            .map(|report| &report.violations),
                        candidate
                            .metadata
                            .correction_acceptance
                            .as_ref()
                            .map(|report| seat_replay::configured_selection_metrics(report).err()),
                    ));
                }
            },
            Err(error) => {
                complete = false;
                check.passed = false;
                check.diagnostic = Some(error.to_string());
            }
        }
        checks.push(check);
    }
    let selected = if let Some((name, candidate)) = best.filter(|_| complete) {
        *result = candidate;
        name
    } else {
        "hybrid"
    };
    result.metadata.stage_outcomes.push(StageOutcome {
        stage: "hybrid_best_part_selection".into(),
        status: if complete {
            StageStatus::Applied
        } else {
            StageStatus::Degraded
        },
        advisories: vec![
            format!("selected={selected}"),
            "score_basis=configured_reference_training_mean".into(),
            "standalone_trials_use_original_hybrid_safety_budgets".into(),
        ],
        checks,
    });
    complete
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a native, complete comparison receipt first; vary only the scalar
    // result of the injected, already-gated candidate evaluator in these tests.
    fn candidate(post: f64, marker: &str) -> RoomOptimizationResult {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let curve = result.channel_results["L"].initial_curve.clone();
        let chain = result.channels.get_mut("L").unwrap();
        chain.initial_curve = Some((&curve).into());
        chain.target_curve = Some((&curve).into());
        let mut config = RoomConfig::default();
        config.speakers.insert(
            "L".into(),
            SpeakerConfig::Single(MeasurementSource::InMemory(curve)),
        );
        let captures = seat_replay::capture_training(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        finalization::rebuild(&mut result, &config, &HashMap::new(), 48000.0, dir.path()).unwrap();
        let baseline = result.clone();
        seat_replay::validate_candidate_final_seats(
            &mut result,
            &baseline,
            &captures,
            &HashMap::new(),
            &config,
            48000.0,
            dir.path(),
        )
        .unwrap();
        seat_replay::attach_configured_comparison(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48000.0,
            dir.path(),
        );
        let report = result.metadata.correction_acceptance.as_mut().unwrap();
        report.outcome = roomeq_model::RoomEqOutcome::Accepted;
        report.accepted = true;
        for row in &mut report
            .acoustic_quality
            .as_mut()
            .unwrap()
            .training
            .configured_baseline
            .as_mut()
            .unwrap()
            .residuals
        {
            row.score.post_weighted_rms_db = Some(post);
            row.score.improvement_db = Some(row.score.pre_weighted_rms_db - post);
        }
        result.metadata.stage_outcomes.push(StageOutcome {
            stage: marker.into(),
            status: StageStatus::Applied,
            advisories: vec![],
            checks: vec![],
        });
        result
    }

    #[test]
    fn candidate_level_match_is_not_configured_reference_identity() {
        let first = candidate(0.5, "first");
        let mut second = first.clone();
        let rows = &mut second
            .metadata
            .correction_acceptance
            .as_mut()
            .unwrap()
            .acoustic_quality
            .as_mut()
            .unwrap()
            .training
            .configured_baseline
            .as_mut()
            .unwrap()
            .residuals;
        for row in rows {
            row.score.post_reference_gain_db += 3.0;
        }
        assert_eq!(score(&first), score(&second));
        let rows = &mut second
            .metadata
            .correction_acceptance
            .as_mut()
            .unwrap()
            .acoustic_quality
            .as_mut()
            .unwrap()
            .training
            .configured_baseline
            .as_mut()
            .unwrap()
            .residuals;
        for row in rows {
            row.score.pre_weighted_rms_db += 1.0;
        }
        assert_ne!(score(&first).unwrap().1, score(&second).unwrap().1);
    }

    #[test]
    fn selects_whole_better_part_and_retains_failed_hybrid_margin() {
        let mut hybrid = candidate(0.5, "hybrid_graph");
        let complete = select(&mut hybrid, |mode| {
            Ok(match mode {
                ProcessingMode::LowLatency => candidate(0.4, "iir_graph"),
                _ => candidate(0.3, "fir_graph"),
            })
        });
        assert!(complete);
        assert!(
            hybrid
                .metadata
                .stage_outcomes
                .iter()
                .any(|s| s.stage == "fir_graph")
        );
        assert!(
            !hybrid
                .metadata
                .stage_outcomes
                .iter()
                .any(|s| s.stage == "hybrid_graph")
        );
        let stage = hybrid.metadata.stage_outcomes.last().unwrap();
        assert!(stage.advisories.contains(&"selected=fir".into()));
        assert!(stage.checks.iter().all(|c| !c.passed));
        assert!((stage.checks[1].observed.unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn missing_part_or_reference_mismatch_cannot_publish_incomplete_selection() {
        for mismatched in [false, true] {
            let mut hybrid = candidate(0.5, "hybrid_graph");
            assert!(!select(&mut hybrid, |mode| {
                if mode == ProcessingMode::LowLatency {
                    return Ok(candidate(0.2, "iir_graph"));
                }
                if !mismatched {
                    return Err(AutoeqError::OptimizationFailed {
                        message: "test safety refusal".into(),
                    });
                }
                let mut part = candidate(0.1, "fir_graph");
                part.metadata
                    .correction_acceptance
                    .as_mut()
                    .unwrap()
                    .acoustic_quality
                    .as_mut()
                    .unwrap()
                    .training
                    .configured_baseline
                    .as_mut()
                    .unwrap()
                    .playback_graph_sha256 = Some("different_reference".into());
                Ok(part)
            }));
            assert!(
                hybrid
                    .metadata
                    .stage_outcomes
                    .iter()
                    .any(|s| s.stage == "hybrid_graph")
            );
            assert_eq!(
                hybrid.metadata.stage_outcomes.last().unwrap().status,
                StageStatus::Degraded
            );
        }
    }

    #[test]
    fn fallback_competes_with_validated_parts_and_numerical_tie_retains_hybrid() {
        let mut fallback = candidate(0.5, "fallback");
        let acceptance = fallback.metadata.correction_acceptance.as_mut().unwrap();
        acceptance.accepted = false;
        acceptance.decision = roomeq_model::CorrectionDecision::SafeFallback;
        acceptance.outcome = roomeq_model::RoomEqOutcome::SafeFallback;
        let mut calls = 0;
        assert!(select(&mut fallback, |_| {
            calls += 1;
            Ok(candidate(0.3, "part_graph"))
        }));
        assert_eq!(calls, 2);
        assert_eq!(
            fallback
                .metadata
                .correction_acceptance
                .as_ref()
                .unwrap()
                .outcome,
            roomeq_model::RoomEqOutcome::Accepted
        );
        assert!(
            fallback
                .metadata
                .stage_outcomes
                .last()
                .unwrap()
                .advisories
                .contains(&"selected=iir".into())
        );
        let mut flat = candidate(0.0, "flat_graph");
        assert!(select(&mut flat, |_| panic!(
            "zero RMS already certifies the part bound"
        )));
        assert!(
            flat.metadata
                .stage_outcomes
                .last()
                .unwrap()
                .advisories
                .contains(&"certified_nonnegative_rms_lower_bound".into())
        );
        let mut hybrid = candidate(0.3, "hybrid_graph");
        assert!(select(&mut hybrid, |_| Ok(candidate(
            0.3 - 0.5e-9,
            "part_graph"
        ))));
        assert!(
            hybrid
                .metadata
                .stage_outcomes
                .last()
                .unwrap()
                .advisories
                .contains(&"selected=hybrid".into())
        );
    }

    #[test]
    fn observer_stop_persists_across_candidate_observers() {
        let observers = Observers::new(Some(Box::new(|_: &PipelineEvent| PipelineControl::Stop)));
        let event = PipelineEvent::new(PipelineStepId::Validation, PipelineStepStatus::Started);
        assert_eq!(
            observers.observer().unwrap().on_event(&event),
            PipelineControl::Stop
        );
        assert!(observers.stopped());
        assert_eq!(
            observers.observer().unwrap().on_event(&event),
            PipelineControl::Stop
        );
    }
}
