//! RoomEQ Synthetic QA: Tests optimization against synthetic speaker scenarios.
//!
//! Uses deterministic synthetic curves with known room modes and noise to validate
//! that optimization consistently improves the response across all processing modes,
//! targets, and option combinations.
//!
//! Usage:
//!   cargo run --bin roomeq-qa-synthetic --no-default-features --release
//!   cargo run --bin roomeq-qa-synthetic --no-default-features --release -- --list
//!   cargo run --bin roomeq-qa-synthetic --no-default-features --release -- --difficulty easy
//!   cargo run --bin roomeq-qa-synthetic --no-default-features --release -- --multiseat-guards-only

use anyhow::Result;
use math_audio_iir_fir::{Biquad, BiquadFilterType};
use roomeq_model::{Curve, ProcessingMode};
use roomeq_synthetic::{
    generate_flat_curve, generate_harman_tilt_curve, generate_multisub_scenario, generate_scenario,
    generate_speaker_rolloff_curve,
};
use std::fmt::Write as _;
use std::time::Instant;

mod build;
mod channel_layout;
mod consts;
mod decision;
mod generate;
mod misc;
mod option;
mod run;
mod types;

use crate::parameter_matrix::generate_pr_matrix;
use channel_layout::sub_topos_for_layout;
use consts::ALL_DIFFICULTIES;
use consts::ALL_LAYOUTS;
use consts::ALL_MS_DIFFICULTIES;
use consts::ALL_SUB_TOPOS;
use consts::KAUTZ_REFERENCE_MODES;
use consts::MS_OPTIONS;
use consts::MS_TOPOLOGIES;
use consts::OPTIONS;
use consts::SAMPLE_RATE;
use consts::SEED;
use generate::generate_ms_option_combos;
use generate::generate_option_combos;
use misc::fmt_epa;
use run::multiseat_api_guard_test_count;
use run::report_multiseat_api_guard_tests;
use run::run_multichannel_test;
use run::run_multiseat_api_guard_tests;
use run::run_multisub_test;
use run::run_single_test;
use types::DifficultyLevel;
use types::MultiSubDifficulty;
use types::QaOutcome;
use types::TestResult;

fn multichannel_mode_supported(
    layout: &channel_layout::ChannelLayout,
    mode: &ProcessingMode,
) -> bool {
    // Kautz correction is validated for individual speakers and no-LFE
    // multichannel systems. Independently corrected mains and subs cannot yet
    // be safely recombined by the routed-bass workflow.
    !(layout.has_lfe && *mode == ProcessingMode::KautzModal)
}

/// Run the synthetic QA command and report whether the binary should exit
/// unsuccessfully because one or more scenarios failed.
/// Execute the bounded PR covering array against real synthetic optimizations.
/// Render the staged listening-stimulus set plus manifest (Stage 2).
///
/// Speech and music are not synthesized: programme material enters staged
/// corpora as external hash-pinned files (see
/// `roomeq_quality::StimulusKind::ExternalProgramme`), never as
/// generated audio. Levels stay clear of the click crest-factor clip
/// point; production corpora choose levels per stimulus class.
pub fn run_stimuli(outdir: Option<&str>) -> Result<bool> {
    use roomeq_quality::{
        CONVERSION_AFFINE_FS_SPL_V1, SplCalibration, StimulusKind, StimulusRequest,
        render_stimulus_set,
    };
    let dir = outdir.unwrap_or("target/qa/stimuli");
    let requests = [
        (
            StimulusKind::Tone {
                freq_hz: 1000.0,
                duration_s: 0.5,
            },
            1u64,
        ),
        (
            StimulusKind::ToneBurst {
                freq_hz: 440.0,
                duration_s: 0.3,
                ramp_ms: 5.0,
            },
            2,
        ),
        (
            StimulusKind::Sweep {
                f0_hz: 100.0,
                f1_hz: 8000.0,
                duration_s: 1.0,
                log: true,
            },
            3,
        ),
        (StimulusKind::TransientClick { duration_s: 0.2 }, 4),
        (
            StimulusKind::ShapedNoise {
                pink: false,
                duration_s: 1.0,
            },
            5,
        ),
        (
            StimulusKind::ShapedNoise {
                pink: true,
                duration_s: 1.0,
            },
            6,
        ),
        (
            StimulusKind::BandLimitedNoise {
                low_hz: 500.0,
                high_hz: 1500.0,
                duration_s: 1.0,
            },
            7,
        ),
        (
            StimulusKind::HarmonicComplex {
                f0_hz: 220.0,
                n_harmonics: 8,
                tilt_db_per_octave: 6.0,
                duration_s: 1.0,
            },
            8,
        ),
        (
            StimulusKind::MaskerProbe {
                masker_freq_hz: 1000.0,
                probe_freq_hz: 1000.0,
                masker_duration_s: 0.2,
                gap_ms: 20.0,
                probe_duration_s: 0.1,
            },
            9,
        ),
    ]
    .into_iter()
    .map(|(kind, seed)| StimulusRequest { kind, seed })
    .collect::<Vec<_>>();
    let calibration = SplCalibration {
        conversion: String::from(CONVERSION_AFFINE_FS_SPL_V1),
        db_spl_at_0dbfs_rms: 90.0,
        levels_db: vec![55.0, 65.0],
    };
    let manifest =
        render_stimulus_set(std::path::Path::new(dir), &requests, &calibration, 48_000.0)
            .map_err(|error| anyhow::anyhow!("stimulus render failed: {error}"))?;
    println!(
        "stimuli: {} files + manifest in {dir} (renderer {})",
        manifest.files.len(),
        manifest.renderer
    );
    Ok(false)
}

/// Exercise the Stage 3 acceptance policies on synthetic fixtures.
///
/// Not listening evidence: a deterministic plumbing demo showing the four
/// policy modules (inversion support, band splits, chain constraints,
/// final validation) accept/reject the expected synthetic cases.
/// Returns `true` when every demo expectation holds.
pub fn run_stage3_policies() -> Result<QaRunOutcome> {
    use roomeq_quality::{
        AggregationOrder, BandSplitPolicy, BassPolicy, BoundedInversionPolicy, ChainConstraints,
        ChainEvidence, FinalReferences, InversionEvidence, InversionVerdict, NullClassification,
        PercentileDomain, SeatMetricDefinition, SubMainSumEvidence, TemporalGate,
        TemporalGateBasis, WindowDiagnostics, assess_inversion, evaluate_chain_constraints,
        evaluate_final_validation,
    };
    let mut checks = 0usize;
    let mut passed = 0usize;
    let mut expect = |label: &str, holds: bool| {
        checks += 1;
        if holds {
            passed += 1;
        } else {
            println!("stage3 demo FAIL: {label}");
        }
    };

    // 1. Inversion support: ideal null boosts, cancellation never does.
    let policy = BoundedInversionPolicy {
        max_boost_db: 9.0,
        min_confidence: 0.8,
        min_depth_scale: 0.5,
        min_support_bins: 8,
        min_seat_agreement: 2,
    };
    let ideal = InversionEvidence {
        null_depth_db: 12.0,
        classification: NullClassification::MinimumPhase,
        classification_confidence: 0.95,
        measurement_depth_scale: 1.0,
        supporting_seats: 4,
        supporting_bins: 24,
        requested_boost_db: 6.0,
    };
    let decision = assess_inversion(&ideal, &policy)
        .map_err(|error| anyhow::anyhow!("inversion demo: {error}"))?;
    expect(
        "ideal null allowed",
        decision.verdict == InversionVerdict::Allowed,
    );
    let cancellation = InversionEvidence {
        classification: NullClassification::NonMinimumPhase,
        ..ideal
    };
    let decision = assess_inversion(&cancellation, &policy)
        .map_err(|error| anyhow::anyhow!("inversion demo: {error}"))?;
    expect(
        "cancellation cuts-only",
        decision.verdict == InversionVerdict::CutsOnly,
    );
    let absent = InversionEvidence {
        supporting_bins: 0,
        ..ideal
    };
    let decision = assess_inversion(&absent, &policy)
        .map_err(|error| anyhow::anyhow!("inversion demo: {error}"))?;
    expect(
        "absent evidence refused",
        decision.verdict == InversionVerdict::Refuse,
    );

    // 2. Band split: opt-in, smooth, cuts-only bass; diagnostics advisory.
    let split = BandSplitPolicy::disabled();
    expect("disabled split validates", split.validate().is_ok());
    let mut cuts_only = BandSplitPolicy {
        enabled: true,
        bass: BassPolicy::CutsOnly,
        ..BandSplitPolicy::disabled()
    };
    expect(
        "cuts-only denies boost",
        cuts_only.authorize_bass_correction(6.0).is_err(),
    );
    expect(
        "cuts-only keeps cuts",
        cuts_only.authorize_bass_correction(-4.0).is_ok(),
    );
    cuts_only.width_octaves = 0.0;
    expect("hard boundary refused", cuts_only.validate().is_err());
    let diagnostics = WindowDiagnostics {
        early_direct_delta_db: Some(18.0),
        late_delta_db: None,
        induced_group_delay_rms_ms: Some(40.0),
        timing_trusted: false,
    };
    expect(
        "diagnostics advisory only",
        diagnostics
            .advisory_notes()
            .iter()
            .all(|note| note.starts_with("advisory:")),
    );

    // 3. Chain constraints + temporal gate with stated basis.
    let constraints = ChainConstraints {
        max_peak_gain_db: 12.0,
        min_headroom_db: -12.0,
        max_latency_ms: 100.0,
        export_sample_rates_hz: vec![44_100, 48_000],
    };
    let evidence = ChainEvidence {
        peak_gain_db: Some(6.0),
        headroom_db: Some(-3.0),
        latency_ms: Some(20.0),
        export_sample_rate_hz: Some(48_000),
    };
    let violations = evaluate_chain_constraints(&evidence, &constraints)
        .map_err(|error| anyhow::anyhow!("chain demo: {error}"))?;
    expect("clean chain passes", violations.is_empty());
    let gate = TemporalGate {
        name: String::from("induced-group-delay-rms"),
        limit: 10.0,
        basis: TemporalGateBasis::EngineeringLimit {
            rationale: String::from("hybrid output-class latency budget"),
        },
        timing_trusted: true,
    };
    expect(
        "trusted excess fails enforced",
        matches!(
            gate.apply(Some(12.0), "ms")
                .map_err(|error| anyhow::anyhow!("gate demo: {error}"))?,
            roomeq_quality::GateOutcome::Fail { .. }
        ),
    );
    let untrusted = TemporalGate {
        timing_trusted: false,
        ..gate
    };
    expect(
        "untrusted excess advisory",
        matches!(
            untrusted
                .apply(Some(12.0), "ms")
                .map_err(|error| anyhow::anyhow!("gate demo: {error}"))?,
            roomeq_quality::GateOutcome::Advisory { .. }
        ),
    );

    // 4. Final validation: candidate beats the identity baseline.
    let grid: ndarray::Array1<f64> = (0..64)
        .map(|index| 20.0 * (1000.0_f64).powf(index as f64 / 63.0))
        .collect();
    let flat = |offset_db: f64| Curve {
        freq: grid.clone(),
        spl: ndarray::Array1::from_elem(64, offset_db),
        ..Default::default()
    };
    let target = flat(80.0);
    let pre = vec![flat(84.0), flat(84.0)];
    let candidate = vec![flat(82.0), flat(83.0)];
    let definition = SeatMetricDefinition {
        support_band_hz: [50.0, 16_000.0],
        weighting: String::from("erb-rate"),
        measure_version: String::from("auditory-frequency-measure-v1"),
        percentile: 0.95,
        percentile_domain: PercentileDomain::Bins,
        aggregation_order: AggregationOrder::BinsThenSeats,
        uncertainty: String::from("seeded-bootstrap-ci95"),
        uncertainty_resamples: 200,
        uncertainty_seed: 7,
        max_residual_percentile_db: 6.0,
        permitted_degradation_db: 0.5,
        min_aggregate_gain_db: 1.0,
        max_pruning_loss_db: 0.5,
        min_support_bins: 8,
    };
    let references = FinalReferences {
        baseline_post: pre.clone(),
        full_chain_post: None,
    };
    let sub_main = SubMainSumEvidence {
        sub: flat(80.0),
        main: flat(80.0),
        summed: flat(86.0),
        crossover_band_hz: [60.0, 120.0],
        max_cancellation_db: 3.0,
    };
    let report = evaluate_final_validation(
        &definition,
        &pre,
        &candidate,
        &target,
        &references,
        &sub_main,
    )
    .map_err(|error| anyhow::anyhow!("final demo: {error}"))?;
    expect("better candidate accepted", report.accepted());
    expect("worst seat is seat-1", report.worst_seat.seat == "seat-1");
    println!(
        "stage3 policies: {passed}/{checks} demo expectations hold (aggregate gain {:.2} dB, worst {})",
        report.aggregate_gain_db, report.worst_seat.seat
    );
    Ok(QaRunOutcome::from_counts(passed, checks))
}

/// Exercise the Stage 4 rerank pipeline on synthetic fixtures.
///
/// Plumbing demo, not proof of improvement: builds a bounded shortlist
/// (optimizer, Pareto, identity), reranks it with a staged-metric
/// evaluator under budgets and a transform cache, refines the winner
/// without switching losses, and compares against EPA/flat baselines on
/// held-out data. Returns `true` when every demo expectation holds.
pub fn run_stage4_rerank() -> Result<QaRunOutcome> {
    use autoeq_optim::rerank::{
        AuditoryEvaluator, BaselineEntry, BudgetLedger, EvaluatorBasis, GridResolution, LossPin,
        NominationSource, RerankCache, ShortlistCandidate, StageBudgets, build_shortlist,
        check_final_resolution, compare_to_baselines, record_refinement, rerank,
    };
    let mut checks = 0usize;
    let mut passed = 0usize;
    let mut expect = |label: &str, holds: bool| {
        checks += 1;
        if holds {
            passed += 1;
        } else {
            println!("stage4 demo FAIL: {label}");
        }
    };

    let pin = LossPin {
        loss: String::from("speaker-flat"),
        version: String::from("v3"),
    };
    let nominate = |id: &str, value: f64, source: NominationSource| ShortlistCandidate {
        id: String::from(id),
        params: vec![value],
        fast_value: value,
        source,
        loss_pin: pin.clone(),
        seed: Some(11),
    };
    let shortlist = build_shortlist(
        vec![
            nominate("opt-a", 2.0, NominationSource::OptimizerRun),
            nominate("pareto-b", 3.0, NominationSource::ParetoFront),
            nominate("identity", 9.0, NominationSource::Identity),
        ],
        8,
        true,
    )
    .map_err(|error| anyhow::anyhow!("shortlist demo: {error}"))?;
    expect("shortlist holds three", shortlist.candidates.len() == 3);

    let evaluator = AuditoryEvaluator {
        name: String::from("final-validation-metric"),
        model_version: String::from("roomeq-quality-final-v1"),
        basis: EvaluatorBasis::StagedMetric {
            metric: String::from("seat-aggregate-gain"),
        },
    };
    let transform = RerankCache::hash_transforms(b"measurement-v1");
    let mut cache = RerankCache::default();
    let mut ledger = BudgetLedger::default().with_budgets(StageBudgets {
        max_evaluations: 10,
        wall_time_ms: 60_000,
        max_memory_bytes: u64::MAX,
    });
    let report = rerank(
        &shortlist,
        &evaluator,
        &pin,
        &transform,
        &mut cache,
        &mut ledger,
        |candidate| {
            Ok(match candidate.id.as_str() {
                "identity" => 1.0,
                "opt-a" => 2.5,
                _ => 4.0,
            })
        },
    )
    .map_err(|error| anyhow::anyhow!("rerank demo: {error}"))?;
    let winner = &report.ranked[0];
    expect("identity wins the demo rerank", winner.id == "identity");
    expect("three evaluations consumed", ledger.evaluations == 3);

    let refined = record_refinement(&winner.id, &pin, &pin, 5, vec![0.0], 1.0)
        .map_err(|error| anyhow::anyhow!("refine demo: {error}"))?;
    expect("refinement carries the pin", refined.loss_pin == pin);

    expect(
        "validated final grid accepted",
        check_final_resolution(200, 400, GridResolution::Validated, 256).is_ok(),
    );

    let baselines = vec![
        BaselineEntry {
            name: String::from("epa"),
            held_out_value: 3.0,
            held_out_id: String::from("held-out-rooms"),
        },
        BaselineEntry {
            name: String::from("speaker-flat"),
            held_out_value: 4.0,
            held_out_id: String::from("held-out-rooms"),
        },
    ];
    let comparison =
        compare_to_baselines(&winner.id, winner.evaluator_score, &baselines, 0.5, None)
            .map_err(|error| anyhow::anyhow!("baseline demo: {error}"))?;
    // Identity scores 1.0 vs EPA 3.0: adopted on held-out data (demo
    // plumbing — a steering fixture, not proof of improvement).
    expect(
        "demo winner adopted on held-out",
        comparison.verdict == autoeq_optim::rerank::ComparisonVerdict::AdoptCandidate,
    );
    println!(
        "stage4 rerank: {passed}/{checks} demo expectations hold (winner {}, {} evaluations)",
        winner.id, ledger.evaluations
    );
    Ok(QaRunOutcome::from_counts(passed, checks))
}

/// Exercise the Stage 5 release gates on synthetic records.
///
/// Conformance demo, not a release decision: evaluates promotion rules
/// for legacy, advisory, enforcing physical, and perceptual-claim policy
/// records against synthetic gate assessments. Returns `true` when every
/// demo expectation holds.
pub fn run_release_gates() -> Result<QaRunOutcome> {
    use crate::release_gates::{
        GateAssessment, PolicyBehavior, PolicyRelease, ReleaseGate, veto_policy_release,
    };
    use roomeq_model::FilterAudibilityConfig;
    let mut checks = 0usize;
    let mut passed = 0usize;
    let mut expect = |label: &str, holds: bool| {
        checks += 1;
        if holds {
            passed += 1;
        } else {
            println!("release-gates demo FAIL: {label}");
        }
    };
    let gate = |gate: ReleaseGate, passed: bool| GateAssessment {
        gate,
        passed,
        evidence: String::from("demo-evidence"),
    };
    let physical_only = vec![
        gate(ReleaseGate::ImplementationCorrectness, true),
        gate(ReleaseGate::PhysicalSafety, true),
    ];
    // Physical safeguard promotes without any Stage 4 evidence.
    let safeguard = PolicyRelease {
        policy_id: String::from("headroom-guard"),
        version: String::from("1.0.0"),
        behavior: PolicyBehavior::Enforcing,
        perceptual_claim: false,
    };
    let promotion = safeguard
        .promotion(&physical_only, false)
        .map_err(|error| anyhow::anyhow!("gates demo: {error}"))?;
    expect("safeguard promotes on two gates", promotion.promoted());
    // Default veto config is advisory: never promotes, even with full gates.
    let default = FilterAudibilityConfig::default();
    let veto = veto_policy_release(Some(&default), "phase-a-1.0");
    let full = vec![
        gate(ReleaseGate::ImplementationCorrectness, true),
        gate(ReleaseGate::PhysicalSafety, true),
        gate(ReleaseGate::PerceptualValidation, true),
        gate(ReleaseGate::ListeningBenefit, true),
    ];
    let promotion = veto
        .promotion(&full, false)
        .map_err(|error| anyhow::anyhow!("gates demo: {error}"))?;
    expect("advisory veto never promotes", !promotion.promoted());
    // No selection: legacy promotes unconditionally.
    let legacy = veto_policy_release(None, "phase-a-1.0");
    let promotion = legacy
        .promotion(&[], true)
        .map_err(|error| anyhow::anyhow!("gates demo: {error}"))?;
    expect("legacy rollback never blocked", promotion.promoted());
    println!("release gates: {passed}/{checks} demo expectations hold");
    Ok(QaRunOutcome::from_counts(passed, checks))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QaRunOutcome {
    Passed,
    Failed,
}

impl QaRunOutcome {
    fn from_counts(passed: usize, total: usize) -> Self {
        if total > 0 && passed == total {
            Self::Passed
        } else {
            Self::Failed
        }
    }

    pub fn has_failures(self) -> bool {
        self == Self::Failed
    }
}

#[cfg(test)]
mod outcome_tests {
    use super::{QaRunOutcome, write_parameter_matrix_artifact};

    #[test]
    fn parameter_matrix_bundles_are_distinct_and_retain_measurement_inputs() {
        let root = tempfile::tempdir().unwrap();
        let first = super::create_parameter_matrix_bundle(root.path(), 0).unwrap();
        let second = super::create_parameter_matrix_bundle(root.path(), 0).unwrap();
        assert_ne!(first, second);
        assert!(first.is_dir() && second.is_dir());
        let rows = super::generate_pr_matrix();
        let curve = super::generate_flat_curve(20.0, 20_000.0, 100);
        let config = super::build::build_parameter_config(
            &curve, &rows[0], roomeq_model::ProcessingMode::Hybrid, 96_000.0,
        );
        let path = first.join("request.json");
        write_parameter_matrix_artifact(&path, &super::parameter_matrix_request(&config).unwrap()).unwrap();
        let saved: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&path).unwrap(),
        ).unwrap();
        let mut recovered: roomeq_model::RoomConfig = serde_json::from_value(
            saved["configuration_without_speakers"].clone(),
        ).unwrap();
        let measurements: std::collections::BTreeMap<String, roomeq_model::Curve> =
            serde_json::from_value(saved["single_speaker_measurements"].clone()).unwrap();
        for (name, curve) in measurements {
            recovered.speakers.insert(name, roomeq_model::SpeakerConfig::Single(
                roomeq_model::MeasurementSource::InMemory(curve),
            ));
        }
        assert_eq!(super::parameter_matrix_request(&recovered).unwrap(), saved);
        assert!(!second.join("request.json").exists());
    }

    #[test]
    #[ignore = "focused full five-seed matrix row; run explicitly during outcome audit"]
    fn hybrid_short_lfe_parameter_row_retains_requested_fir() {
        let result = run_parameter_row_fixture(10);
        let lfe = &result.channel_results["LFE"];
        assert!(lfe.post_score < lfe.pre_score);
        assert!(result.combined_post_score < result.combined_pre_score);
        assert_eq!(lfe.fir_coeffs.as_ref().map(Vec::len), Some(480));
    }

    #[test]
    #[ignore = "focused full five-seed matrix row; run explicitly during outcome audit"]
    fn phase_linear_routed_row_preserves_correction_scores_after_report_refresh() {
        let result = run_parameter_row_fixture(1);
        for (name, channel) in &result.channel_results {
            assert!(channel.post_score < channel.pre_score,
                "{name}: report refresh must retain correction-only scores: {} -> {}",
                channel.pre_score, channel.post_score);
            assert_eq!(channel.fir_coeffs.as_ref().map(Vec::len), Some(960));
        }
        assert!(result.combined_post_score < result.combined_pre_score);
    }

    fn run_parameter_row_fixture(index: usize) -> roomeq_engine::room_result::RoomOptimizationResult {
        let rows = super::generate_pr_matrix();
        let row = &rows[index];
        let sample_rate = [44_100.0, 48_000.0, 96_000.0][row.sample_rate as usize];
        let curve = super::generate_flat_curve(
            20.0,
            (sample_rate / 2.0 - 100.0_f64).max(1_000.0_f64),
            [100, 200, 400][row.grid_size as usize],
        );
        let mut config = super::build::build_parameter_config(
            &curve, row, match row.mode {
                0 => roomeq_model::ProcessingMode::LowLatency,
                1 => roomeq_model::ProcessingMode::PhaseLinear,
                _ => roomeq_model::ProcessingMode::Hybrid,
            }, sample_rate,
        );
        config.optimizer.num_filters = [3, 7, 11][row.filter_count as usize];
        config.optimizer.max_freq = (sample_rate / 2.0 - 100.0_f64).min(config.optimizer.max_freq);
        config.optimizer.max_iter = 120;
        config.optimizer.seed = Some(super::SEED + index as u64);
        super::run::run_optimization_at_rate(&config, sample_rate).unwrap()
    }

    #[test]
    fn parameter_matrix_checkpoints_preserve_completed_rows_and_final_format() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("matrix.json");
        let rows = serde_json::json!([{"row": 0, "post_score": 1.25}]);
        let checkpoint = serde_json::json!({
            "status": "running", "expected_rows": 16, "completed_rows": rows,
        });
        write_parameter_matrix_artifact(&path, &checkpoint).unwrap();
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved, checkpoint);
        // A partially serialized replacement must not destroy the last checkpoint.
        let invalid = std::collections::BTreeMap::from([((1, 2), 3)]);
        assert!(write_parameter_matrix_artifact(&path, &invalid).is_err());
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved, checkpoint);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
        write_parameter_matrix_artifact(&path, &rows).unwrap();
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(
            saved, rows,
            "completed artifacts retain the existing array format"
        );
    }

    #[test]
    fn parameter_matrix_artifact_write_failure_is_fatal() {
        let directory = tempfile::tempdir().unwrap();
        let obstruction = directory.path().join("not_a_directory");
        std::fs::write(&obstruction, b"keep").unwrap();
        assert!(
            write_parameter_matrix_artifact(
                &obstruction.join("matrix.json"),
                &serde_json::json!([]),
            )
            .is_err()
        );
        assert_eq!(std::fs::read(&obstruction).unwrap(), b"keep");
    }

    #[test]
    fn qa_outcomes_use_the_launcher_failure_convention() {
        assert!(!QaRunOutcome::from_counts(16, 16).has_failures());
        assert!(QaRunOutcome::from_counts(15, 16).has_failures());
        assert!(QaRunOutcome::from_counts(0, 0).has_failures());
    }
}

fn write_parameter_matrix_artifact(
    path: &std::path::Path,
    value: &impl serde::Serialize,
) -> Result<()> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut pending = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut pending, value)?;
    pending.as_file().sync_all()?;
    pending.persist(path)?;
    Ok(())
}

fn create_parameter_matrix_bundle(root: &std::path::Path, row: usize) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(root)?;
    Ok(tempfile::Builder::new()
        .prefix(&format!("row-{row:02}-"))
        .tempdir_in(root)?
        .keep())
}

fn parameter_matrix_request(config: &roomeq_model::RoomConfig) -> Result<serde_json::Value> {
    let mut configuration = config.clone();
    configuration.speakers.clear();
    let mut measurements = std::collections::BTreeMap::new();
    for (name, speaker) in &config.speakers {
        match speaker {
            roomeq_model::SpeakerConfig::Single(roomeq_model::MeasurementSource::InMemory(curve)) => {
                measurements.insert(name, curve);
            }
            _ => anyhow::bail!("matrix replay bundle requires an explicit single in-memory measurement for {name}"),
        }
    }
    Ok(serde_json::json!({
        "format": "roomeq_parameter_request_v1",
        "configuration_without_speakers": configuration,
        "single_speaker_measurements": measurements,
    }))
}

fn parameter_crossover_execution(topology: usize, crossover: usize, phase: usize,
    bass: &serde_json::Value) -> Result<&'static str> {
    if topology == 0 {
        anyhow::ensure!(bass.is_null(), "unrouted row unexpectedly has bass management");
        return Ok("not_applicable");
    }
    anyhow::ensure!(bass["enabled"] == true, "routed row did not enable bass management");
    let expected_type = if crossover == 2 { "LR48" } else { "LR24" };
    anyhow::ensure!(bass["optimization"]["crossover_type"] == expected_type,
        "crossover type did not reach runtime");
    anyhow::ensure!(bass["routing_graph"]["routes"].as_array().is_some_and(|routes| !routes.is_empty()),
        "crossover produced no routes");
    if crossover == 1 && phase != 1 {
        anyhow::ensure!(bass["optimization"]["applied"] == false,
            "phase-free automatic crossover unexpectedly reported applied");
        return Ok("unsupported_missing_phase");
    }
    if crossover == 1 {
        anyhow::ensure!(bass["optimization"]["applied"] == true,
            "automatic crossover selection did not execute");
        Ok("automatic_selection_applied")
    } else { Ok("fixed_routes_emitted") }
}

#[cfg(test)]
mod crossover_execution_tests {
    use super::parameter_crossover_execution;
    use serde_json::json;

    #[test]
    fn crossover_status_distinguishes_execution_support_and_applicability() {
        let mut bass = json!({"enabled": true,
            "optimization": {"crossover_type": "LR24", "applied": true},
            "routing_graph": {"routes": [{}]}});
        assert_eq!(parameter_crossover_execution(0, 1, 0, &json!(null)).unwrap(), "not_applicable");
        assert_eq!(parameter_crossover_execution(1, 0, 1, &bass).unwrap(), "fixed_routes_emitted");
        assert_eq!(parameter_crossover_execution(1, 1, 1, &bass).unwrap(), "automatic_selection_applied");
        bass["optimization"]["applied"] = json!(false);
        assert_eq!(parameter_crossover_execution(1, 1, 0, &bass).unwrap(), "unsupported_missing_phase");
        assert!(parameter_crossover_execution(1, 1, 1, &bass).is_err());
        bass["routing_graph"]["routes"] = json!([]);
        assert!(parameter_crossover_execution(1, 0, 1, &bass).is_err());
    }
}

pub fn run_parameter_matrix() -> Result<QaRunOutcome> {
    let rows = generate_pr_matrix();
    let artifact = std::path::Path::new("target/qa/roomeq-parameter-matrix.json");
    if let Some(parent) = artifact.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_parameter_matrix_artifact(
        artifact,
        &serde_json::json!({
            "status": "running", "expected_rows": rows.len(), "completed_rows": []
        }),
    )?;
    let mut passed = 0usize;
    let mut records = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let sample_rate = [44_100.0, 48_000.0, 96_000.0][row.sample_rate as usize];
        let point_count = [100, 200, 400][row.grid_size as usize];
        let mode = match row.mode {
            0 => ProcessingMode::LowLatency,
            1 => ProcessingMode::PhaseLinear,
            _ => ProcessingMode::Hybrid,
        };
        let mode_name = format!("{mode:?}");
        let curve = generate_flat_curve(
            20.0,
            (sample_rate / 2.0 - 100.0_f64).max(1_000.0_f64),
            point_count,
        );
        let mut config = build::build_parameter_config(&curve, row, mode, sample_rate);
        config.optimizer.num_filters = [3, 7, 11][row.filter_count as usize];
        config.optimizer.max_freq = (sample_rate / 2.0 - 100.0_f64).min(config.optimizer.max_freq);
        config.optimizer.max_iter = 120;
        config.optimizer.seed = Some(SEED + index as u64);
        // Preserve the selected rerun's sidecars for independent backend replay.
        // Unique directories prevent a later run from overwriting old evidence.
        let bundle = create_parameter_matrix_bundle(
            std::path::Path::new("target/qa/roomeq-parameter-bundles"), index,
        )?;
        write_parameter_matrix_artifact(&bundle.join("request.json"), &serde_json::json!({
            "row": index, "requested_axes": row, "sample_rate_hz": sample_rate,
            "inputs": parameter_matrix_request(&config)?,
        }))?;
        let result = match crate::optimize_room(&config, sample_rate, Some(&bundle)) {
            Ok(result) => result,
            Err(error) => {
                write_parameter_matrix_artifact(
                    artifact,
                    &serde_json::json!({
                        "status": "failed", "expected_rows": rows.len(),
                        "failed_row": index, "requested_axes": row, "replay_bundle": bundle,
                        "sample_rate_hz": sample_rate, "optimizer": config.optimizer,
                        "error": format!("{error:#}"), "completed_rows": records,
                    }),
                )?;
                return Err(error);
            }
        };
        if !result.combined_post_score.is_finite() {
            write_parameter_matrix_artifact(
                artifact,
                &serde_json::json!({
                    "status": "failed", "expected_rows": rows.len(),
                    "failed_row": index, "requested_axes": row, "replay_bundle": bundle,
                    "sample_rate_hz": sample_rate, "optimizer": config.optimizer,
                    "error": "non-finite post score",
                    "non_finite_post_score": result.combined_post_score.to_string(),
                    "completed_rows": records,
                }),
            )?;
            anyhow::bail!("pairwise row {index} produced a non-finite score");
        }
        write_parameter_matrix_artifact(
            &bundle.join("selected-output.json"), &result.to_dsp_chain_output(),
        )?;
        let crossover_execution = match parameter_crossover_execution(
            row.topology as usize, row.crossover as usize, row.phase as usize,
            &serde_json::to_value(&result.metadata.bass_management)?) {
            Ok(status) => status,
            Err(error) => {
                write_parameter_matrix_artifact(artifact, &serde_json::json!({
                    "status": "failed", "expected_rows": rows.len(), "failed_row": index,
                    "requested_axes": row, "replay_bundle": bundle,
                    "sample_rate_hz": sample_rate, "error": format!("{error:#}"),
                    "completed_rows": records,
                }))?;
                return Err(error);
            }
        };
        records.push(serde_json::json!({
            "row": index,
            "replay_bundle": {
                "directory": bundle,
                "request": "request.json",
                "selected_output": "selected-output.json",
                "scope": "selected_rerun_dsp_and_sidecars_not_backend_certification",
            },
            "requested_axes": row,
            "crossover_execution": crossover_execution,
            "unexecuted_axes": if crossover_execution == "unsupported_missing_phase" {
                vec!["crossover"]
            } else { Vec::<&str>::new() },
            "not_applicable_axes": if crossover_execution == "not_applicable" {
                vec!["crossover"]
            } else { Vec::<&str>::new() },
            "outcome_scope": "finite_output_smoke",
            "effective_config": {
                "optimizer": config.optimizer,
                "system": config.system,
                "crossovers": config.crossovers,
                "speaker_names": ({
                    let mut names: Vec<_> = config.speakers.keys().collect();
                    names.sort();
                    names
                }),
                "measurement_source": "in_memory_parameter_fixture",
                "measurement_profile": "4db_log_gaussian_300hz_sigma_0.7",
                "measurements": config.speakers.iter().filter_map(|(name, speaker)| {
                    if let roomeq_model::SpeakerConfig::Single(roomeq_model::MeasurementSource::InMemory(curve)) = speaker {
                        Some((name.clone(), serde_json::json!({
                            "band_hz": [curve.freq[0], curve.freq[curve.freq.len()-1]],
                            "grid_points": curve.freq.len(), "has_phase": curve.phase.is_some(),
                        })))
                    } else { None }
                }).collect::<std::collections::BTreeMap<_, _>>(),
            },
            "sample_rate_hz": sample_rate,
            "delivered_biquad_sample_rates_hz": result.channel_results.iter().map(|(name, channel)|
                (name.clone(), channel.biquads.iter().map(|filter| filter.srate).collect::<Vec<_>>())
            ).collect::<std::collections::BTreeMap<_, _>>(),
            "requested_grid_points": point_count,
            "mode": mode_name,
            "filter_count": config.optimizer.num_filters,
            "requested_fir_duration_ms": ([5, 10, 20][row.fir_duration as usize]),
            "fir_duration_applicable": config.optimizer.processing_mode != ProcessingMode::LowLatency,
            "delivered_fir_taps": result.channel_results.iter().map(|(name, channel)|
                (name.clone(), channel.fir_coeffs.as_ref().map(Vec::len))
            ).collect::<std::collections::BTreeMap<_, _>>(),
            "pre_score": result.combined_pre_score,
            "post_score": result.combined_post_score,
            "stage_outcomes": result.metadata.stage_outcomes,
            "selected_acceptance": result.metadata.correction_acceptance,
            "selected_channel_scores": result.channel_results.iter().map(|(name, channel)| {
                (name.clone(), serde_json::json!({"pre": channel.pre_score, "post": channel.post_score}))
            }).collect::<std::collections::BTreeMap<_, _>>(),
            "home_cinema_layout": result.metadata.home_cinema_layout,
            "bass_management": result.metadata.bass_management,
            "seed_distribution": result.metadata.qa_seed_distribution,
        }));
        write_parameter_matrix_artifact(
            artifact,
            &serde_json::json!({
                "status": "running", "expected_rows": rows.len(),
                "completed_rows": records,
            }),
        )?;
        passed += 1;
    }
    let artifact = std::path::Path::new("target/qa/roomeq-parameter-matrix.json");
    if let Some(parent) = artifact.parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_parameter_matrix_artifact(artifact, &records)?;
    println!(
        "parameter matrix: {passed}/{} finite-output smoke rows passed (full axis execution is not yet established)",
        rows.len()
    );
    Ok(QaRunOutcome::from_counts(passed, rows.len()))
}

pub fn run() -> Result<bool> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let registry = crate::registry::load_registry()?;
    let suite = registry
        .suite_for_runner("synthetic")
        .ok_or_else(|| anyhow::anyhow!("RoomEQ QA registry has no synthetic suite"))?;
    for required_claim in ["pairwise_options", "topology_mode_cartesian", "multi_seed"] {
        anyhow::ensure!(
            suite.claims.iter().any(|claim| claim == required_claim),
            "synthetic registry suite is missing claim '{required_claim}'"
        );
    }
    let registered_options = suite.cases.iter().map(String::as_str).collect::<Vec<_>>();
    let implemented_options = OPTIONS.iter().map(|option| option.name).collect::<Vec<_>>();
    anyhow::ensure!(
        registered_options == implemented_options,
        "synthetic option axes drifted from the RoomEQ QA registry: registry={registered_options:?}, implemented={implemented_options:?}"
    );

    let args: Vec<String> = std::env::args().collect();
    let help = args.iter().any(|a| a == "--help" || a == "-h");
    let list_only = args.iter().any(|a| a == "--list");
    let multiseat_guards_only = args.iter().any(|a| a == "--multiseat-guards-only");
    let full_matrix = args.iter().any(|a| a == "--full-matrix");
    let pr_matrix = args.iter().any(|a| a == "--pr");
    if args.iter().any(|a| a == "--parameter-matrix") {
        return run_parameter_matrix().map(QaRunOutcome::has_failures);
    }
    if args.iter().any(|a| a == "--stimuli") {
        return run_stimuli(
            args.windows(2)
                .find(|w| w[0] == "--stimuli-dir")
                .map(|w| w[1].as_str()),
        );
    }
    if args.iter().any(|a| a == "--stage3-policies") {
        return run_stage3_policies().map(QaRunOutcome::has_failures);
    }
    if args.iter().any(|a| a == "--stage4-rerank") {
        return run_stage4_rerank().map(QaRunOutcome::has_failures);
    }
    if args.iter().any(|a| a == "--release-gates") {
        return run_release_gates().map(QaRunOutcome::has_failures);
    }
    let fail_fast = args.iter().any(|a| a == "--fail-fast");
    let difficulty_filter = args
        .windows(2)
        .find(|w| w[0] == "--difficulty")
        .map(|w| w[1].clone());
    let mode_filter = args
        .windows(2)
        .find(|w| w[0] == "--mode")
        .map(|w| w[1].clone());
    let layout_filter = args
        .windows(2)
        .find(|w| w[0] == "--layout")
        .map(|w| w[1].clone());
    let sub_topology_filter = args
        .windows(2)
        .find(|w| w[0] == "--sub-topology")
        .map(|w| w[1].clone());

    if help {
        println!("RoomEQ Synthetic QA");
        println!();
        println!("Usage:");
        println!(
            "  roomeq-qa-synthetic [--list] [--pr] [--difficulty NAME] [--mode NAME] [--layout NAME] [--sub-topology NAME] [--full-matrix] [--multiseat-guards-only]"
        );
        println!();
        println!("Options:");
        println!("  --list                   Print the synthetic QA matrix and exit");
        println!("  --difficulty NAME        Run only one difficulty: easy, medium, hard");
        println!(
            "  --mode NAME              Run one mode: LowLatency, PhaseLinear, Hybrid, MixedPhase, WarpedIir, KautzModal"
        );
        println!("  --layout NAME            Run one multichannel layout, for example 7.1.4");
        println!("  --sub-topology NAME      Run one sub topology, for example mso_8sub");
        println!("  --multiseat-guards-only  Run only multi-seat API guard tests");
        println!(
            "  --full-matrix            Include WarpedIir/KautzModal and every multichannel processing mode"
        );
        println!("  --pr                     Run the bounded pull-request audibility matrix");
        println!(
            "  --parameter-matrix       Run bounded pairwise smoke rows; artifacts identify unexecuted axes"
        );
        println!("  --stimuli [--stimuli-dir DIR]");
        println!(
            "                           Render the staged listening-stimulus set plus manifest (default DIR: target/qa/stimuli)"
        );
        println!(
            "  --stage3-policies        Exercise the Stage 3 acceptance policies on synthetic fixtures (plumbing demo, not listening evidence)"
        );
        println!(
            "  --stage4-rerank          Exercise the Stage 4 shortlist/rerank/refine pipeline on synthetic fixtures (plumbing demo, not proof of improvement)"
        );
        println!(
            "  --release-gates          Exercise the Stage 5 release-gate promotion rules on synthetic records (conformance demo, not a release decision)"
        );
        println!("  --help, -h               Print this help");
        return Ok(false);
    }

    if multiseat_guards_only {
        return report_multiseat_api_guard_tests();
    }

    let difficulties: Vec<&DifficultyLevel> = if pr_matrix {
        vec![&consts::EASY]
    } else if let Some(ref filter) = difficulty_filter {
        ALL_DIFFICULTIES
            .iter()
            .filter(|d| d.name == filter.as_str())
            .collect()
    } else {
        ALL_DIFFICULTIES.iter().collect()
    };

    let ms_difficulties: Vec<&MultiSubDifficulty> = if pr_matrix {
        vec![&consts::MS_EASY]
    } else if let Some(ref filter) = difficulty_filter {
        ALL_MS_DIFFICULTIES
            .iter()
            .filter(|d| d.name == filter.as_str())
            .collect()
    } else {
        ALL_MS_DIFFICULTIES.iter().collect()
    };

    let default_modes = [
        ProcessingMode::LowLatency,
        ProcessingMode::PhaseLinear,
        ProcessingMode::Hybrid,
        ProcessingMode::MixedPhase,
    ];
    let full_modes = [
        ProcessingMode::LowLatency,
        ProcessingMode::PhaseLinear,
        ProcessingMode::Hybrid,
        ProcessingMode::MixedPhase,
        ProcessingMode::WarpedIir,
        ProcessingMode::KautzModal,
    ];
    let pr_modes = [ProcessingMode::LowLatency, ProcessingMode::Hybrid];
    let selected_modes: &[ProcessingMode] = if pr_matrix {
        &pr_modes
    } else if full_matrix {
        &full_modes
    } else {
        &default_modes
    };
    let selected_multichannel_modes: &[ProcessingMode] = if full_matrix {
        &full_modes
    } else {
        &default_modes
    };
    let mode_matches = |mode: &ProcessingMode, filter: &str| {
        let filter = filter.to_ascii_lowercase().replace(['-', '_'], "");
        let name = format!("{mode:?}").to_ascii_lowercase();
        name == filter
    };
    let modes: Vec<ProcessingMode> = selected_modes
        .iter()
        .filter(|mode| {
            mode_filter
                .as_deref()
                .is_none_or(|filter| mode_matches(mode, filter))
        })
        .cloned()
        .collect();
    let multichannel_modes: Vec<ProcessingMode> = selected_multichannel_modes
        .iter()
        .filter(|mode| {
            mode_filter
                .as_deref()
                .is_none_or(|filter| mode_matches(mode, filter))
        })
        .cloned()
        .collect();
    if modes.is_empty() || multichannel_modes.is_empty() {
        anyhow::bail!(
            "mode filter '{}' does not select a mode in this matrix",
            mode_filter.as_deref().unwrap_or_default()
        );
    }

    let flat_target = generate_flat_curve(20.0, 20000.0, 200);
    let harman_target = generate_harman_tilt_curve(20.0, 20000.0, 200);
    let targets: Vec<(&str, &Curve)> = vec![("flat", &flat_target), ("harman", &harman_target)];

    // Speaker rolloff: 0 dB above 80 Hz, -12 dB/oct below (realistic 2nd-order highpass)
    let speaker_rolloff = generate_speaker_rolloff_curve(20.0, 20000.0, 200, 80.0, -12.0);

    let mut option_combos = generate_option_combos();
    let mut ms_option_combos = generate_ms_option_combos();
    if pr_matrix {
        option_combos.truncate(OPTIONS.len() + 1);
        ms_option_combos.truncate(1);
    }
    let layouts: Vec<_> = ALL_LAYOUTS
        .iter()
        .filter(|layout| {
            layout_filter
                .as_ref()
                .is_none_or(|filter| layout.name == filter)
        })
        .filter(|layout| !pr_matrix || matches!(layout.name, "2.0" | "2.1" | "5.1" | "7.1.4"))
        .collect();

    // Count total tests
    let single_total = difficulties.len() * modes.len() * targets.len() * option_combos.len();
    let ms_total = ms_difficulties.len() * MS_TOPOLOGIES.len() * ms_option_combos.len();
    let multiseat_guard_total = multiseat_api_guard_test_count();
    let mc_total: usize = layouts
        .iter()
        .map(|layout| {
            let n_topos = sub_topos_for_layout(layout)
                .iter()
                .filter(|topology| {
                    !pr_matrix
                        || matches!(
                            topology.name,
                            "single_sub" | "mso_2sub" | "cardioid" | "dba"
                        )
                })
                .count();
            let topology_cases = if n_topos == 0 {
                difficulties.len() // no LFE → 1 test per difficulty
            } else {
                n_topos * difficulties.len()
            };
            let supported_modes = multichannel_modes
                .iter()
                .filter(|mode| multichannel_mode_supported(layout, mode))
                .count();
            topology_cases * supported_modes
        })
        .sum();
    let total = single_total + ms_total + multiseat_guard_total + mc_total;

    if list_only {
        println!("Synthetic QA Test Matrix:");
        println!();
        println!("  Single-speaker:");
        println!(
            "    Difficulties: {}",
            difficulties
                .iter()
                .map(|d| d.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Modes: {}",
            modes
                .iter()
                .map(|mode| format!("{mode:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("    Targets: flat, harman");
        println!(
            "    Option combos: {} (baseline + {} singles + {} pairs + 1 all)",
            option_combos.len(),
            OPTIONS.len(),
            OPTIONS.len() * (OPTIONS.len() - 1) / 2,
        );
        println!("    Subtotal: {}", single_total);
        println!();
        println!("  Multi-sub:");
        println!(
            "    Difficulties: {}",
            ms_difficulties
                .iter()
                .map(|d| d.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Topologies: {}",
            MS_TOPOLOGIES
                .iter()
                .map(|t| t.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Option combos: {} (baseline + {} singles)",
            ms_option_combos.len(),
            MS_OPTIONS.len(),
        );
        println!("    Subtotal: {}", ms_total);
        println!();
        println!("  Multi-seat API guards:");
        println!(
            "    Checks: missing phase rejection, MinimizeVariance/Average/PrimaryWithConstraints/ModalBasis metrics, polarity/all-pass controls, registry release-decision matrix"
        );
        println!("    Subtotal: {}", multiseat_guard_total);
        println!();
        println!("  Multi-channel:");
        println!(
            "    Layouts: {}",
            layouts
                .iter()
                .map(|l| l.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Sub topologies (with LFE): {}",
            ALL_SUB_TOPOS
                .iter()
                .map(|t| t.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Difficulties: {}",
            difficulties
                .iter()
                .map(|d| d.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "    Modes: {}",
            multichannel_modes
                .iter()
                .map(|mode| format!("{mode:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        if multichannel_modes.contains(&ProcessingMode::KautzModal) {
            println!("    KautzModal: no-LFE layouts only (routed-bass integration unsupported)");
        }
        println!("    Subtotal: {}", mc_total);
        println!();
        println!("  Total tests: {}", total);
        return Ok(false);
    }

    println!(
        "RoomEQ Synthetic QA -- {} tests ({} single + {} multi-sub + {} multi-seat guards + {} multi-channel)",
        total, single_total, ms_total, multiseat_guard_total, mc_total
    );
    println!("============================================================");

    let start = Instant::now();
    let mut all_results = Vec::new();
    let mut passed = 0;
    let mut failed = 0;

    for difficulty in &difficulties {
        // Build room modes from difficulty config
        let modes_biquad: Vec<Biquad> = difficulty
            .modes
            .iter()
            .map(|&(freq, q, gain)| Biquad::new(BiquadFilterType::Peak, freq, SAMPLE_RATE, q, gain))
            .collect();
        let kautz_modes_biquad: Vec<Biquad> = difficulty
            .modes
            .iter()
            .copied()
            .chain(KAUTZ_REFERENCE_MODES.iter().copied())
            .map(|(freq, q, gain)| Biquad::new(BiquadFilterType::Peak, freq, SAMPLE_RATE, q, gain))
            .collect();

        for &(target_name, target) in &targets {
            // Combine target shape with speaker rolloff so that broadband/excursion
            // options see a realistic low-frequency limit in the measurement.
            let speaker_base = Curve {
                freq: target.freq.clone(),
                spl: &target.spl + &speaker_rolloff.spl,
                phase: None,
                ..Default::default()
            };
            let scenario = generate_scenario(
                &format!("{}/{}", difficulty.name, target_name),
                &speaker_base,
                &modes_biquad,
                difficulty.noise_rms * 0.3,
                difficulty.noise_rms * 0.7,
                SEED,
                SAMPLE_RATE,
            );
            let kautz_scenario = generate_scenario(
                &format!("{}/{}-kautz", difficulty.name, target_name),
                &speaker_base,
                &kautz_modes_biquad,
                difficulty.noise_rms * 0.3,
                difficulty.noise_rms * 0.7,
                SEED,
                SAMPLE_RATE,
            );

            for mode in &modes {
                let degraded = if *mode == ProcessingMode::KautzModal {
                    &kautz_scenario.degraded_curve
                } else {
                    &scenario.degraded_curve
                };
                let mut baseline_post_score = None;
                for combo in &option_combos {
                    let result = run_single_test(
                        degraded,
                        mode.clone(),
                        target_name,
                        combo,
                        difficulty,
                        baseline_post_score,
                    );
                    if combo.is_empty() {
                        baseline_post_score = Some(result.post_score);
                    }

                    if result.passed {
                        passed += 1;
                    } else {
                        failed += 1;
                        println!(
                            "  FAIL: {} -- {} (epa={})",
                            result.name,
                            result.reason,
                            fmt_epa(result.epa_preference)
                        );
                        if fail_fast {
                            return Ok(true);
                        }
                    }

                    all_results.push(result);
                }
            }
        }
    }

    // ====================================================================
    // Multi-sub tests
    // ====================================================================
    for ms_diff in &ms_difficulties {
        let shared_biquads: Vec<Biquad> = ms_diff
            .shared_modes
            .iter()
            .map(|&(f, q, g)| Biquad::new(BiquadFilterType::Peak, f, SAMPLE_RATE, q, g))
            .collect();

        let per_sub_biquads: Vec<Vec<Biquad>> = ms_diff
            .per_sub_modes
            .iter()
            .map(|modes| {
                modes
                    .iter()
                    .map(|&(f, q, g)| Biquad::new(BiquadFilterType::Peak, f, SAMPLE_RATE, q, g))
                    .collect()
            })
            .collect();

        let scenario = generate_multisub_scenario(
            &format!("multisub/{}", ms_diff.name),
            ms_diff.n_subs,
            &shared_biquads,
            &per_sub_biquads,
            ms_diff.delays_ms,
            ms_diff.noise_rms,
            SEED,
            SAMPLE_RATE,
        );

        for topo in MS_TOPOLOGIES {
            for combo in &ms_option_combos {
                let result = run_multisub_test(&scenario.sub_curves, topo, combo, ms_diff);

                if result.passed {
                    passed += 1;
                } else {
                    failed += 1;
                    println!(
                        "  FAIL: {} -- {} (epa={})",
                        result.name,
                        result.reason,
                        fmt_epa(result.epa_preference)
                    );
                    if fail_fast {
                        return Ok(true);
                    }
                }

                all_results.push(result);
            }
        }
    }

    // ====================================================================
    // Multi-seat public API guard tests
    // ====================================================================
    for result in run_multiseat_api_guard_tests() {
        if result.passed {
            passed += 1;
        } else {
            failed += 1;
            println!("  FAIL: {} -- {}", result.name, result.reason);
            if fail_fast {
                return Ok(true);
            }
        }
        all_results.push(result);
    }

    // ====================================================================
    // Multi-channel topology tests
    // ====================================================================
    let base_fullrange = generate_speaker_rolloff_curve(20.0, 20000.0, 200, 80.0, -6.0);

    for layout in layouts {
        let topos: Vec<_> = sub_topos_for_layout(layout)
            .iter()
            .filter(|topology| {
                sub_topology_filter
                    .as_ref()
                    .is_none_or(|filter| topology.name == filter)
            })
            .filter(|topology| {
                !pr_matrix
                    || matches!(
                        topology.name,
                        "single_sub" | "mso_2sub" | "cardioid" | "dba"
                    )
            })
            .collect();

        if topos.is_empty() {
            // No LFE — test mains only
            for difficulty in &difficulties {
                for mode in multichannel_modes
                    .iter()
                    .filter(|mode| multichannel_mode_supported(layout, mode))
                {
                    let result = run_multichannel_test(
                        layout,
                        None,
                        difficulty,
                        &base_fullrange,
                        mode.clone(),
                        SAMPLE_RATE,
                    );
                    if result.passed {
                        passed += 1;
                    } else {
                        failed += 1;
                        println!(
                            "  FAIL: {} -- {} (epa={})",
                            result.name,
                            result.reason,
                            fmt_epa(result.epa_preference)
                        );
                        if fail_fast {
                            return Ok(true);
                        }
                    }
                    all_results.push(result);
                }
            }
        } else {
            // With LFE — test each sub topology
            for sub_topo in topos {
                for difficulty in &difficulties {
                    for mode in multichannel_modes
                        .iter()
                        .filter(|mode| multichannel_mode_supported(layout, mode))
                    {
                        let result = run_multichannel_test(
                            layout,
                            Some(sub_topo),
                            difficulty,
                            &base_fullrange,
                            mode.clone(),
                            SAMPLE_RATE,
                        );
                        if result.passed {
                            passed += 1;
                        } else {
                            failed += 1;
                            println!(
                                "  FAIL: {} -- {} (epa={})",
                                result.name,
                                result.reason,
                                fmt_epa(result.epa_preference)
                            );
                            if fail_fast {
                                return Ok(true);
                            }
                        }
                        all_results.push(result);
                    }
                }
            }
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("============================================================");
    println!(
        "Results: {} passed, {} failed, {} total ({:.1}s)",
        passed,
        failed,
        all_results.len(),
        elapsed.as_secs_f64()
    );

    // Print summary table
    let mut summary = String::new();
    for difficulty in &difficulties {
        let diff_results: Vec<&TestResult> = all_results
            .iter()
            .filter(|r| r.name.starts_with(difficulty.name))
            .collect();
        let diff_pass = diff_results.iter().filter(|r| r.passed).count();
        let diff_total = diff_results.len();
        writeln!(
            &mut summary,
            "  {}: {}/{} passed ({:.1}%)",
            difficulty.name,
            diff_pass,
            diff_total,
            diff_pass as f64 / diff_total as f64 * 100.0
        )
        .ok();
    }
    // Multi-sub summary
    let ms_results: Vec<&TestResult> = all_results
        .iter()
        .filter(|r| r.name.starts_with("multisub/"))
        .collect();
    if !ms_results.is_empty() {
        let ms_pass = ms_results.iter().filter(|r| r.passed).count();
        let ms_total_count = ms_results.len();
        writeln!(
            &mut summary,
            "  multi-sub: {}/{} passed ({:.1}%)",
            ms_pass,
            ms_total_count,
            ms_pass as f64 / ms_total_count as f64 * 100.0
        )
        .ok();
    }

    // Multi-seat API guard summary
    let multiseat_results: Vec<&TestResult> = all_results
        .iter()
        .filter(|r| r.name.starts_with("multiseat/"))
        .collect();
    if !multiseat_results.is_empty() {
        let multiseat_pass = multiseat_results.iter().filter(|r| r.passed).count();
        let multiseat_total_count = multiseat_results.len();
        writeln!(
            &mut summary,
            "  multi-seat API guards: {}/{} passed ({:.1}%)",
            multiseat_pass,
            multiseat_total_count,
            multiseat_pass as f64 / multiseat_total_count as f64 * 100.0
        )
        .ok();
    }

    // Multi-channel summary
    let mc_results: Vec<&TestResult> = all_results
        .iter()
        .filter(|r| r.name.starts_with("multichannel/"))
        .collect();
    if !mc_results.is_empty() {
        let mc_pass = mc_results.iter().filter(|r| r.passed).count();
        let mc_total_count = mc_results.len();
        writeln!(
            &mut summary,
            "  multi-channel: {}/{} passed ({:.1}%)",
            mc_pass,
            mc_total_count,
            mc_pass as f64 / mc_total_count as f64 * 100.0
        )
        .ok();
    }

    println!("\nPer-difficulty summary:");
    print!("{}", summary);

    let passed_outcomes = all_results
        .iter()
        .filter(|result| result.outcome() == QaOutcome::Passed)
        .count();
    let reverted_outcomes = all_results
        .iter()
        .filter(|result| result.outcome() == QaOutcome::Reverted)
        .count();
    let failed_outcomes = all_results
        .iter()
        .filter(|result| result.outcome() == QaOutcome::Failed)
        .count();
    println!(
        "Outcome summary: PASS={passed_outcomes}, REVERTED={reverted_outcomes}, FAIL={failed_outcomes}"
    );

    if failed > 0 {
        println!("\nFailed tests:");
        for r in &all_results {
            if !r.passed {
                println!(
                    "  {} -- {} (epa={})",
                    r.name,
                    r.reason,
                    fmt_epa(r.epa_preference)
                );
            }
        }
    }

    Ok(failed > 0)
}
