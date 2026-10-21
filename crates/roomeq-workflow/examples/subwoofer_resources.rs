//! One workload per process, with allocation counts and explicit acoustic outcomes.

// Rust guideline compliant 2026-02-21
#[path = "../benches/fixtures/subwoofers.rs"]
mod fixtures;

use anyhow::{Result, ensure};
use roomeq_model::{CorrectionDecision, OptimizerConfig};
use serde_json::{Value, json};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::{alloc::System, num::NonZeroUsize, time::Instant};

#[global_allocator]
static ALLOCATOR: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

struct Trace(std::path::PathBuf);

impl roomeq_workflow::pipeline::FinalizationDiagnosticSink for Trace {
    fn write_event(&self, name: &str, json: &[u8]) -> std::io::Result<()> {
        let mut file = tempfile::NamedTempFile::new_in(&self.0)?;
        std::io::Write::write_all(&mut file, json)?;
        file.as_file().sync_all()?;
        file.persist_noclobber(self.0.join(format!("{name}.json")))
            .map_err(|error| error.error)?;
        Ok(())
    }
}

fn measure<T>(operation: impl FnOnce() -> Result<T>) -> Result<(T, Value)> {
    // Fixtures and report serialization are excluded; lazy initialization and
    // all worker allocations during the operation are intentionally included.
    let region = Region::new(ALLOCATOR);
    let start = Instant::now();
    let result = operation()?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let stats = region.change();
    Ok((
        result,
        json!({
            "wall_ms": elapsed_ms,
            "allocations": stats.allocations,
            "deallocations": stats.deallocations,
            "reallocations": stats.reallocations,
            "bytes_allocated": stats.bytes_allocated,
            "bytes_deallocated": stats.bytes_deallocated,
            "scope": "operation including lazy initialization; excludes fixture construction, assertions and JSON reporting"
        }),
    ))
}

fn joint(subs: usize, seats: usize, bins: usize) -> Result<Value> {
    let request = fixtures::joint_request(subs, seats, bins);
    let config = OptimizerConfig {
        min_freq: 20.0,
        max_freq: 160.0,
        min_db: -3.0,
        max_db: 3.0,
        algorithm: "autoeq:de".into(),
        seed: Some(7),
        max_iter: 100,
        population: 20,
        tolerance: 0.0,
        atolerance: 0.0,
        parallel_threads: Some(1),
        joint_sub_evaluation_limit: NonZeroUsize::new(500),
        ..Default::default()
    };
    let (result, resources) = measure(|| {
        Ok(roomeq_engine::multisub::optimize_joint_sub_array(
            &request, &config, 48_000.0,
        )?)
    })?;
    ensure!(
        result.search.evaluation_count == 500,
        "joint search did not spend its declared quota"
    );
    ensure!(
        result.per_seat_post.len() == seats && result.physical_outputs == request.physical_outputs,
        "joint output/seat identity changed"
    );
    ensure!(
        result
            .output_views
            .iter()
            .all(|view| view.delta_db.is_finite()),
        "nonfinite output level"
    );
    Ok(json!({
        "resources": resources, "subs": subs, "seats": seats, "bins": bins,
        "sample_rate_hz": 48000,
        "search": {"evaluations": result.search.evaluation_count, "status": result.search.message,
            "elapsed_ms": result.search.elapsed_ms, "converged": result.converged},
        "quality": {"scope": "array only; excludes shared EQ, routing and finalization",
            "array_rejection_reason": result.array_rejection_reason,
            "objective_before": result.pre_components.total, "objective_after": result.post_components.total,
            "seat_output_delta_db": result.output_views.iter().map(|view| view.delta_db).collect::<Vec<_>>()}
    }))
}

fn cinema(
    subs: usize,
    seats: usize,
    fs: f64,
    accepted: bool,
    dba: bool,
    fir_mode: Option<(roomeq_model::ProcessingMode, usize)>,
) -> Result<Value> {
    let mut config = if accepted {
        fixtures::correctable_home_cinema(fs)
    } else {
        fixtures::home_cinema_joint(subs, seats)
    };
    if let Some((mode, taps)) = &fir_mode {
        config.optimizer.processing_mode = mode.clone();
        config.optimizer.fir = Some(roomeq_model::FirConfig {
            taps: *taps,
            phase: "linear".into(),
            ..Default::default()
        });
        config.optimizer.mixed_config =
            (mode == &roomeq_model::ProcessingMode::Hybrid).then(|| {
                roomeq_model::MixedModeConfig {
                    crossover_freq: 300.0,
                    fir_band: "high".into(),
                    ..Default::default()
                }
            });
    }
    if dba {
        config.optimizer.multi_seat = None;
        config.speakers.insert(
            "array".into(),
            roomeq_model::SpeakerConfig::Dba(roomeq_model::DBAConfig {
                name: "array".into(),
                speaker_name: None,
                front: vec![fixtures::stationary_source(1); 2],
                rear: vec![fixtures::stationary_source(1); 3],
            }),
        );
        config
            .system
            .as_mut()
            .unwrap()
            .subwoofers
            .as_mut()
            .unwrap()
            .config = roomeq_model::SubwooferStrategy::Dba;
    }
    let output = tempfile::tempdir()?;
    let trace = std::env::var_os("ROOMEQ_RESOURCE_TRACE")
        .map(|root| {
            let directory =
                std::path::PathBuf::from(root).join(format!("run-{}", std::process::id()));
            std::fs::create_dir_all(&directory)?;
            Ok::<_, std::io::Error>(Trace(directory))
        })
        .transpose()?;
    let (result, resources) = measure(|| {
        let pipeline = roomeq_workflow::pipeline::RoomPipeline::new(
            roomeq_workflow::pipeline::RoomPipelineRequest {
                config: &config,
                sample_rate: fs,
                output_dir: Some(output.path()),
                probe_arrival_overrides: None,
            },
        );
        Ok(if let Some(trace) = &trace {
            pipeline
                .with_finalization_diagnostic_sink(
                    roomeq_workflow::pipeline::FinalizationDiagnosticTrial::FullStrengthOutput,
                    trace,
                )
                .run(None)?
        } else {
            pipeline.run(None)?
        })
    })?;
    let mut fir_artifacts = Vec::new();
    if let Some(path) = std::env::var_os("ROOMEQ_RESOURCE_DIAGNOSTICS") {
        serde_json::to_writer(std::fs::File::create(path)?, &result.to_dsp_chain_output())?;
    }
    if let Some((_, taps)) = fir_mode {
        if config.optimizer.processing_mode == roomeq_model::ProcessingMode::Hybrid
            && config.optimizer.mixed_config.is_some()
        {
            ensure!(
                ["L", "R"].iter().all(|name| result.channels[*name]
                    .plugins
                    .iter()
                    .any(|plugin| plugin.plugin_type == "band_split")
                    && result.channels[*name]
                        .plugins
                        .iter()
                        .any(|plugin| plugin.plugin_type == "band_merge")),
                "requested split-band hybrid processing was not retained"
            );
        }
        ensure!(
            result.channels.values().any(|chain| chain
                .plugins
                .iter()
                .any(|plugin| plugin.plugin_type == "convolution")
                || chain.drivers.iter().flatten().any(|driver| driver
                    .plugins
                    .iter()
                    .any(|plugin| plugin.plugin_type == "convolution"))),
            "FIR workflow skipped or reverted its convolution stage; acceptance={:?}; stages={:?}",
            result
                .metadata
                .correction_acceptance
                .as_ref()
                .map(|report| (&report.decision, &report.violations)),
            result
                .metadata
                .stage_outcomes
                .iter()
                .map(|stage| (&stage.stage, &stage.status, &stage.advisories))
                .collect::<Vec<_>>()
        );
        for entry in std::fs::read_dir(output.path())? {
            let path = entry?.path();
            if path.extension().is_some_and(|extension| extension == "wav") {
                let reader = hound::WavReader::open(&path)?;
                fir_artifacts.push(json!({"file": path.file_name().unwrap().to_string_lossy(), "taps": reader.duration()}));
            }
        }
        ensure!(
            fir_artifacts
                .iter()
                .any(|artifact| artifact["taps"] == taps),
            "requested long FIR was not emitted"
        );
    }
    let bass = result
        .metadata
        .bass_management
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing bass routing"))?;
    let expected: Vec<_> = (1..=subs).map(|index| format!("Sub{index}")).collect();
    ensure!(
        bass.physical_sub_outputs == expected,
        "physical output identity changed"
    );
    let searches: Vec<_> = result
        .channels
        .values()
        .filter_map(|chain| chain.joint_sub.as_ref())
        .collect();
    if !dba {
        ensure!(
            searches
                .iter()
                .any(|search| search.search_evaluations > 0 && search.seats.len() == seats),
            "intended joint multi-seat search did not run"
        );
    } else {
        ensure!(
            bass.sub_outputs
                .iter()
                .filter(|output| output.strategy_source == "dba_front")
                .count()
                == 2
                && bass
                    .sub_outputs
                    .iter()
                    .filter(|output| output.strategy_source == "dba_rear")
                    .count()
                    == 3,
            "DBA controls lost physical branches"
        );
    }
    let acceptance = result
        .metadata
        .correction_acceptance
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing acceptance"))?;
    let quality = acceptance.acoustic_quality.as_ref();
    if accepted {
        ensure!(
            acceptance.accepted && acceptance.decision == CorrectionDecision::Accepted,
            "known-solvable correction was not delivered: {:?}",
            acceptance.decision
        );
        let quality = quality.ok_or_else(|| anyhow::anyhow!("missing acoustic evidence"))?;
        let training: Vec<_> = quality
            .final_seats
            .iter()
            .filter(|seat| seat.partition == "training")
            .collect();
        ensure!(
            !training.is_empty()
                && training.iter().any(|seat| seat.logical_input == "LFE")
                && training
                    .iter()
                    .all(|seat| seat.improvement_lower_bound_db > 0.0),
            "known-solvable fixture did not improve every training input/seat"
        );
        ensure!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .any(|stage| stage.stage == "final_correlated_bass_acceptance"
                    && !stage.checks.is_empty()
                    && stage.checks.iter().all(|check| check.passed)),
            "accepted fixture has no passing correlated-bass assessment"
        );
    }
    let graph = result.to_dsp_chain_output();
    let replay: roomeq_model::DspGraph = serde_json::from_slice(&serde_json::to_vec(&graph)?)?;
    replay.validate().map_err(anyhow::Error::msg)?;
    let serialized = serde_json::to_value(&replay)?;
    ensure!(
        serialized["metadata"]["bass_management"]["routing_graph"]["physical_sub_outputs"]
            == json!(expected),
        "native JSON dropped physical outputs"
    );
    let has_split = replay.channels.values().any(|chain| {
        chain
            .plugins
            .iter()
            .any(|plugin| plugin.plugin_type == "band_split")
    });
    let camilla =
        roomeq_export::render_dsp_graph(&replay, roomeq_export::ExportFormat::CamillaDsp, fs);
    let export = if has_split {
        let error = camilla
            .expect_err("the CamillaDSP capability contract rejects split-band chains")
            .to_string();
        ensure!(
            error.contains("does not support") && error.contains("band_split"),
            "unexpected export refusal: {error}"
        );
        json!({"native_json_validated": true, "camilladsp_supported": false, "camilladsp_limitation": error})
    } else {
        let yaml = camilla?;
        ensure!(
            expected
                .iter()
                .all(|output| yaml.contains(&format!("_to_{output}"))),
            "export dropped physical outputs"
        );
        json!({"native_json_validated": true, "camilladsp_supported": true})
    };
    Ok(json!({
        "resources": resources, "subs": subs, "seats": seats, "sample_rate_hz": fs,
        "requires_accepted_correction": accepted, "fir_artifacts": fir_artifacts, "export": export,
        "search": searches.iter().map(|search| json!({"evaluations": search.search_evaluations,
            "elapsed_ms": search.search_elapsed_ms, "status": search.search_status})).collect::<Vec<_>>(),
        "quality": {"accepted": acceptance.accepted, "decision": acceptance.decision,
            "violations": acceptance.violations, "acoustic": quality,
            "stages": result.metadata.stage_outcomes, "physical_outputs": bass.physical_sub_outputs}
    }))
}

fn fir(taps: usize, fs: f64, phase: &str) -> Result<Value> {
    let frequency = ndarray::Array1::logspace(10.0, 20.0_f64.log10(), 20_000.0_f64.log10(), 512);
    let measurement = roomeq_model::Curve {
        freq: frequency.clone(),
        spl: frequency.mapv(|f| 80.0 + 6.0 * (-0.5 * ((f / 65.0).log2() / 0.2).powi(2)).exp()),
        ..Default::default()
    };
    let target = roomeq_model::Curve {
        freq: frequency,
        spl: ndarray::Array1::from_elem(512, 80.0),
        ..Default::default()
    };
    let config = OptimizerConfig {
        min_freq: 20.0,
        max_freq: 20_000.0,
        fir: Some(roomeq_model::FirConfig {
            taps,
            phase: phase.into(),
            ..Default::default()
        }),
        ..Default::default()
    };
    let (coefficients, resources) = measure(|| {
        roomeq_engine::fir::generate_fir_correction_prepared(&measurement, &config, &target, fs)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
    })?;
    ensure!(
        coefficients.len() == taps
            && coefficients.iter().all(|value| value.is_finite())
            && coefficients
                .iter()
                .filter(|value| value.abs() > 1e-8)
                .count()
                > 1,
        "invalid or identity FIR"
    );
    Ok(
        json!({"resources": resources, "taps": taps, "sample_rate_hz": fs, "phase": phase,
        "quality": {"scope": "FIR design only; no delivered room-correction claim", "finite_nonidentity": true}}),
    )
}

fn main() -> Result<()> {
    let workload = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("expected workload name"))?;
    let mut result = match workload.as_str() {
        "joint_2_1_128" => joint(2, 1, 128)?,
        "joint_4_4_512" => joint(4, 4, 512)?,
        "joint_8_8_1024" => joint(8, 8, 1024)?,
        "cinema_2_2_44100" => cinema(2, 2, 44_100.0, false, false, None)?,
        "cinema_4_3_48000" => cinema(4, 3, 48_000.0, false, false, None)?,
        "cinema_8_4_96000" => cinema(8, 4, 96_000.0, false, false, None)?,
        "accepted_cinema_48000" => cinema(2, 2, 48_000.0, true, false, None)?,
        "dba_2_3_48000" => cinema(5, 1, 48_000.0, false, true, None)?,
        "hybrid_cinema_8192_48000" => cinema(
            2,
            2,
            48_000.0,
            true,
            false,
            Some((roomeq_model::ProcessingMode::Hybrid, 8192)),
        )?,
        "fir_cinema_16384_96000" => cinema(
            2,
            2,
            96_000.0,
            true,
            false,
            Some((roomeq_model::ProcessingMode::PhaseLinear, 16384)),
        )?,
        "fir_minimum_8192_44100" => fir(8192, 44_100.0, "minimum")?,
        "fir_linear_8192_48000" => fir(8192, 48_000.0, "linear")?,
        "fir_minimum_16384_96000" => fir(16384, 96_000.0, "minimum")?,
        _ => anyhow::bail!("unknown workload {workload}"),
    };
    result["workload"] = json!(workload);
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
