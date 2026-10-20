//! Ideal + deformation round-trip QA: reconstruct a known-perfect response.
//!
//! Each case starts from a ground-truth ideal, deforms it with a few
//! parametric peaks plus light noise ([`roomeq_synthetic::roundtrip`]), runs
//! the production [`roomeq_workflow::optimize_room`] path once with a fixed
//! seed (no median-seed search: these cases must be fast and deterministic),
//! and asserts the corrected response matches the ideal in-band.
//!
//! Failing modes this catches (beyond "score improved"):
//! - main/sub role reversal and misrouted bass (2.1 / 5.1),
//! - cross-channel leakage and channel indexing bugs (distinct peaks/channel),
//! - invented corrections on flat inputs (identity case),
//! - joint-sub and seat-aggregation bugs (MSO / multiseat cases).
//!
//! Failure diagnostics (env-gated, off by default): `ROUNDTRIP_STAGES=1`
//! prints stage outcomes, acceptance, published keys, DSP chains, and
//! per-channel biquads; `ROUNDTRIP_SPLICE=1` prints deployed-vs-ideal curves
//! across 40-150 Hz; `ROUNDTRIP_GRADE=1` prints grading band statistics.
//! `ROUNDTRIP_DIAGNOSTICS=<directory>` retains the production finalization trace.

// Rust guideline compliant 2026-02-21

use super::consts::POSITIVE_ROUTED_INPUT_PEAK;
use super::consts::SAMPLE_RATE;
use super::consts::SEED;
use autoeq_core::MeasurementProvenance;
use autoeq_core::ProvenanceCaptureKind;
use roomeq_engine::room_result::RoomOptimizationResult;
use roomeq_model::CrossoverConfig;
use roomeq_model::Curve;
use roomeq_model::InlineMeasurement;
use roomeq_model::MeasurementMultiple;
use roomeq_model::MeasurementRef;
use roomeq_model::MeasurementSingle;
use roomeq_model::MeasurementSource;
use roomeq_model::RoomConfig;
use roomeq_model::SpeakerConfig;
use roomeq_model::SubwooferStrategy;
use roomeq_model::SubwooferSystemConfig;
use roomeq_model::SystemConfig;
use roomeq_model::SystemModel;
use roomeq_model::default_config_version;
use roomeq_synthetic::RoundtripCase;
use std::collections::HashMap;
use std::time::Instant;

/// Fail-fast optimizer budget: tiny population, few thousand evals. Simple
/// few-peak inversions must converge inside this; otherwise the test (not the
/// threshold) deserves scrutiny.
const ROUNDTRIP_MAX_ITER: usize = 8_000;
const ROUNDTRIP_POPULATION: usize = 12;

/// Shared synthetic timing identity. In-memory curves carry no acquisition
/// declaration, so routed stages refuse coherent timing for them ("unknown"
/// capture kind). Round-trip fixtures declare the analytic truth explicitly:
/// one time origin for every channel, stationary synthetic capture.
const ROUNDTRIP_TIMING_REF: &str = "roundtrip-shared-clock";

fn roundtrip_provenance() -> MeasurementProvenance {
    MeasurementProvenance {
        capture_kind: ProvenanceCaptureKind::StationaryIr,
        timing_reference_id: Some(ROUNDTRIP_TIMING_REF.to_string()),
        ..Default::default()
    }
}

fn roundtrip_inline(curve: &Curve, name: &str) -> MeasurementRef {
    MeasurementRef::Inline(InlineMeasurement {
        frequencies: curve.freq.to_vec(),
        magnitude_db: curve.spl.to_vec(),
        phase_deg: curve.phase.as_ref().map(|phase| phase.to_vec()),
        name: Some(name.to_string()),
        wav_path: None,
        csv_path: None,
    })
}

/// Single-measurement source with declared synthetic provenance.
fn roundtrip_source(curve: &Curve, key: &str) -> MeasurementSource {
    // Speaker names allow only alphanumerics, spaces, and hyphens.
    let speaker = format!("roundtrip-{}", key.replace('_', "-"));
    MeasurementSource::Single(MeasurementSingle {
        measurement: roundtrip_inline(curve, key),
        speaker_name: Some(speaker),
        provenance: roundtrip_provenance(),
    })
}

/// Multi-measurement source with declared synthetic provenance.
///
/// Seat names are identical across speakers (`seat-0`, `seat-1`, ...): the
/// routed crossover requires the same physical seat order everywhere.
fn roundtrip_multi_source(curves: &[Curve], key: &str) -> MeasurementSource {
    MeasurementSource::Multiple(MeasurementMultiple {
        measurements: curves
            .iter()
            .enumerate()
            .map(|(seat, curve)| roundtrip_inline(curve, &format!("seat-{seat}")))
            .collect(),
        speaker_name: Some(format!("roundtrip-{}", key.replace('_', "-"))),
        provenance: roundtrip_provenance(),
        estimated_position: None,
    })
}

/// Sub topology for the round-trip config builder.
pub(super) enum RoundtripSubs {
    /// No subwoofer (2.0 / identity layouts).
    None,
    /// Single LFE branch fed by the case's `"lfe"` channel.
    Single,
    /// Production MSO topology: one speaker key per physical sub
    /// (`lfe_1`, `lfe_2`, ...) exactly like the FEM MSO corpus, each with
    /// multi-seat measurements. Single-measurement subs collapse the
    /// coherent combined curve to a phaseless magnitude, which the routed
    /// replay refuses; production MSO always measures multiple seats.
    SplitMso {
        /// Per-sub per-seat deformed measurements, one speaker key per sub.
        curves: Vec<Vec<Curve>>,
    },
}

/// Multi-measurement (multi-seat) input: per-channel seat curves replace the
/// single deformed measurement via `InMemoryMultiple`.
pub(super) struct RoundtripSeats {
    /// Measurement key -> per-seat deformed curves.
    pub(super) seats: HashMap<String, Vec<Curve>>,
    pub(super) strategy: roomeq_model::MultiMeasurementStrategy,
}

/// Build a production-shaped [`RoomConfig`] from deformed round-trip curves.
///
/// Mirrors `build_multichannel_config` (v3 sub outputs, per-output LR24@80 Hz
/// crossovers, LFE removed from logical speakers) with a fail-fast optimizer.
pub(super) fn build_roundtrip_config(
    case: &RoundtripCase,
    mains: &[(&str, &str)],
    subs: RoundtripSubs,
    seats: Option<RoundtripSeats>,
) -> RoomConfig {
    let mut speakers = HashMap::new();
    let mut sys_speakers = HashMap::new();

    for (key, role) in mains {
        let channel = case
            .get(key)
            .unwrap_or_else(|| panic!("round-trip case {} has no channel {key}", case.name));
        let source = match &seats {
            Some(seat_set) => seat_set
                .seats
                .get(*key)
                .map(|curves| roundtrip_multi_source(curves, key)),
            None => None,
        }
        .unwrap_or_else(|| roundtrip_source(&channel.deformed, key));
        speakers.insert(key.to_string(), SpeakerConfig::Single(source));
        sys_speakers.insert((*role).to_string(), (*key).to_string());
    }

    let mut sub_config: Option<SubwooferSystemConfig> = None;
    match &subs {
        RoundtripSubs::None => {}
        RoundtripSubs::Single => {
            let channel = case.get("lfe").expect("round-trip case has no lfe channel");
            speakers.insert(
                "lfe".to_string(),
                SpeakerConfig::Single(roundtrip_source(&channel.deformed, "lfe")),
            );
            sub_config = Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Single,
                crossover: None,
                routing: Default::default(),
                outputs: Vec::new(),
            });
        }
        RoundtripSubs::SplitMso { curves } => {
            for (index, seats) in curves.iter().enumerate() {
                let key = format!("lfe_{}", index + 1);
                speakers.insert(
                    key.clone(),
                    SpeakerConfig::Single(roundtrip_multi_source(seats, &key)),
                );
            }
            sub_config = Some(SubwooferSystemConfig {
                config: SubwooferStrategy::Mso,
                crossover: None,
                routing: Default::default(),
                outputs: Vec::new(),
            });
        }
    }

    // Physical outputs: (output id, speaker key) per topology.
    let sub_branches: Vec<(String, String)> = match &subs {
        RoundtripSubs::None => Vec::new(),
        RoundtripSubs::Single => vec![("Sub1".to_string(), "lfe".to_string())],
        RoundtripSubs::SplitMso { curves } => curves
            .iter()
            .enumerate()
            .map(|(index, _)| (format!("Sub{}", index + 1), format!("lfe_{}", index + 1)))
            .collect(),
    };

    let mut crossovers_map = None;
    if let Some(ref mut subs) = sub_config {
        subs.outputs = sub_branches
            .iter()
            .map(|(id, speaker)| roomeq_model::SubwooferOutput {
                id: id.clone(),
                speaker: speaker.clone(),
            })
            .collect();
        subs.crossover = Some(roomeq_model::SubwooferCrossoverRef::PerSub(
            sub_branches
                .iter()
                .map(|_| String::from("lfe_xover"))
                .collect(),
        ));
        let mut xovers = HashMap::new();
        xovers.insert(
            "lfe_xover".to_string(),
            CrossoverConfig {
                crossover_type: "LR24".to_string(),
                frequency: Some(80.0),
                frequencies: None,
                frequency_range: None,
            },
        );
        crossovers_map = Some(xovers);
    }

    let model = if mains.len() <= 2
        && !sub_config
            .as_ref()
            .is_some_and(|subs| subs.outputs.len() > 2)
    {
        SystemModel::Stereo
    } else {
        SystemModel::HomeCinema
    };
    let system = SystemConfig {
        model,
        speakers: sys_speakers,
        subwoofers: sub_config,
        bass_management: None,
        supporting_source_outputs: None,
    };

    let mut config = RoomConfig {
        version: default_config_version(),
        system: Some(system),
        speakers,
        crossovers: crossovers_map,
        target_curve: None,
        optimizer: Default::default(),
        provenance: Default::default(),
        recording_config: None,
        measured_impulse_responses: Default::default(),
        ctc: None,
        reporting: None,
        cea2034_cache: None,
    };

    // Differential evolution: on few-peak multimodal landscapes it inverts
    // reliably inside the fail-fast budget, while CMA-ES stagnates in local
    // optima (half-gain corrections, missed peaks). CMA-ES stays covered by
    // the existing synthetic/quality QA matrix.
    config.optimizer.algorithm = "autoeq:de".to_string();
    config.optimizer.max_iter = ROUNDTRIP_MAX_ITER;
    config.optimizer.population = ROUNDTRIP_POPULATION;
    config.optimizer.refine = false;
    config.optimizer.seed = Some(SEED);
    // Symmetric objective: the default asymmetric loss deliberately
    // under-corrects dips (peaks 2x, bass peaks 5x vs dips 1x). Round-trip
    // cases assert full reconstruction of the ideal, so voicing policy is
    // disabled here; it stays covered by the option-effect QA matrix.
    config.optimizer.asymmetric_loss = false;
    config.optimizer.processing_mode = roomeq_model::ProcessingMode::LowLatency;
    config.optimizer.num_filters = case.num_filters;
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20_000.0;
    // Positive inverse fixtures need reserve to restore their injected dips
    // in stereo as well as routed systems. Retain the ordinary output ceiling
    // and attenuation limits while declaring the same bounded input stimulus.
    config.optimizer.finalization.default_input_peak = POSITIVE_ROUTED_INPUT_PEAK;
    if let Some(seat_set) = seats {
        config.optimizer.multi_measurement = Some(roomeq_model::MultiMeasurementConfig {
            strategy: seat_set.strategy,
            ..Default::default()
        });
        // The fixture builds every speaker's seats in the same physical
        // order (`seat-0`, `seat-1`, ...); declare that identity explicitly
        // so the routed coherent join uses the configured-identity path
        // instead of requiring a verified fixed-projection capture receipt
        // that in-memory synthetic curves cannot carry.
        let seat_count = seat_set.seats.values().next().map_or(0, Vec::len);
        config.optimizer.multi_seat = Some(roomeq_model::MultiSeatConfig {
            seat_identity: Some(roomeq_model::SeatIdentityMap {
                ids: (0..seat_count).map(|seat| format!("seat-{seat}")).collect(),
            }),
            ..Default::default()
        });
    }

    config
}

/// One direct workflow run with a fixed seed (no median-seed search).
fn run_once(config: &RoomConfig) -> anyhow::Result<RoomOptimizationResult> {
    let id = super::consts::TEMP_DIR_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temp_dir =
        std::env::temp_dir().join(format!("roomeq_qa_roundtrip_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&temp_dir)?;
    let result = if let Some(root) = std::env::var_os("ROUNDTRIP_DIAGNOSTICS") {
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
        let directory =
            std::path::PathBuf::from(root).join(format!("run-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&directory)?;
        let trace = Trace(directory);
        roomeq_workflow::pipeline::RoomPipeline::new(
            roomeq_workflow::pipeline::RoomPipelineRequest {
                config,
                sample_rate: SAMPLE_RATE,
                output_dir: Some(&temp_dir),
                probe_arrival_overrides: None,
            },
        )
        .with_finalization_diagnostic_sink(
            roomeq_workflow::pipeline::FinalizationDiagnosticTrial::ZeroStrengthOutput,
            &trace,
        )
        .run(None)
    } else {
        roomeq_workflow::optimize_room(config, SAMPLE_RATE, None, Some(&temp_dir))
    }
    .map_err(|error| anyhow::anyhow!(error.to_string()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    result
}

/// Where the corrected curve for a channel came from (diagnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CurveSource {
    Deployed,
    ChannelResults,
    Channels,
}

fn final_curve_for(
    result: &RoomOptimizationResult,
    key: &str,
    role: &str,
) -> Option<(CurveSource, Curve)> {
    // Physical sub outputs publish as Sub1..SubN, never as the "lfe" config
    // keys ("lfe" -> Sub1, "lfe_2" -> Sub2, ...).
    let sub_alias = if key.eq_ignore_ascii_case("lfe") || role.eq_ignore_ascii_case("lfe") {
        Some("Sub1".to_string())
    } else {
        key.strip_prefix("lfe_")
            .or_else(|| role.strip_prefix("lfe_"))
            .or_else(|| key.strip_prefix("LFE_"))
            .or_else(|| role.strip_prefix("LFE_"))
            .map(|index| format!("Sub{index}"))
    };
    let key_lower = key.to_ascii_lowercase();
    let role_lower = role.to_ascii_lowercase();
    let mut candidates: Vec<&str> = vec![key, role, &key_lower, &role_lower];
    if let Some(alias) = sub_alias.as_deref() {
        candidates.push(alias);
    }
    for candidate in candidates {
        if let Some(curve) = result.deployed_source_curves.get(candidate) {
            return Some((CurveSource::Deployed, curve.clone()));
        }
        if let Some(channel) = result.channel_results.get(candidate) {
            return Some((CurveSource::ChannelResults, channel.final_curve.clone()));
        }
        if let Some(chain) = result.channels.get(candidate)
            && let Some(curve) = chain.final_curve.as_ref()
        {
            return Some((CurveSource::Channels, Curve::from(curve.clone())));
        }
    }
    // Case-insensitive fallback across all known keys.
    let matches = |name: &str| name.eq_ignore_ascii_case(key) || name.eq_ignore_ascii_case(role);
    if let Some((_, curve)) = result
        .deployed_source_curves
        .iter()
        .find(|(name, _)| matches(name))
    {
        return Some((CurveSource::Deployed, curve.clone()));
    }
    if let Some((_, channel)) = result
        .channel_results
        .iter()
        .find(|(name, _)| matches(name))
    {
        return Some((CurveSource::ChannelResults, channel.final_curve.clone()));
    }
    result
        .channels
        .iter()
        .find(|(name, _)| matches(name))
        .and_then(|(_, chain)| {
            chain
                .final_curve
                .as_ref()
                .map(|curve| (CurveSource::Channels, Curve::from(curve.clone())))
        })
}

/// Log-frequency linear interpolation of `curve` at `frequency`.
fn interp_db(curve: &Curve, frequency: f64) -> Option<f64> {
    if curve.freq.len() < 2 {
        return None;
    }
    let log_f = frequency.ln();
    let mut previous: Option<(f64, usize)> = None;
    for (index, &grid) in curve.freq.iter().enumerate() {
        if !grid.is_finite() {
            continue;
        }
        if grid >= frequency {
            let level = curve.spl.get(index).copied().unwrap_or(f64::NAN);
            match previous {
                None => return Some(level),
                Some((previous_grid, previous_index)) => {
                    let previous_level = curve.spl.get(previous_index).copied().unwrap_or(f64::NAN);
                    if !level.is_finite() || !previous_level.is_finite() {
                        return None;
                    }
                    let log_previous = previous_grid.ln();
                    let log_grid = grid.ln();
                    let span = log_grid - log_previous;
                    if span <= 0.0 {
                        return Some(level);
                    }
                    let t = ((log_f - log_previous) / span).clamp(0.0, 1.0);
                    return Some(previous_level + t * (level - previous_level));
                }
            }
        }
        previous = Some((grid, index));
    }
    previous.and_then(|(_, index)| {
        let level = curve.spl.get(index).copied().unwrap_or(f64::NAN);
        level.is_finite().then_some(level)
    })
}

/// Demeaned RMS and max-absolute residual of `corrected` vs `ideal` in band.
///
/// Both curves are demeaned in-band because the optimizer normalizes absolute
/// level; the round-trip asserts shape reconstruction, not absolute SPL.
/// Returns `(rms_db, max_db, worst_freq_hz, worst_signed_db)` for diagnostics.
fn residual_in_band(
    corrected: &Curve,
    ideal: &Curve,
    band_hz: (f64, f64),
) -> Option<(f64, f64, f64, f64)> {
    let mut corrected_in_band = Vec::new();
    let mut ideal_in_band = Vec::new();
    let mut freqs_in_band = Vec::new();
    for (frequency, level) in corrected.freq.iter().zip(corrected.spl.iter()) {
        if *frequency < band_hz.0 || *frequency > band_hz.1 || !level.is_finite() {
            continue;
        }
        if let Some(expected) = interp_db(ideal, *frequency)
            && expected.is_finite()
        {
            corrected_in_band.push(*level);
            ideal_in_band.push(expected);
            freqs_in_band.push(*frequency);
        }
    }
    if corrected_in_band.len() < 3 {
        return None;
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let corrected_mean = mean(&corrected_in_band);
    let ideal_mean = mean(&ideal_in_band);
    if std::env::var("ROUNDTRIP_GRADE").is_ok() {
        eprintln!(
            "GRADE: n_inband={} corrected_mean={corrected_mean:+.2} ideal_mean={ideal_mean:+.2} grid=[{:.1},{:.1}]",
            corrected_in_band.len(),
            corrected.freq.first().unwrap_or(&f64::NAN),
            corrected.freq.last().unwrap_or(&f64::NAN),
        );
    }
    let mut sum: f64 = 0.0;
    let mut max: f64 = 0.0;
    let mut worst_freq: f64 = f64::NAN;
    let mut worst_signed: f64 = 0.0;
    for ((actual, expected), frequency) in corrected_in_band
        .iter()
        .zip(ideal_in_band.iter())
        .zip(freqs_in_band.iter())
    {
        let residual = (actual - corrected_mean) - (expected - ideal_mean);
        sum += residual * residual;
        if residual.abs() > max {
            max = residual.abs();
            worst_freq = *frequency;
            worst_signed = residual;
        }
    }
    Some((
        (sum / corrected_in_band.len() as f64).sqrt(),
        max,
        worst_freq,
        worst_signed,
    ))
}

/// Verdict for one channel of a round-trip case.
#[derive(Debug)]
pub(super) struct ChannelVerdict {
    pub(super) key: String,
    pub(super) role: String,
    pub(super) source: Option<CurveSource>,
    pub(super) rms_db: Option<f64>,
    pub(super) max_db: Option<f64>,
    pub(super) filter_match: Option<String>,
    pub(super) passed: bool,
    pub(super) detail: String,
}

/// Check recovered PEQs against injected peaks: curve grading alone cannot
/// see a label swap between symmetric channels (each branch flattens
/// whatever it is fed), so when the result publishes per-channel biquads,
/// every significant injected peak must have a counterpart: same octave
/// within ~0.3, opposing gain of at least 35% of the injected magnitude.
/// Partial gains are accepted (the optimizer routinely under-fills); wrong
/// frequencies and wrong signs are not. Returns unmatched descriptions.
fn unmatched_injected(
    injected: &[roomeq_synthetic::PkSpec],
    recovered: &[(f64, f64)],
) -> Vec<String> {
    let mut unmatched = Vec::new();
    for spec in injected {
        if spec.gain_db.abs() < 3.0 {
            continue;
        }
        let matched = recovered.iter().any(|(freq, gain_db)| {
            let octave_gap = (freq / spec.freq_hz).log2().abs();
            octave_gap <= 0.3
                && gain_db.signum() == -spec.gain_db.signum()
                && gain_db.abs() >= 0.35 * spec.gain_db.abs()
        });
        if !matched {
            unmatched.push(format!("{:.0}Hz/{:+.1}dB", spec.freq_hz, spec.gain_db));
        }
    }
    unmatched
}

/// Recovered per-channel PEQs as (freq_hz, gain_db). Routed mains report
/// Post-EQ filters in `channel_results` and the DE-optimized Pre-EQ pass as
/// EQ plugins on the delivered chain; both are delivered per-channel PEQs,
/// so attribution recovers from both places.
fn channel_biquads_for(result: &RoomOptimizationResult, key: &str, role: &str) -> Vec<(f64, f64)> {
    let biquads = |name: &str| {
        result
            .channel_results
            .get(name)
            .map(|channel| &channel.biquads)
    };
    let found = biquads(key).or_else(|| biquads(role)).or_else(|| {
        let matches =
            |name: &str| name.eq_ignore_ascii_case(key) || name.eq_ignore_ascii_case(role);
        result
            .channel_results
            .iter()
            .find(|(name, _)| matches(name))
            .map(|(_, channel)| &channel.biquads)
    });
    let mut recovered: Vec<(f64, f64)> = found
        .map(|list| {
            list.iter()
                .map(|biquad| (biquad.freq, biquad.db_gain))
                .collect()
        })
        .unwrap_or_default();
    let chain = result
        .channels
        .get(key)
        .or_else(|| result.channels.get(role))
        .or_else(|| {
            result
                .channels
                .iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(key) || name.eq_ignore_ascii_case(role))
                .map(|(_, chain)| chain)
        });
    if let Some(chain) = chain {
        for plugin in &chain.plugins {
            if plugin.plugin_type != "eq" {
                continue;
            }
            let Some(filters) = plugin.parameters.get("filters").and_then(|v| v.as_array()) else {
                continue;
            };
            for filter in filters {
                if let (Some(freq), Some(gain)) = (
                    filter.get("freq").and_then(|v| v.as_f64()),
                    filter.get("db_gain").and_then(|v| v.as_f64()),
                ) {
                    recovered.push((freq, gain));
                }
            }
        }
    }
    recovered
}

/// Full report for one round-trip case run.
#[derive(Debug)]
pub(super) struct RoundtripReport {
    pub(super) case_name: String,
    pub(super) verdicts: Vec<ChannelVerdict>,
    pub(super) pre_score: f64,
    pub(super) post_score: f64,
    pub(super) elapsed_ms: u128,
    pub(super) error: Option<String>,
    pub(super) available_keys: Vec<String>,
    pub(super) corrections: Vec<String>,
}

impl RoundtripReport {
    pub(super) fn passed(&self) -> bool {
        self.error.is_none() && self.verdicts.iter().all(|verdict| verdict.passed)
    }

    pub(super) fn summary(&self) -> String {
        use std::fmt::Write as _;
        let mut text = format!(
            "roundtrip {}: {} (pre={:.3} post={:.3} {}ms)\n",
            self.case_name,
            if self.passed() { "PASS" } else { "FAIL" },
            self.pre_score,
            self.post_score,
            self.elapsed_ms
        );
        for verdict in &self.verdicts {
            let _ = writeln!(
                text,
                "  {} [{}] via {:?}: rms={} max={} filters={} -> {} ({})",
                verdict.key,
                verdict.role,
                verdict.source,
                verdict
                    .rms_db
                    .map(|value| format!("{value:.3}dB"))
                    .unwrap_or_else(|| "-".to_string()),
                verdict
                    .max_db
                    .map(|value| format!("{value:.3}dB"))
                    .unwrap_or_else(|| "-".to_string()),
                verdict.filter_match.as_deref().unwrap_or("-"),
                if verdict.passed { "pass" } else { "FAIL" },
                verdict.detail
            );
        }
        if let Some(error) = &self.error {
            let _ = writeln!(text, "  error: {error}");
        }
        if !self.passed() {
            let _ = writeln!(text, "  available keys: {:?}", self.available_keys);
            for correction in &self.corrections {
                let _ = writeln!(text, "  {correction}");
            }
        }
        text
    }
}

/// Run one round-trip case and grade every channel against its ideal.
pub(super) fn run_roundtrip_case(
    case: &RoundtripCase,
    config: &RoomConfig,
    mso_ideal: Option<&Curve>,
) -> RoundtripReport {
    let started = Instant::now();
    let mut available_keys = Vec::new();
    let result = match run_once(config) {
        Ok(result) => result,
        Err(error) => {
            return RoundtripReport {
                case_name: case.name.clone(),
                verdicts: Vec::new(),
                pre_score: f64::NAN,
                post_score: f64::NAN,
                elapsed_ms: started.elapsed().as_millis(),
                error: Some(format!("optimization failed: {error:#}")),
                available_keys,
                corrections: Vec::new(),
            };
        }
    };
    available_keys.extend(result.deployed_source_curves.keys().cloned());
    available_keys.extend(result.channel_results.keys().cloned());
    available_keys.extend(result.channels.keys().cloned());
    available_keys.sort();
    available_keys.dedup();
    if std::env::var("ROUNDTRIP_STAGES").is_ok() {
        for outcome in result.metadata.stage_outcomes.iter() {
            eprintln!(
                "STAGE {}: {:?} checks={:?} advisories={:?}",
                outcome.stage, outcome.status, outcome.checks, outcome.advisories
            );
        }
        eprintln!("ACCEPTANCE: {:?}", result.metadata.correction_acceptance);
        eprintln!(
            "DEPLOYED-KEYS: {:?}",
            result.deployed_source_curves.keys().collect::<Vec<_>>()
        );
        for (name, channel) in result.channel_results.iter() {
            let filters: Vec<String> = channel
                .biquads
                .iter()
                .map(|biquad| {
                    format!(
                        "{:.0}Hz/Q{:.1}/{:+.1}dB",
                        biquad.freq, biquad.q, biquad.db_gain
                    )
                })
                .collect();
            eprintln!("BIQUADS {name}: [{}]", filters.join(", "));
        }
        for (name, chain) in result.channels.iter() {
            let labels: Vec<String> = chain
                .plugins
                .iter()
                .map(|plugin| {
                    let label = plugin
                        .parameters
                        .get("label")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("?");
                    let values = plugin
                        .parameters
                        .as_object()
                        .map(|map| {
                            map.iter()
                                .filter(|(key, _)| {
                                    !["label", "filter_type", "filters", "stages"]
                                        .contains(&key.as_str())
                                })
                                .map(|(key, value)| format!("{key}={value}"))
                                .collect::<Vec<_>>()
                                .join(" ")
                        })
                        .unwrap_or_default();
                    format!("{}/{label} {values}", plugin.plugin_type)
                })
                .collect();
            eprintln!("CHAIN {name}: plugins=[{}]", labels.join(", "));
        }
    }
    let mut corrections: Vec<String> = result
        .channel_results
        .iter()
        .map(|(name, channel)| {
            let filters: Vec<String> = channel
                .biquads
                .iter()
                .map(|biquad| {
                    format!(
                        "{:.0}Hz/Q{:.1}/{:+.1}dB",
                        biquad.freq, biquad.q, biquad.db_gain
                    )
                })
                .collect();
            let evidence: Vec<String> = channel
                .optimizer_evidence
                .iter()
                .map(|run| {
                    format!(
                        "{} conv={} evals={:?}/{} {}",
                        run.algorithm,
                        run.converged,
                        run.evaluation_count,
                        run.evaluation_limit,
                        run.status
                    )
                })
                .collect();
            format!(
                "correction {name}: pre={:.3} post={:.3} [{}] runs=[{}] vetoes={}",
                channel.pre_score,
                channel.post_score,
                filters.join(", "),
                evidence.join("; "),
                channel.audibility_veto.len(),
            )
        })
        .collect();
    corrections.sort();

    let mut verdicts = Vec::with_capacity(case.channels.len());
    for channel in &case.channels {
        let Some((source, corrected)) = final_curve_for(&result, &channel.key, &channel.role)
        else {
            verdicts.push(ChannelVerdict {
                key: channel.key.clone(),
                role: channel.role.clone(),
                source: None,
                rms_db: None,
                max_db: None,
                filter_match: None,
                passed: false,
                detail: "no corrected curve published for channel".to_string(),
            });
            continue;
        };
        // Deployed (post-routing) curves grade against the splice target when
        // the fixture declares one; raw channel curves grade against the
        // channel's own ideal. MSO subs share one flat bass ideal.
        let is_sub_key = channel.key == "lfe" || channel.key.starts_with("lfe_");
        let ideal = if is_sub_key {
            mso_ideal.unwrap_or(&channel.ideal)
        } else if source == CurveSource::Deployed {
            channel.deployed_ideal.as_ref().unwrap_or(&channel.ideal)
        } else {
            &channel.ideal
        };
        if std::env::var("ROUNDTRIP_SPLICE").is_ok() {
            eprintln!("SPLICE {} (source={source:?}):", channel.key);
            for (frequency, level) in corrected.freq.iter().zip(corrected.spl.iter()) {
                if *frequency < 40.0 || *frequency > 150.0 {
                    continue;
                }
                let expected = interp_db(ideal, *frequency).unwrap_or(f64::NAN);
                let raw = interp_db(&channel.deformed, *frequency).unwrap_or(f64::NAN);
                eprintln!(
                    "  {frequency:7.1}Hz deployed={level:+7.2} ideal={expected:+7.2} raw={raw:+7.2}"
                );
            }
        }
        // Attribution check: when per-channel biquads are published, the
        // recovered filters must invert this channel's injected peaks.
        // Skipped where the path publishes no per-channel biquads.
        let recovered = channel_biquads_for(&result, &channel.key, &channel.role);
        let (filter_match, filters_ok) = if recovered.is_empty() || channel.injected.is_empty() {
            (None, true)
        } else {
            let unmatched = unmatched_injected(&channel.injected, &recovered);
            if unmatched.is_empty() {
                (Some("match".to_string()), true)
            } else {
                (Some(format!("UNMATCHED [{}]", unmatched.join(", "))), false)
            }
        };
        match residual_in_band(&corrected, ideal, channel.band_hz) {
            None => verdicts.push(ChannelVerdict {
                key: channel.key.clone(),
                role: channel.role.clone(),
                source: Some(source),
                rms_db: None,
                max_db: None,
                filter_match,
                passed: false,
                detail: format!(
                    "fewer than 3 corrected points in band {:?}",
                    channel.band_hz
                ),
            }),
            Some((rms, max, worst_freq, worst_signed)) => {
                let passed = rms <= channel.max_rms_db && max <= channel.max_abs_db && filters_ok;
                verdicts.push(ChannelVerdict {
                    key: channel.key.clone(),
                    role: channel.role.clone(),
                    source: Some(source),
                    rms_db: Some(rms),
                    max_db: Some(max),
                    filter_match,
                    passed,
                    detail: format!(
                        "limits rms<={:.2}dB max<={:.2}dB band={:?} worst={:.0}Hz ({:+.2}dB)",
                        channel.max_rms_db,
                        channel.max_abs_db,
                        channel.band_hz,
                        worst_freq,
                        worst_signed
                    ),
                });
            }
        }
    }

    RoundtripReport {
        case_name: case.name.clone(),
        verdicts,
        pre_score: result.combined_pre_score,
        post_score: result.combined_post_score,
        elapsed_ms: started.elapsed().as_millis(),
        error: None,
        available_keys,
        corrections,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_synthetic::roundtrip_2_0;
    use roomeq_synthetic::roundtrip_2_1;
    use roomeq_synthetic::roundtrip_2_2_mso_subs;
    use roomeq_synthetic::roundtrip_5_1;
    use roomeq_synthetic::roundtrip_conflicting_seats;
    use roomeq_synthetic::roundtrip_identity_2_0;

    fn assert_roundtrip(case: &RoundtripCase, config: &RoomConfig, mso_ideal: Option<&Curve>) {
        let report = run_roundtrip_case(case, config, mso_ideal);
        eprintln!("{}", report.summary());
        assert!(report.post_score.is_finite(), "non-finite post score");
        assert!(report.passed(), "round-trip failed:\n{}", report.summary());
    }

    #[test]
    fn roundtrip_2_0_reconstructs_ideal() {
        let case = roundtrip_2_0();
        let config =
            build_roundtrip_config(&case, &[("l", "L"), ("r", "R")], RoundtripSubs::None, None);
        config.validate_structure().unwrap();
        assert_roundtrip(&case, &config, None);
    }

    #[test]
    fn roundtrip_identity_2_0_stays_flat() {
        let case = roundtrip_identity_2_0();
        let config =
            build_roundtrip_config(&case, &[("l", "L"), ("r", "R")], RoundtripSubs::None, None);
        config.validate_structure().unwrap();
        assert_roundtrip(&case, &config, None);
    }

    #[test]
    fn roundtrip_2_1_reconstructs_ideal() {
        let case = roundtrip_2_1();
        let config = build_roundtrip_config(
            &case,
            &[("l", "L"), ("r", "R")],
            RoundtripSubs::Single,
            None,
        );
        config.validate_structure().unwrap();
        assert_roundtrip(&case, &config, None);
    }

    #[test]
    fn roundtrip_5_1_reconstructs_ideal() {
        let case = roundtrip_5_1();
        let config = build_roundtrip_config(
            &case,
            &[
                ("l", "L"),
                ("r", "R"),
                ("c", "C"),
                ("sl", "SL"),
                ("sr", "SR"),
            ],
            RoundtripSubs::Single,
            None,
        );
        config.validate_structure().unwrap();
        assert_roundtrip(&case, &config, None);
    }

    #[test]
    fn roundtrip_2_2_mso_reconstructs_bass() {
        let mains = roundtrip_2_1();
        let (ideal, sub_seats, injected) = roundtrip_2_2_mso_subs();
        assert_eq!(sub_seats.len(), 2);
        // Two seats per main, matching the subs' seat order: same peaks,
        // different delays + noise. Singleton mains are not broadcast.
        let main_seats = |key: &str, delays: [f64; 2], seed_base: u64| {
            let channel = mains.get(key).expect("2.1 case carries main");
            delays
                .iter()
                .enumerate()
                .map(|(seat, delay)| {
                    roomeq_synthetic::deform_curve(
                        &roomeq_synthetic::with_delay_phase(&channel.ideal, *delay),
                        &channel.injected,
                        0.0,
                        0.3,
                        roomeq_synthetic::ROUNDTRIP_SEED.wrapping_add(seed_base + seat as u64),
                    )
                })
                .collect::<Vec<_>>()
        };
        let seats_l = main_seats("l", [1.0, 1.15], 80);
        let seats_r = main_seats("r", [1.1, 1.25], 82);
        let mut channels = vec![
            mains.get("l").expect("2.1 case carries l").clone(),
            mains.get("r").expect("2.1 case carries r").clone(),
        ];
        // The workflow publishes one logical sub channel (Sub1) for the MSO
        // array; both physical subs still run the joint path, and their
        // errors would leak into the mains' splice region.
        let (seats, specs) = (&sub_seats[0], &injected[0]);
        channels.push(roomeq_synthetic::RoundtripChannel {
            key: "lfe_1".to_string(),
            role: "LFE_1".to_string(),
            ideal: ideal.clone(),
            deployed_ideal: None,
            deformed: seats[0].clone(),
            injected: specs.clone(),
            band_hz: (25.0, 75.0),
            max_rms_db: 2.0,
            max_abs_db: 5.0,
        });
        let case = RoundtripCase {
            name: "roundtrip_2_2_mso".to_string(),
            channels,
            num_filters: 3,
        };
        let mut seats = HashMap::new();
        seats.insert("l".to_string(), seats_l);
        seats.insert("r".to_string(), seats_r);
        let config = build_roundtrip_config(
            &case,
            &[("l", "L"), ("r", "R")],
            RoundtripSubs::SplitMso { curves: sub_seats },
            Some(RoundtripSeats {
                seats,
                strategy: roomeq_model::MultiMeasurementStrategy::Minimax,
            }),
        );
        config.validate_structure().unwrap();
        assert_roundtrip(&case, &config, Some(&ideal));
    }

    /// Guard of the guard: feed L's measurement to R and vice versa. Curve
    /// grading cannot see the swap (each branch flattens whatever it is
    /// fed), so the report must fail on filter attribution instead. If this
    /// ever passes, the attribution check has lost its teeth.
    #[test]
    fn roundtrip_swapped_measurements_must_mismatch_filters() {
        let case = roundtrip_2_0();
        let left = case.get("l").expect("2.0 carries l").clone();
        let right = case.get("r").expect("2.0 carries r").clone();
        let swapped = RoundtripCase {
            name: "roundtrip_swapped_lr".to_string(),
            channels: vec![
                roomeq_synthetic::RoundtripChannel {
                    deformed: right.deformed.clone(),
                    ..left
                },
                roomeq_synthetic::RoundtripChannel {
                    deformed: left.deformed.clone(),
                    ..right
                },
            ],
            num_filters: 4,
        };
        let config = build_roundtrip_config(
            &swapped,
            &[("l", "L"), ("r", "R")],
            RoundtripSubs::None,
            None,
        );
        let report = run_roundtrip_case(&swapped, &config, None);
        eprintln!("{}", report.summary());
        assert!(
            !report.passed(),
            "swapped L/R still passed: attribution check lost its teeth"
        );
        assert!(
            report.verdicts.iter().any(|verdict| verdict
                .filter_match
                .as_deref()
                .is_some_and(|match_| match_.contains("UNMATCHED"))),
            "swapped L/R must fail specifically on filter attribution"
        );
    }

    /// Guard of the guard: grade a correct reconstruction against a tilted
    /// ideal. Curve grading must fail while filter attribution still matches,
    /// proving the two layers are independent.
    #[test]
    fn roundtrip_tilted_ideal_must_fail_curve_grading() {
        let case = roundtrip_2_0();
        let mut channels = Vec::new();
        for channel in &case.channels {
            let mut tilted = channel.ideal.clone();
            // +10 dB tilt across the band: reconstruction cannot match.
            let n = tilted.freq.len() as f64;
            for (index, level) in tilted.spl.iter_mut().enumerate() {
                *level += 10.0 * index as f64 / n;
            }
            channels.push(roomeq_synthetic::RoundtripChannel {
                ideal: tilted,
                ..channel.clone()
            });
        }
        let tilted = RoundtripCase {
            name: "roundtrip_tilted_ideal".to_string(),
            channels,
            num_filters: 4,
        };
        let config = build_roundtrip_config(
            &tilted,
            &[("l", "L"), ("r", "R")],
            RoundtripSubs::None,
            None,
        );
        let report = run_roundtrip_case(&tilted, &config, None);
        eprintln!("{}", report.summary());
        assert!(
            !report.passed(),
            "tilted ideal still passed: curve grading lost its teeth"
        );
    }

    /// Crossed mains run as pure stereo: without bass management the mains
    /// get full-range EQ and must reconstruct flat, including the rolled-off
    /// bass. Locks the stereo full-range contract (a band restriction here
    /// would leave bass unfilled and fail).
    #[test]
    fn roundtrip_crossed_mains_stereo_recovers_bass() {
        let full = roundtrip_2_1();
        let case = RoundtripCase {
            name: "roundtrip_crossed_stereo".to_string(),
            channels: vec![
                full.get("l").expect("2.1 case carries l").clone(),
                full.get("r").expect("2.1 case carries r").clone(),
            ],
            // Bass rolloff needs its own filters on top of the peaks.
            num_filters: 5,
        };
        let config =
            build_roundtrip_config(&case, &[("l", "L"), ("r", "R")], RoundtripSubs::None, None);
        config.validate_structure().unwrap();
        assert_roundtrip(&case, &config, None);
    }

    #[test]
    fn roundtrip_multiseat_2_0_minimax_splits_conflict() {
        let stereo = roundtrip_2_0();
        let left = stereo.get("l").expect("2.0 case carries l").clone();
        let right = stereo.get("r").expect("2.0 case carries r").clone();
        let (ideal_l, seats_l) = roundtrip_conflicting_seats(&left.ideal, 70);
        let (ideal_r, seats_r) = roundtrip_conflicting_seats(&right.ideal, 72);
        let mut seats = HashMap::new();
        seats.insert("l".to_string(), seats_l);
        seats.insert("r".to_string(), seats_r);
        // Only the shared 150 Hz peak must invert; the opposed 70 Hz conflict
        // must split (~0 gain), so it stays out of the injected list.
        let shared_only = vec![roomeq_synthetic::PkSpec::new(150.0, 3.0, 6.0)];
        let case = RoundtripCase {
            name: "roundtrip_multiseat_2_0_minimax".to_string(),
            channels: vec![
                roomeq_synthetic::RoundtripChannel {
                    ideal: ideal_l,
                    deformed: left.deformed.clone(),
                    injected: shared_only.clone(),
                    band_hz: (40.0, 500.0),
                    // Conflict seats cannot both reconstruct: the shared peak
                    // must collapse while the opposed 70 Hz peak splits.
                    max_rms_db: 3.0,
                    max_abs_db: 6.0,
                    ..left
                },
                roomeq_synthetic::RoundtripChannel {
                    ideal: ideal_r,
                    deformed: right.deformed.clone(),
                    injected: shared_only,
                    band_hz: (40.0, 500.0),
                    max_rms_db: 3.0,
                    max_abs_db: 6.0,
                    ..right
                },
            ],
            num_filters: 3,
        };
        let config = build_roundtrip_config(
            &case,
            &[("l", "L"), ("r", "R")],
            RoundtripSubs::None,
            Some(RoundtripSeats {
                seats,
                strategy: roomeq_model::MultiMeasurementStrategy::Minimax,
            }),
        );
        config.validate_structure().unwrap();
        assert_roundtrip(&case, &config, None);
    }
}
