//! W3 — playback verification bundles and operator-capture comparison.
//!
//! Verification binds later measurement to immutable identities:
//!
//! - [`VerificationBundle`] packages baseline/candidate graph identities,
//!   the exact exported DSP snapshot, calibration and source/seat manifest,
//!   stimulus settings, analysis policy, and expected outputs. Bundles are
//!   built from frozen graphs only.
//! - [`plan_routes`] generates every required test route separately:
//!   isolated sources, main/sub overlap, coherent L+R bass, LFE, and
//!   held-out seats. LFE gain is owned exactly once: [`count_lfe_gain_stages`]
//!   proves no double LFE gain across the route set.
//! - Operator-provided captures enter only through
//!   [`assess_imported_capture`] (measurements APIs plus quality V2
//!   [`compare_prediction_capture`](roomeq_engine::quality::compare_prediction_capture)).
//!   This module never opens a hardware recording session implicitly.
//! - Predicted, backend-rendered, and actually recorded results stay
//!   separately labelled ([`PlaybackEvidenceKind`](roomeq_engine::quality::PlaybackEvidenceKind)).
//!   Small-signal and dynamic/limiter trials are distinct evidence
//!   ([`TrialLevel`]). Missing captures and mismatched hashes stay
//!   unassessed and never promote a playback claim.
//! - [`verify_fir_tails_and_delays`] keeps exact FIR tails and the delay
//!   ledger intact from stimulus packaging to the exported chain.

use roomeq_engine::quality::{
    CaptureComparisonReport, CaptureTolerances, DeclaredAlignment, PlaybackBinding,
    PlaybackEvidenceKind, PlaybackLevel, compare_prediction_capture,
};
use roomeq_model::DspGraph;
use roomeq_model::decision_ledger::{CaptureKind, PlaybackComparison};
use serde::{Deserialize, Serialize};

use crate::final_ledger::GraphIdentity;

/// Verification bundle version pinned by this workflow lane.
pub const VERIFICATION_BUNDLE_VERSION: &str = "workflow-verification-v1";

/// Small-signal response and compression/limiter trials are distinct evidence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialLevel {
    /// Linear transfer check; says nothing about maximum output.
    #[default]
    SmallSignal,
    /// Driven-level compression/limiter assessment.
    DynamicLimiter,
}

impl TrialLevel {
    /// Machine-readable trial label shared with operator manifests:
    /// small-signal and limiter trials never merge into one claim.
    pub fn as_label(self) -> &'static str {
        self.processing_state()
    }

    fn playback_level(self) -> PlaybackLevel {
        match self {
            TrialLevel::SmallSignal => PlaybackLevel::SmallSignal,
            TrialLevel::DynamicLimiter => PlaybackLevel::MaximumOutput,
        }
    }

    fn processing_state(self) -> &'static str {
        match self {
            TrialLevel::SmallSignal => "small_signal",
            TrialLevel::DynamicLimiter => "dynamic_limiter",
        }
    }
}

/// Source/seat manifest plus stimulus and policy identities for one bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationManifest {
    /// Logical sources under test, e.g. `["left", "right", "lfe"]`.
    pub source_ids: Vec<String>,
    /// Seats covered, including held-out seats under test.
    pub seat_ids: Vec<String>,
    /// Held-out seats: measured but never used for fitting.
    pub held_out_seats: Vec<String>,
    /// Sample rate in Hz the stimulus was rendered at.
    pub sample_rate_hz: f64,
    /// Calibration reference identity.
    pub calibration_id: String,
    /// Stimulus content hash.
    pub stimulus_hash: String,
    /// Comparison policy version (quality V2 contract).
    pub comparison_policy_version: String,
}

/// Per-channel DSP snapshot: plugin order, FIR tails, and delay ledger.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChannelSnapshot {
    pub channel: String,
    /// Plugin kinds in order, e.g. `["eq", "gain", "convolution"]`.
    pub plugin_kinds: Vec<String>,
    /// `(ir_file, taps)` for every convolution plugin, in order.
    pub fir_tails: Vec<(String, usize)>,
    /// Sum of serial channel delay plugins, in milliseconds; excludes FIR/driver latency.
    pub delay_ms: Option<f64>,
}

/// Exact exported resources referenced by the bundle (IR sidecars, ...).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleResource {
    /// Resource file name, e.g. the convolution `ir_file`.
    pub resource_id: String,
    /// Tap count for FIR resources.
    pub taps: usize,
    /// Content hash of the resource bytes.
    pub content_hash: String,
}

/// One playback verification bundle, bound to frozen graph identities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationBundle {
    pub bundle_version: String,
    /// Immutable baseline graph fingerprint.
    pub baseline_graph: String,
    /// Immutable candidate graph fingerprint.
    pub candidate_graph: String,
    pub manifest: VerificationManifest,
    pub channels: Vec<ChannelSnapshot>,
    pub resources: Vec<BundleResource>,
    pub trial_level: TrialLevel,
    /// Expected small-signal/delivered outputs per channel, in dB.
    pub expected_outputs_db: Vec<(String, f64)>,
}

impl VerificationBundle {
    /// Build a bundle from frozen baseline/candidate identities and the
    /// exact exported candidate DSP. The stimulus hash, calibration, and
    /// policy version are caller-supplied: this constructor never invents
    /// measurement identities.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        baseline_graph: &GraphIdentity,
        candidate_graph: &GraphIdentity,
        candidate_dsp: &DspGraph,
        source_ids: Vec<String>,
        seat_ids: Vec<String>,
        held_out_seats: Vec<String>,
        sample_rate_hz: f64,
        calibration_id: String,
        stimulus_hash: String,
        comparison_policy_version: String,
        trial_level: TrialLevel,
        resources: Vec<BundleResource>,
    ) -> Result<Self, String> {
        if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
            return Err(String::from(
                "verification sample rate must be finite and positive",
            ));
        }
        if stimulus_hash.trim().is_empty() {
            return Err(String::from("verification stimulus hash must not be empty"));
        }
        if calibration_id.trim().is_empty() {
            return Err(String::from(
                "verification calibration identity must not be empty",
            ));
        }
        candidate_dsp.validate().map_err(|message| {
            format!("verification bundle needs a valid candidate graph: {message}")
        })?;
        let mut channels = Vec::new();
        for (name, chain) in &candidate_dsp.channels {
            let plugin_kinds = chain
                .plugins
                .iter()
                .map(|plugin| plugin.plugin_type.clone())
                .collect();
            let mut fir_tails = Vec::new();
            for plugin in &chain.plugins {
                if plugin.plugin_type == "convolution" {
                    let ir_file = plugin
                        .parameters
                        .get("ir_file")
                        .and_then(|value| value.as_str())
                        .ok_or_else(|| {
                            format!("channel '{name}' convolution plugin has no ir_file")
                        })?;
                    let taps = resources
                        .iter()
                        .find(|resource| resource.resource_id == ir_file)
                        .ok_or_else(|| {
                            format!("channel '{name}' FIR resource '{ir_file}' missing from bundle")
                        })?
                        .taps;
                    fir_tails.push((ir_file.to_string(), taps));
                }
            }
            let delay_ms = channel_delay_ms(&chain.plugins)?;
            channels.push(ChannelSnapshot {
                channel: name.clone(),
                plugin_kinds,
                fir_tails,
                delay_ms,
            });
        }
        channels.sort_by(|left, right| left.channel.cmp(&right.channel));
        Ok(VerificationBundle {
            bundle_version: VERIFICATION_BUNDLE_VERSION.to_string(),
            baseline_graph: baseline_graph.fingerprint.clone(),
            candidate_graph: candidate_graph.fingerprint.clone(),
            manifest: VerificationManifest {
                source_ids,
                seat_ids,
                held_out_seats,
                sample_rate_hz,
                calibration_id,
                stimulus_hash,
                comparison_policy_version,
            },
            channels,
            resources,
            trial_level,
            expected_outputs_db: Vec::new(),
        })
    }

    /// K5 comparison descriptor for one source/seat pair in this bundle.
    pub fn comparison_for(
        &self,
        source_id: &str,
        seat_id: &str,
        capture_kind: CaptureKind,
    ) -> PlaybackComparison {
        PlaybackComparison {
            baseline_graph_identity: Some(self.baseline_graph.clone()),
            candidate_graph_identity: Some(self.candidate_graph.clone()),
            sample_rate_hz: Some(self.manifest.sample_rate_hz as u32),
            calibration_ref: Some(self.manifest.calibration_id.clone()),
            source_id: source_id.to_string(),
            seat_ids: vec![seat_id.to_string()],
            stimulus_hash: Some(self.manifest.stimulus_hash.clone()),
            processing_state: self.trial_level.processing_state().to_string(),
            comparison_policy_version: self.manifest.comparison_policy_version.clone(),
            capture_kind,
            ..PlaybackComparison::default()
        }
    }
}

/// Required verification routes, generated separately with protection kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationRoute {
    /// One source driven in isolation.
    Isolated { source: String },
    /// Main/sub overlap region for one pairing.
    Overlap { main: String, sub: String },
    /// Coherent L+R bass summation.
    CoherentLrBass,
    /// LFE channel (gain owned exactly once).
    Lfe { channel: String },
    /// Seat held out of fitting, measured for generalization.
    HeldOut { seat: String },
}

/// One planned route with its explicit per-channel gain stages in dB.
///
/// Gain lives in this plan exactly once per contribution: summing the LFE
/// entries across all routes must yield one application (see
/// [`count_lfe_gain_stages`]).
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedRoute {
    pub route: VerificationRoute,
    pub gain_stages_db: Vec<(String, f64)>,
}

/// Generate every required route: isolated sources, main/sub overlap,
/// coherent L+R bass, LFE, and held-out seats.
pub fn plan_routes(
    sources: &[String],
    sub_pairings: &[(String, String)],
    lfe_channel: Option<&str>,
    held_out_seats: &[String],
    lfe_gain_db: f64,
) -> Result<Vec<PlannedRoute>, String> {
    if !lfe_gain_db.is_finite() {
        return Err(String::from("LFE verification gain must be finite"));
    }
    let mut routes = Vec::new();
    for source in sources {
        routes.push(PlannedRoute {
            route: VerificationRoute::Isolated {
                source: source.clone(),
            },
            gain_stages_db: vec![(source.clone(), 0.0)],
        });
    }
    for (main, sub) in sub_pairings {
        routes.push(PlannedRoute {
            route: VerificationRoute::Overlap {
                main: main.clone(),
                sub: sub.clone(),
            },
            gain_stages_db: vec![(main.clone(), 0.0), (sub.clone(), 0.0)],
        });
    }
    if sources.len() >= 2 {
        routes.push(PlannedRoute {
            route: VerificationRoute::CoherentLrBass,
            gain_stages_db: vec![(sources[0].clone(), 0.0), (sources[1].clone(), 0.0)],
        });
    }
    if let Some(lfe) = lfe_channel {
        // The LFE gain is applied on this route only — never again on the
        // isolated/L+R routes above.
        routes.push(PlannedRoute {
            route: VerificationRoute::Lfe {
                channel: lfe.to_string(),
            },
            gain_stages_db: vec![(lfe.to_string(), lfe_gain_db)],
        });
    }
    for seat in held_out_seats {
        routes.push(PlannedRoute {
            route: VerificationRoute::HeldOut { seat: seat.clone() },
            gain_stages_db: Vec::new(),
        });
    }
    Ok(routes)
}

/// Count LFE gain applications across the planned routes.
///
/// Must be exactly one when an LFE channel is under test: the LFE route
/// owns it, and no other route re-applies LFE gain (no double LFE).
pub fn count_lfe_gain_stages(routes: &[PlannedRoute], lfe_channel: &str) -> usize {
    routes
        .iter()
        .flat_map(|route| route.gain_stages_db.iter())
        .filter(|(channel, _)| channel == lfe_channel)
        .count()
}

/// Operator-supplied capture with its binding and evidence class.
#[derive(Debug, Clone)]
pub struct ImportedCapture {
    pub binding: PlaybackBinding,
    pub curve: roomeq_model::Curve,
    pub evidence_kind: PlaybackEvidenceKind,
}

/// Outcome of importing one operator capture against a bundle route.
#[derive(Debug, Clone)]
pub enum CaptureAssessment {
    /// V2 comparison ran on bound real data.
    Assessed(CaptureComparisonReport),
    /// Missing capture or mismatched identity: unassessed, never promoted.
    Unassessed { reason: String },
}

impl CaptureAssessment {
    /// True only for a passing acoustic capture: simulated and
    /// backend-rendered passes never verify the room.
    pub fn verifies_room(&self) -> bool {
        match self {
            CaptureAssessment::Assessed(report) => report.counts_as_acoustic_verification(),
            CaptureAssessment::Unassessed { .. } => false,
        }
    }
}

/// Import one operator capture through measurements APIs and quality V2.
///
/// `expected` is the bundle-side binding (graph, source, seat, stimulus,
/// sample rate, calibration, processing state). A missing capture stays
/// unassessed; any binding mismatch (F13) stays unassessed with its reason.
/// Only a fully bound capture reaches [`compare_prediction_capture`], and
/// only its report can change playback status.
#[allow(clippy::too_many_arguments)]
pub fn assess_imported_capture(
    expected: &PlaybackBinding,
    capture: Option<&ImportedCapture>,
    prediction: &roomeq_model::Curve,
    declared: &DeclaredAlignment,
    tolerances: &CaptureTolerances,
    band_hz: [f64; 2],
) -> Result<CaptureAssessment, String> {
    let Some(capture) = capture else {
        return Ok(CaptureAssessment::Unassessed {
            reason: String::from("no operator capture supplied; playback evidence is unassessed"),
        });
    };
    if let Err(mismatch) = expected.check_compatible(&capture.binding) {
        return Ok(CaptureAssessment::Unassessed {
            reason: format!("capture manifest mismatch: {mismatch}"),
        });
    }
    let report = compare_prediction_capture(
        prediction,
        &capture.curve,
        expected,
        &capture.binding,
        capture.evidence_kind,
        TrialLevel::SmallSignal.playback_level(),
        declared,
        tolerances,
        band_hz,
    )?;
    Ok(CaptureAssessment::Assessed(report))
}

/// Real playback status: only bound, passing, acoustic V2 evidence promotes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaybackStatus {
    #[default]
    Unassessed,
    SimulatedPass,
    Verified,
    /// At least one required comparison lacks complete, compatible evidence.
    InsufficientEvidence,
    /// At least one required assessed comparison failed its declared budget.
    Failed,
}

/// Advance playback status from one capture assessment.
///
/// Only an assessed, passing, acoustic capture verifies the room. Simulated
/// or backend-rendered passes record software behavior (`SimulatedPass`) and
/// never verify the room. Failed/incomplete comparisons are sticky within a
/// batch, so a later passing seat cannot erase a required-seat failure. Begin
/// a new independent verification batch with `Unassessed`.
pub fn update_playback_status(
    current: PlaybackStatus,
    assessment: &CaptureAssessment,
    evidence_kind: PlaybackEvidenceKind,
) -> PlaybackStatus {
    if current == PlaybackStatus::Failed {
        return current;
    }
    let CaptureAssessment::Assessed(report) = assessment else {
        return PlaybackStatus::InsufficientEvidence;
    };
    if report.evidence_class != evidence_kind.as_str() {
        return PlaybackStatus::InsufficientEvidence;
    }
    if report
        .metric_outcomes
        .iter()
        .any(|metric| metric.assessed && !metric.passed)
    {
        return PlaybackStatus::Failed;
    }
    if !report.is_complete_pass() || current == PlaybackStatus::InsufficientEvidence {
        return PlaybackStatus::InsufficientEvidence;
    }
    if evidence_kind == PlaybackEvidenceKind::Acoustic && current != PlaybackStatus::SimulatedPass {
        PlaybackStatus::Verified
    } else {
        // A batch containing any software-only route cannot verify all acoustic
        // routes, even when another route has a passing microphone capture.
        PlaybackStatus::SimulatedPass
    }
}

// This is only explicit channel delay, not the frequency-dependent group delay
// or latency of FIRs, drivers, limiters, and routed parallel branches.
fn channel_delay_ms(plugins: &[roomeq_model::PluginConfigWrapper]) -> Result<Option<f64>, String> {
    let mut total = None;
    let mut in_split = false;
    for plugin in plugins {
        match plugin.plugin_type.as_str() {
            "band_split" if !in_split => in_split = true,
            "band_merge" if in_split => in_split = false,
            "band_split" | "band_merge" => {
                return Err("invalid or nested band split in delay snapshot".into());
            }
            "delay" => {
                if in_split {
                    return Err(
                        "branch-local delay cannot be represented by a channel delay snapshot"
                            .into(),
                    );
                }
                let delay = plugin
                    .parameters
                    .get("delay_ms")
                    .and_then(serde_json::Value::as_f64)
                    .filter(|value| value.is_finite() && *value >= 0.0)
                    .ok_or_else(|| {
                        "channel delay must be finite, nonnegative milliseconds".to_string()
                    })?;
                let sum = total.unwrap_or(0.0) + delay;
                if !sum.is_finite() {
                    return Err("channel delay sum overflows".into());
                }
                total = Some(sum);
            }
            _ => {}
        }
    }
    if in_split {
        return Err("unclosed band split in delay snapshot".into());
    }
    Ok(total)
}

/// Keep exact FIR tails and the delay ledger intact to the exported chain.
///
/// Every bundle FIR tail must resolve to a convolution `ir_file` in the
/// exported graph with the same declared resource tap count, and every delay
/// ledger entry must match the sum of serial channel delay plugins. Tails are
/// never truncated. Resource bytes must be checked separately by the importer;
/// this function has no file access and cannot authenticate content hashes.
///
/// # Errors
/// Returns an error for missing resources/channels, malformed delay values,
/// branch-local delays, or inconsistent snapshot and ledger delays.
pub fn verify_fir_tails_and_delays(
    bundle: &VerificationBundle,
    graph: &DspGraph,
    delay_ledger_ms: &[(String, f64)],
) -> Result<(), String> {
    graph
        .validate()
        .map_err(|message| format!("exported graph invalid: {message}"))?;
    for (name, delay) in delay_ledger_ms {
        if !delay.is_finite()
            || *delay < 0.0
            || !bundle
                .channels
                .iter()
                .any(|channel| &channel.channel == name)
        {
            return Err(format!("invalid or unknown delay ledger channel '{name}'"));
        }
    }
    for channel in &bundle.channels {
        let chain = graph.channels.get(&channel.channel).ok_or_else(|| {
            format!(
                "exported graph lost verification channel '{}'",
                channel.channel
            )
        })?;
        let exported_ms = channel_delay_ms(&chain.plugins)?;
        // The 1e-9 ms tolerance is for serialization arithmetic,
        // not an acoustic timing acceptance budget.
        let delays_match = |left: Option<f64>, right: Option<f64>| match (left, right) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.is_finite() && b.is_finite() && a >= 0.0 && b >= 0.0 && (a - b).abs() <= 1e-9
            }
            _ => false,
        };
        if !delays_match(channel.delay_ms, exported_ms) {
            return Err(format!(
                "delay snapshot for '{}' does not match the exported chain",
                channel.channel
            ));
        }
        for (ir_file, taps) in &channel.fir_tails {
            let resource = bundle
                .resources
                .iter()
                .find(|resource| &resource.resource_id == ir_file)
                .ok_or_else(|| format!("bundle resource '{ir_file}' missing"))?;
            if resource.taps != *taps {
                return Err(format!(
                    "FIR tail for '{ir_file}' changed: bundle has {taps} taps, resource has {}",
                    resource.taps
                ));
            }
            let exported = chain.plugins.iter().find(|plugin| {
                plugin.plugin_type == "convolution"
                    && plugin
                        .parameters
                        .get("ir_file")
                        .and_then(|value| value.as_str())
                        == Some(ir_file.as_str())
            });
            if exported.is_none() {
                return Err(format!(
                    "exported chain lost convolution resource '{ir_file}' on '{}'",
                    channel.channel
                ));
            }
        }
        for (ledger_channel, ledger_ms) in delay_ledger_ms {
            if ledger_channel == &channel.channel {
                match exported_ms {
                    Some(exported_ms) if (exported_ms - ledger_ms).abs() <= 1e-9 => {}
                    _ => {
                        return Err(format!(
                            "delay ledger for '{}' ({} ms) does not match the exported chain",
                            channel.channel, ledger_ms
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use roomeq_model::contracts::Plugin;

    #[test]
    fn roadmap_correction_capture_batch_cannot_hide_failed_or_missing_seats() {
        let binding = binding_fixture("graph-test");
        let mut prediction = flat_curve(20.0, 20_000.0, 64, 80.0);
        prediction.phase = Some(prediction.freq.mapv(|_| 0.0));
        let (declared, tolerances) = tolerances_fixture();
        let assess = |level: f64, kind| {
            let mut curve = prediction.clone();
            curve.spl.fill(level);
            assess_imported_capture(
                &binding,
                Some(&ImportedCapture {
                    binding: binding.clone(),
                    curve,
                    evidence_kind: kind,
                }),
                &prediction,
                &declared,
                &tolerances,
                [40.0, 4000.0],
            )
            .unwrap()
        };
        let good = assess(80.0, PlaybackEvidenceKind::Acoustic);
        let failed = assess(70.0, PlaybackEvidenceKind::Acoustic);
        let missing = CaptureAssessment::Unassessed {
            reason: "missing required seat".to_owned(),
        };
        for (bad, expected) in [
            (&failed, PlaybackStatus::Failed),
            (&missing, PlaybackStatus::InsufficientEvidence),
        ] {
            for sequence in [[&good, bad], [bad, &good]] {
                let status =
                    sequence
                        .into_iter()
                        .fold(PlaybackStatus::Unassessed, |status, assessment| {
                            update_playback_status(
                                status,
                                assessment,
                                PlaybackEvidenceKind::Acoustic,
                            )
                        });
                assert_eq!(status, expected);
            }
        }
        let simulated = assess(80.0, PlaybackEvidenceKind::Simulated);
        let status = update_playback_status(
            PlaybackStatus::Verified,
            &simulated,
            PlaybackEvidenceKind::Simulated,
        );
        assert_eq!(status, PlaybackStatus::SimulatedPass);
        assert_eq!(
            update_playback_status(status, &good, PlaybackEvidenceKind::Acoustic),
            PlaybackStatus::SimulatedPass
        );
        assert_eq!(
            update_playback_status(
                PlaybackStatus::Unassessed,
                &simulated,
                PlaybackEvidenceKind::Acoustic
            ),
            PlaybackStatus::InsufficientEvidence
        );
    }

    fn fingerprint_graph(channels: &[&str]) -> (DspGraph, GraphIdentity) {
        let mut graph = DspGraph::new("test");
        for channel in channels {
            graph.add_channel(
                (*channel).to_string(),
                vec![Plugin {
                    kind: String::from("eq"),
                    parameters: serde_json::json!({}),
                }],
            );
        }
        let identity = crate::final_ledger::canonical_graph_identity(&graph);
        (graph, identity)
    }

    fn flat_curve(
        first_hz: f64,
        last_hz: f64,
        points: usize,
        level_db: f64,
    ) -> roomeq_model::Curve {
        roomeq_model::Curve {
            freq: Array1::logspace(10.0, first_hz.log10(), last_hz.log10(), points),
            spl: Array1::from_elem(points, level_db),
            phase: None,
            ..Default::default()
        }
    }

    fn bundle_fixture() -> (VerificationBundle, DspGraph) {
        let (_, baseline) = fingerprint_graph(&["left"]);
        let (mut graph, _) = fingerprint_graph(&["left", "lfe"]);
        graph
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .push(roomeq_model::PluginConfigWrapper {
                plugin_type: String::from("convolution"),
                parameters: serde_json::json!({"ir_file": "left_fir.wav"}),
            });
        graph
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .push(roomeq_model::PluginConfigWrapper {
                plugin_type: String::from("delay"),
                parameters: serde_json::json!({"delay_ms": 0.35}),
            });
        let candidate = crate::final_ledger::canonical_graph_identity(&graph);
        let bundle = VerificationBundle::build(
            &baseline,
            &candidate,
            &graph,
            vec![String::from("left"), String::from("lfe")],
            vec![String::from("seat-a"), String::from("seat-held")],
            vec![String::from("seat-held")],
            48_000.0,
            String::from("cal-1"),
            String::from("stim-1"),
            String::from("capture-v2"),
            TrialLevel::SmallSignal,
            vec![BundleResource {
                resource_id: String::from("left_fir.wav"),
                taps: 1024,
                content_hash: String::from("hash-fir-1"),
            }],
        )
        .unwrap();
        (bundle, graph)
    }

    fn binding_fixture(graph_fingerprint: &str) -> PlaybackBinding {
        PlaybackBinding {
            graph_id: graph_fingerprint.to_string(),
            source_id: String::from("left"),
            seat_id: String::from("seat-a"),
            stimulus_hash: String::from("stim-1"),
            sample_rate_hz: 48_000.0,
            calibration_id: String::from("cal-1"),
            processing_state: String::from("final-delivered"),
        }
    }

    fn tolerances_fixture() -> (DeclaredAlignment, CaptureTolerances) {
        (
            DeclaredAlignment {
                gain_db: 0.0,
                delay_ms: 0.0,
            },
            CaptureTolerances {
                max_magnitude_deviation_db: 3.0,
                max_timing_error_ms: 1.0,
                max_output_loss_db: 1.0,
            },
        )
    }

    #[test]
    fn workflow_verification_bundle_matches_final_graph() {
        let (bundle, graph) = bundle_fixture();
        let candidate = crate::final_ledger::canonical_graph_identity(&graph);
        assert_eq!(bundle.candidate_graph, candidate.fingerprint);
        assert_ne!(bundle.baseline_graph, bundle.candidate_graph);
        // Routes cover isolated, overlap, L+R, LFE, and held-out seats.
        let routes = plan_routes(
            &bundle.manifest.source_ids,
            &[(String::from("left"), String::from("lfe"))],
            Some("lfe"),
            &bundle.manifest.held_out_seats,
            10.0,
        )
        .unwrap();
        assert!(
            routes
                .iter()
                .any(|route| matches!(route.route, VerificationRoute::Isolated { .. }))
        );
        assert!(
            routes
                .iter()
                .any(|route| matches!(route.route, VerificationRoute::Overlap { .. }))
        );
        assert!(
            routes
                .iter()
                .any(|route| matches!(route.route, VerificationRoute::CoherentLrBass))
        );
        assert!(
            routes
                .iter()
                .any(|route| matches!(route.route, VerificationRoute::Lfe { .. }))
        );
        assert!(
            routes
                .iter()
                .any(|route| matches!(route.route, VerificationRoute::HeldOut { .. }))
        );
        // Bundle JSON round-trips with identities intact.
        let json = serde_json::to_value(&bundle).unwrap();
        let back: VerificationBundle = serde_json::from_value(json).unwrap();
        assert_eq!(back, bundle);
        // The K5 descriptor is assessable only with all identities present.
        let comparison = bundle.comparison_for("left", "seat-a", CaptureKind::StationaryIr);
        assert!(comparison.is_assessed());
        assert!(comparison.unassessed_reason().is_none());
        let missing_hash = PlaybackComparison {
            stimulus_hash: None,
            ..comparison
        };
        assert!(!missing_hash.is_assessed());
    }

    #[test]
    fn workflow_capture_manifest_mismatch_not_accepted() {
        // F13: valid but mismatched graph/capture/stimulus hashes stay
        // unassessed. No promotion, no scored comparison.
        let (bundle, _) = bundle_fixture();
        let expected = binding_fixture(&bundle.candidate_graph);
        let mut prediction = flat_curve(20.0, 20_000.0, 64, 80.0);
        prediction.phase = Some(prediction.freq.mapv(|_| 0.0));
        let (declared, tolerances) = tolerances_fixture();
        for (label, mutate) in [
            (
                "graph",
                Box::new(|binding: &mut PlaybackBinding| {
                    binding.graph_id = String::from("graph-other");
                }) as Box<dyn Fn(&mut PlaybackBinding)>,
            ),
            (
                "stimulus",
                Box::new(|binding: &mut PlaybackBinding| {
                    binding.stimulus_hash = String::from("stim-other");
                }),
            ),
            (
                "seat",
                Box::new(|binding: &mut PlaybackBinding| {
                    binding.seat_id = String::from("seat-other");
                }),
            ),
            (
                "calibration",
                Box::new(|binding: &mut PlaybackBinding| {
                    binding.calibration_id = String::from("cal-other");
                }),
            ),
        ] {
            let mut binding = expected.clone();
            mutate(&mut binding);
            let capture = ImportedCapture {
                binding,
                curve: flat_curve(20.0, 20_000.0, 64, 80.0),
                evidence_kind: PlaybackEvidenceKind::Acoustic,
            };
            let assessment = assess_imported_capture(
                &expected,
                Some(&capture),
                &prediction,
                &declared,
                &tolerances,
                [40.0, 4000.0],
            )
            .unwrap();
            match &assessment {
                CaptureAssessment::Unassessed { reason } => {
                    assert!(reason.contains("mismatch"), "{label}: {reason}");
                }
                CaptureAssessment::Assessed(_) => {
                    panic!("{label} mismatch must not reach a scored comparison")
                }
            }
            assert!(!assessment.verifies_room(), "{label}");
            assert_eq!(
                update_playback_status(
                    PlaybackStatus::Unassessed,
                    &assessment,
                    PlaybackEvidenceKind::Acoustic
                ),
                PlaybackStatus::InsufficientEvidence,
                "{label}"
            );
        }
        // A fully bound acoustic capture that passes V2 verifies the room;
        // the identical backend-rendered pass does not.
        let capture = ImportedCapture {
            binding: expected.clone(),
            curve: prediction.clone(),
            evidence_kind: PlaybackEvidenceKind::Acoustic,
        };
        let assessment = assess_imported_capture(
            &expected,
            Some(&capture),
            &prediction,
            &declared,
            &tolerances,
            [40.0, 4000.0],
        )
        .unwrap();
        assert!(matches!(assessment, CaptureAssessment::Assessed(_)));
        assert!(assessment.verifies_room());
        assert_eq!(
            update_playback_status(
                PlaybackStatus::Unassessed,
                &assessment,
                PlaybackEvidenceKind::Acoustic
            ),
            PlaybackStatus::Verified
        );
        let rendered = ImportedCapture {
            evidence_kind: PlaybackEvidenceKind::BackendRendered,
            ..capture
        };
        let rendered_assessment = assess_imported_capture(
            &expected,
            Some(&rendered),
            &prediction,
            &declared,
            &tolerances,
            [40.0, 4000.0],
        )
        .unwrap();
        assert!(!rendered_assessment.verifies_room());
    }

    #[test]
    fn workflow_missing_capture_stays_unassessed() {
        let (bundle, _) = bundle_fixture();
        let expected = binding_fixture(&bundle.candidate_graph);
        let prediction = flat_curve(20.0, 20_000.0, 64, 80.0);
        let (declared, tolerances) = tolerances_fixture();
        let assessment = assess_imported_capture(
            &expected,
            None,
            &prediction,
            &declared,
            &tolerances,
            [40.0, 4000.0],
        )
        .unwrap();
        assert!(matches!(assessment, CaptureAssessment::Unassessed { .. }));
        assert!(!assessment.verifies_room());
        assert_eq!(
            update_playback_status(
                PlaybackStatus::Unassessed,
                &assessment,
                PlaybackEvidenceKind::Acoustic
            ),
            PlaybackStatus::InsufficientEvidence
        );
    }

    #[test]
    fn roadmap_correction_verification_cascaded_delays() {
        let (bundle, mut graph) = bundle_fixture();
        graph
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .push(roomeq_model::PluginConfigWrapper {
                plugin_type: "delay".into(),
                parameters: serde_json::json!({"delay_ms": 0.65}),
            });
        let identity = crate::final_ledger::canonical_graph_identity(&graph);
        let rebuilt = VerificationBundle::build(
            &identity,
            &identity,
            &graph,
            bundle.manifest.source_ids.clone(),
            bundle.manifest.seat_ids.clone(),
            bundle.manifest.held_out_seats.clone(),
            bundle.manifest.sample_rate_hz,
            bundle.manifest.calibration_id.clone(),
            bundle.manifest.stimulus_hash.clone(),
            bundle.manifest.comparison_policy_version.clone(),
            bundle.trial_level,
            bundle.resources.clone(),
        )
        .unwrap();
        assert_eq!(
            rebuilt
                .channels
                .iter()
                .find(|c| c.channel == "left")
                .unwrap()
                .delay_ms,
            Some(1.0)
        );
        assert!(verify_fir_tails_and_delays(&rebuilt, &graph, &[("left".into(), 1.0)]).is_ok());
        assert!(verify_fir_tails_and_delays(&rebuilt, &graph, &[("left".into(), 0.35)]).is_err());
        // An omitted external ledger must not hide a changed snapshot delay.
        assert!(verify_fir_tails_and_delays(&bundle, &graph, &[]).is_err());
    }

    #[test]
    fn roadmap_correction_verification_invalid_delay_is_not_absent() {
        for parameters in [
            serde_json::json!({}),
            serde_json::json!({"delay_ms": "1"}),
            serde_json::json!({"delay_ms": -1.0}),
            serde_json::json!({"delay_ms": null}),
        ] {
            let (bundle, mut graph) = bundle_fixture();
            graph.channels.get_mut("left").unwrap().plugins.push(
                roomeq_model::PluginConfigWrapper {
                    plugin_type: "delay".into(),
                    parameters,
                },
            );
            assert!(verify_fir_tails_and_delays(&bundle, &graph, &[]).is_err());
        }
        let delay = roomeq_model::PluginConfigWrapper {
            plugin_type: "delay".into(),
            parameters: serde_json::json!({"delay_ms": f64::MAX}),
        };
        assert!(channel_delay_ms(&[delay.clone(), delay]).is_err());
        let split = roomeq_model::PluginConfigWrapper {
            plugin_type: "band_split".into(),
            parameters: serde_json::json!({}),
        };
        let delay = roomeq_model::PluginConfigWrapper {
            plugin_type: "delay".into(),
            parameters: serde_json::json!({"delay_ms": 1.0}),
        };
        assert!(channel_delay_ms(&[split, delay]).is_err());
    }

    #[test]
    fn roadmap_correction_verification_unknown_delay_channel() {
        let (bundle, graph) = bundle_fixture();
        assert!(verify_fir_tails_and_delays(&bundle, &graph, &[("missing".into(), 1.0)]).is_err());
    }

    #[test]
    fn workflow_verification_preserves_fir_tail_and_delay() {
        let (bundle, graph) = bundle_fixture();
        let delay_ledger = vec![(String::from("left"), 0.35)];
        assert!(verify_fir_tails_and_delays(&bundle, &graph, &delay_ledger).is_ok());
        // A truncated FIR tail is detected, never silently accepted.
        let mut short_resources = bundle.clone();
        short_resources.resources[0].taps = 512;
        assert!(verify_fir_tails_and_delays(&short_resources, &graph, &delay_ledger).is_err());
        // A dropped convolution plugin is detected.
        let mut dropped = graph.clone();
        dropped
            .channels
            .get_mut("left")
            .unwrap()
            .plugins
            .retain(|plugin| {
                plugin
                    .parameters
                    .get("ir_file")
                    .and_then(|value| value.as_str())
                    != Some("left_fir.wav")
            });
        assert!(verify_fir_tails_and_delays(&bundle, &dropped, &delay_ledger).is_err());
        // A re-derived delay is detected.
        let drifted_ledger = vec![(String::from("left"), 0.5)];
        assert!(verify_fir_tails_and_delays(&bundle, &graph, &drifted_ledger).is_err());
        // A lost channel is detected.
        let mut lost = graph.clone();
        lost.channels.remove("left");
        assert!(verify_fir_tails_and_delays(&bundle, &lost, &delay_ledger).is_err());
    }

    #[test]
    fn workflow_no_double_lfe_gain_in_verification_routes() {
        let sources = vec![String::from("left"), String::from("right")];
        let routes = plan_routes(
            &sources,
            &[(String::from("left"), String::from("sub"))],
            Some("lfe"),
            &[String::from("seat-held")],
            10.0,
        )
        .unwrap();
        // Exactly one LFE gain application across all routes.
        assert_eq!(count_lfe_gain_stages(&routes, "lfe"), 1);
        let lfe_routes: Vec<_> = routes
            .iter()
            .filter(|route| matches!(route.route, VerificationRoute::Lfe { .. }))
            .collect();
        assert_eq!(lfe_routes.len(), 1);
        assert_eq!(
            lfe_routes[0].gain_stages_db,
            vec![(String::from("lfe"), 10.0)]
        );
        // Isolated and coherent routes carry no LFE gain.
        for route in &routes {
            if !matches!(route.route, VerificationRoute::Lfe { .. }) {
                assert!(
                    route
                        .gain_stages_db
                        .iter()
                        .all(|(channel, _)| channel != "lfe"),
                    "non-LFE route must not re-apply LFE gain: {:?}",
                    route.route
                );
            }
        }
        assert!(plan_routes(&sources, &[], Some("lfe"), &[], f64::NAN).is_err());
    }
}
