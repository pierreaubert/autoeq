//! Plan X1 tests: canonical DSP identity binding.
//!
//! Parse-back success is not playback proof, and no identity here is a
//! listening-benefit claim. These tests pin the evidence binding only:
//! what processing a delivered report may claim, and what must fail.

use super::super::{
    ExportFormat, build_export_package, canonical_dsp_identity, package_fingerprint,
};
use super::make::{add_convolution, make_routed_bass_output, make_test_output, resource, test_wav};
use roomeq_model::decision_ledger::{
    CorrectionDecisionLedger, DECISION_LEDGER_VERSION, DecisionRecord, DecisionStatus,
};
use roomeq_model::{CurveData, DspGraph};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap};

fn bind_inventory(graph: &mut DspGraph, reference: &str, sha: String) {
    graph
        .metadata
        .as_mut()
        .expect("fixture has metadata")
        .final_convolution_sha256 = Some(BTreeMap::from([(reference.to_string(), Some(sha))]));
}

fn hybrid_graph() -> (DspGraph, Vec<super::super::ConvolutionResource>) {
    let mut graph = make_test_output();
    add_convolution(&mut graph, "left", "left.wav");
    let wav = test_wav(48_000, 1, 64);
    let current = resource("left.wav", wav);
    bind_inventory(&mut graph, "left.wav", current.sha256());
    (graph, vec![current])
}

fn set_left_gain(graph: &mut DspGraph, gain_db: f64) {
    let plugin = graph
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .iter_mut()
        .find(|plugin| plugin.plugin_type == "gain")
        .unwrap();
    plugin.parameters["gain_db"] = json!(gain_db);
}

#[test]
fn export_identity_changes_with_gain_delay_routing_or_ir() {
    let base = make_test_output();
    let base_id = canonical_dsp_identity(&base, 48_000.0, &[]).unwrap();
    assert_eq!(base_id.dsp_identity.len(), 64);
    assert!(base_id.convolution.is_empty());

    let mut gain = base.clone();
    set_left_gain(&mut gain, -1.5);
    assert_ne!(
        canonical_dsp_identity(&gain, 48_000.0, &[])
            .unwrap()
            .dsp_identity,
        base_id.dsp_identity,
        "a 1 dB gain change must change the DSP identity"
    );

    let mut delay = base.clone();
    let plugin = delay
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .iter_mut()
        .find(|plugin| plugin.plugin_type == "delay")
        .unwrap();
    plugin.parameters["delay_ms"] = json!(2.0);
    assert_ne!(
        canonical_dsp_identity(&delay, 48_000.0, &[])
            .unwrap()
            .dsp_identity,
        base_id.dsp_identity,
        "a delay change must change the DSP identity"
    );

    let mut routed = make_routed_bass_output();
    let routed_base = canonical_dsp_identity(&routed, 48_000.0, &[]).unwrap();
    let route = routed
        .metadata
        .as_mut()
        .unwrap()
        .bass_management
        .as_mut()
        .unwrap()
        .routing_graph
        .as_mut()
        .unwrap()
        .routes
        .get_mut(1)
        .unwrap();
    route.gain_db = 0.0;
    route.gain_linear = 1.0;
    route.matrix_gain = 1.0;
    assert_ne!(
        canonical_dsp_identity(&routed, 48_000.0, &[])
            .unwrap()
            .dsp_identity,
        routed_base.dsp_identity,
        "a route gain change must change the DSP identity"
    );

    let (graph, resources) = hybrid_graph();
    let graph_id = canonical_dsp_identity(&graph, 48_000.0, &resources).unwrap();
    assert_eq!(graph_id.convolution.len(), 1);
    assert_eq!(graph_id.convolution[0].reference, "left.wav");
    let mut replaced = resources[0].bytes.to_vec();
    replaced[44] = replaced[44].wrapping_add(1);
    let altered = vec![resource("left.wav", replaced)];
    // No final-inventory bypass here: the bytes genuinely differ, so the
    // content hash (and therefore the DSP identity) must differ as well.
    let mut unbound = graph.clone();
    unbound.metadata.as_mut().unwrap().final_convolution_sha256 = None;
    assert_ne!(
        canonical_dsp_identity(&unbound, 48_000.0, &altered)
            .unwrap()
            .dsp_identity,
        graph_id.dsp_identity,
        "changed IR bytes must change the DSP identity"
    );

    // Sample rate is DSP: the same graph at another rate is another identity.
    assert_ne!(
        canonical_dsp_identity(&base, 96_000.0, &[])
            .unwrap()
            .dsp_identity,
        base_id.dsp_identity,
    );
}

#[test]
fn export_identity_stable_under_metadata_and_map_order() {
    let base = make_test_output();
    let base_id = canonical_dsp_identity(&base, 48_000.0, &[]).unwrap();

    // Explanatory metadata is not DSP: scores, timestamps, algorithm names,
    // and the K4 ledger (which references the graph, so hashing it would be
    // circular) must not move the identity.
    let mut relabeled = base.clone();
    let metadata = relabeled.metadata.as_mut().unwrap();
    metadata.pre_score = 9.5;
    metadata.post_score = 8.5;
    metadata.algorithm = "renamed-algorithm".to_string();
    metadata.iterations = 7;
    metadata.timestamp = "2031-05-05T05:05:05Z".to_string();
    relabeled.correction_decisions = Some(CorrectionDecisionLedger {
        acceptance_evidence: None,
        payload_binding: None,
        ledger_version: DECISION_LEDGER_VERSION.to_string(),
        decisions: vec![DecisionRecord::example(DecisionStatus::Applied)],
        channel_summaries: Vec::new(),
    });
    relabeled.version = "9.9.9".to_string();
    assert_eq!(
        canonical_dsp_identity(&relabeled, 48_000.0, &[])
            .unwrap()
            .dsp_identity,
        base_id.dsp_identity,
        "explanatory metadata, ledger, and schema version must not move the DSP identity"
    );
    let mut relabeled_again = relabeled.clone();
    relabeled_again
        .correction_decisions
        .as_mut()
        .unwrap()
        .decisions
        .push(DecisionRecord::example(DecisionStatus::Constrained));
    assert_eq!(
        canonical_dsp_identity(&relabeled_again, 48_000.0, &[])
            .unwrap()
            .dsp_identity,
        base_id.dsp_identity,
        "ledger content must not move the DSP identity"
    );

    // Display curves are evidence, not delivered processing.
    let mut curved = base.clone();
    let curve = CurveData {
        freq: vec![100.0, 1000.0],
        spl: vec![80.0, 82.0],
        phase: None,
        norm_range: None,
        ..Default::default()
    };
    let chain = curved.channels.get_mut("left").unwrap();
    chain.initial_curve = Some(curve.clone());
    chain.final_curve = Some(curve.clone());
    chain.eq_response = Some(curve.clone());
    chain.target_curve = Some(curve);
    assert_eq!(
        canonical_dsp_identity(&curved, 48_000.0, &[])
            .unwrap()
            .dsp_identity,
        base_id.dsp_identity,
        "display curves must not move the DSP identity"
    );

    // Channel map iteration order must not leak into the hash.
    let mut reordered = base.clone();
    let mut channels = HashMap::new();
    for key in ["right", "left"] {
        channels.insert(key.to_string(), reordered.channels.remove(key).unwrap());
    }
    reordered.channels = channels;
    assert_eq!(
        canonical_dsp_identity(&reordered, 48_000.0, &[])
            .unwrap()
            .dsp_identity,
        base_id.dsp_identity,
        "channel map order must not move the DSP identity"
    );

    // JSON object key order inside plugin parameters must not matter.
    let mut key_order = base.clone();
    let mut first = serde_json::Map::new();
    first.insert("gain_db".to_string(), json!(-2.5));
    first.insert("invert".to_string(), json!(false));
    let mut second = serde_json::Map::new();
    second.insert("invert".to_string(), json!(false));
    second.insert("gain_db".to_string(), json!(-2.5));
    let left = key_order.channels.get_mut("left").unwrap();
    left.plugins[0].parameters = serde_json::Value::Object(first);
    let mut swapped = key_order.clone();
    swapped.channels.get_mut("left").unwrap().plugins[0].parameters =
        serde_json::Value::Object(second);
    assert_eq!(
        canonical_dsp_identity(&key_order, 48_000.0, &[])
            .unwrap()
            .dsp_identity,
        canonical_dsp_identity(&swapped, 48_000.0, &[])
            .unwrap()
            .dsp_identity,
        "parameter key order must not move the DSP identity"
    );
}

#[test]
fn export_missing_ir_cannot_claim_bound_package() {
    let mut graph = make_test_output();
    add_convolution(&mut graph, "left", "left.wav");

    let error = canonical_dsp_identity(&graph, 48_000.0, &[]).unwrap_err();
    assert!(error.to_string().contains("unavailable"), "{error}");
    let error = build_export_package(
        &graph,
        ExportFormat::CamillaDsp,
        std::path::Path::new("room.yaml"),
        48_000.0,
        &[],
        &BTreeSet::new(),
        &HashMap::new(),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("missing explicit convolution resource"),
        "{error}"
    );

    // A final inventory does not conjure missing bytes either.
    bind_inventory(&mut graph, "left.wav", "0".repeat(64));
    assert!(canonical_dsp_identity(&graph, 48_000.0, &[]).is_err());
    assert!(
        build_export_package(
            &graph,
            ExportFormat::CamillaDsp,
            std::path::Path::new("room.yaml"),
            48_000.0,
            &[],
            &BTreeSet::new(),
            &HashMap::new(),
        )
        .is_err()
    );
}

#[test]
fn export_stale_candidate_evidence_not_rebound_silently() {
    let (mut graph, committed) = hybrid_graph();
    let committed_sha = committed[0].sha256();
    let replacement = resource("left.wav", test_wav(48_000, 1, 128));
    assert_ne!(replacement.sha256(), committed_sha);

    // The final inventory binds the committed bytes: presenting replacement
    // bytes for the same reference must fail, never rebind.
    let error =
        canonical_dsp_identity(&graph, 48_000.0, std::slice::from_ref(&replacement)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("changed since workflow completion"),
        "{error}"
    );
    let error = build_export_package(
        &graph,
        ExportFormat::CamillaDsp,
        std::path::Path::new("room.yaml"),
        48_000.0,
        &[replacement],
        &BTreeSet::new(),
        &HashMap::new(),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("changed since workflow completion"),
        "{error}"
    );

    // An emptied inventory no longer matches the graph references.
    graph.metadata.as_mut().unwrap().final_convolution_sha256 = Some(BTreeMap::new());
    let error = build_export_package(
        &graph,
        ExportFormat::CamillaDsp,
        std::path::Path::new("room.yaml"),
        48_000.0,
        &committed,
        &BTreeSet::new(),
        &HashMap::new(),
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("inventory does not match"),
        "{error}"
    );

    // A null inventory member is unbound evidence, not a successful identity.
    graph.metadata.as_mut().unwrap().final_convolution_sha256 =
        Some(BTreeMap::from([("left.wav".to_string(), None)]));
    let error = build_export_package(
        &graph,
        ExportFormat::CamillaDsp,
        std::path::Path::new("room.yaml"),
        48_000.0,
        &committed,
        &BTreeSet::new(),
        &HashMap::new(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("unbound"), "{error}");

    // Positive control: the committed bytes bind, and the graph hash and the
    // package hash are distinguished in the public API.
    bind_inventory(&mut graph, "left.wav", committed_sha);
    let identity = canonical_dsp_identity(&graph, 48_000.0, &committed).unwrap();
    let package = build_export_package(
        &graph,
        ExportFormat::CamillaDsp,
        std::path::Path::new("room.yaml"),
        48_000.0,
        &committed,
        &BTreeSet::new(),
        &HashMap::new(),
    )
    .unwrap();
    package.validate_integrity().unwrap();
    let sidecar = package.member(std::path::Path::new("left.wav")).unwrap();
    assert_eq!(sidecar.sha256, committed[0].sha256());
    let fingerprint = package_fingerprint(&package);
    assert_eq!(fingerprint.len(), 64);
    assert_ne!(
        fingerprint, identity.dsp_identity,
        "DSP identity and package fingerprint must be distinct values"
    );
}
