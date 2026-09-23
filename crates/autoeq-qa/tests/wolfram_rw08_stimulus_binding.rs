//! Wolfram cross-check: listening-stimulus binding and route audit (RW08).
//!
//! Oracle: `wolfram/rw08_stimulus_binding.wls` (engine SHA-256 over the
//! rendered bytes, independent routing-label table, independent LFE
//! stage count). The Rust test builds the render spec from the golden
//! inputs and calls the real `roomeq_workflow::listening_stimuli` and
//! `verification` paths. Exact identity on hash, labels, and counts
//! (class I/Q).

use autoeq_qa::require_reference;
use autoeq_qa::{QaResult, assert_case_id, emit_result, provenance};
use roomeq_quality::SourcePresentation;
use roomeq_workflow::listening_stimuli::{
    ListeningRenderSpec, StimulusLevelMatch, bind_rendered_stimulus, check_binding_current,
    route_label,
};
use roomeq_workflow::verification::{PlannedRoute, VerificationRoute, count_lfe_gain_stages};

const CASE: &str = "rw08_stimulus_binding";
const CASE_ID: &str = "autoeq-qa.rw08-stimulus-binding.v1";

fn presentation(name: &str) -> SourcePresentation {
    match name {
        "SingleSpeakerMono" => SourcePresentation::SingleSpeakerMono,
        "IdenticalLrSum" => SourcePresentation::IdenticalLrSum,
        "Spatial" => SourcePresentation::Spatial,
        other => panic!("{CASE}: unknown presentation {other}"),
    }
}

#[test]
fn wolfram_rw08_stimulus_binding() {
    let ref_json = require_reference(CASE, "rw08_stimulus_binding.wls");
    assert_case_id(&ref_json, CASE_ID, CASE);
    let str_field = |key: &str| ref_json[key].as_str().unwrap().to_string();
    let pcm: Vec<u8> = ref_json["pcm_bytes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    assert_eq!(pcm.len(), 16, "{CASE}: expected 16 rendered PCM bytes");

    let spec = ListeningRenderSpec {
        baseline_graph_id: str_field("baseline_graph_id"),
        candidate_graph_id: str_field("candidate_graph_id"),
        full_graph_id: str_field("full_graph_id"),
        pruned_graph_id: Some(str_field("pruned_graph_id")),
        programme_id: str_field("programme_id"),
        programme_wav_hash: str_field("programme_wav_hash"),
        sample_rate_hz: serde_json::from_value(ref_json["sample_rate_hz"].clone()).unwrap(),
        calibration_id: str_field("calibration"),
        processing_state: str_field("processing_state"),
        presentation: presentation(str_field("presentation").as_str()),
        level_match: StimulusLevelMatch {
            matching_method: str_field("matching_method"),
            matched_within_db: serde_json::from_value(ref_json["matched_within_db"].clone())
                .unwrap(),
            absolute_level_db_spl: serde_json::from_value(
                ref_json["absolute_level_db_spl"].clone(),
            )
            .unwrap(),
            calibration_id: str_field("calibration"),
        },
    };
    let binding = bind_rendered_stimulus(&spec, &pcm).expect("valid spec must bind");
    assert_eq!(
        binding.stimulus_hash,
        ref_json["stimulus_hash"].as_str().unwrap(),
        "{CASE}: stimulus SHA-256 mismatch"
    );
    assert!(
        binding.sample_rate_hz.is_finite() && binding.sample_rate_hz > 0.0,
        "{CASE}: bound sample rate must stay positive"
    );
    check_binding_current(&spec, &binding, &pcm).expect("fresh binding must verify");

    let presentations: Vec<String> =
        serde_json::from_value(ref_json["presentations"].clone()).unwrap();
    let labels: Vec<String> = serde_json::from_value(ref_json["route_labels"].clone()).unwrap();
    assert_eq!(presentations.len(), 3, "{CASE}: expected 3 presentations");
    for (name, expected) in presentations.iter().zip(labels.iter()) {
        assert_eq!(
            route_label(presentation(name)),
            expected.as_str(),
            "{CASE}: route label mismatch for {name}"
        );
    }

    let mut planned = Vec::new();
    for entry in ref_json["routes"].as_array().unwrap() {
        let route = match entry["route"].as_str().unwrap() {
            "Isolated:L" => VerificationRoute::Isolated {
                source: "L".to_string(),
            },
            "Lfe:sub" => VerificationRoute::Lfe {
                channel: "sub".to_string(),
            },
            other => panic!("{CASE}: unknown route {other}"),
        };
        let gains: Vec<(String, f64)> = entry["gains"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                (
                    pair[0].as_str().unwrap().to_string(),
                    pair[1].as_f64().unwrap(),
                )
            })
            .collect();
        planned.push(PlannedRoute {
            route,
            gain_stages_db: gains,
        });
    }
    let lfe = str_field("lfe_channel");
    let expected: usize = serde_json::from_value(ref_json["expected_lfe_stages"].clone()).unwrap();
    assert_eq!(
        count_lfe_gain_stages(&planned, &lfe),
        expected,
        "{CASE}: LFE gain-stage count mismatch"
    );
    emit_result(&QaResult {
        case: CASE_ID.to_string(),
        pass: true,
        max_rel_error: 0.0,
        max_abs_error: 0.0,
        tolerance: 0.0,
        tolerance_kind: "abs".to_string(),
        provenance: provenance(),
    });
}
