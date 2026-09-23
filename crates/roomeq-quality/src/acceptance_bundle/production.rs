//! Acceptance diagnostics derived from retained finalized response data.
//!
//! Retained channel curves are predictions, not raw acquisition or independently
//! replayed final routes. Their original conditioning is not undone. Reconstructed
//! display IRs are deliberately not promoted into common-reference captures.

use super::{AcceptanceBundle, BassDetail, MagnitudeView, ViewProvenance, ViewSettings};
use roomeq_model::acceptance_evidence::AcceptanceEvidence;
use roomeq_model::decision_ledger::canonical_value_identity;
use roomeq_model::{CurveData, DspGraph};
use std::collections::BTreeMap;

fn supported_pair(pre: &CurveData, post: &CurveData, sample_rate_hz: f64) -> bool {
    pre.freq.len() >= 2
        && pre.freq == post.freq
        && pre.freq.len() == pre.spl.len()
        && post.freq.len() == post.spl.len()
        && pre.norm_range == post.norm_range
        && pre
            .freq
            .iter()
            .all(|f| f.is_finite() && *f > 0.0 && *f <= sample_rate_hz / 2.0)
        && pre.freq.windows(2).all(|pair| pair[0] < pair[1])
        && pre
            .spl
            .iter()
            .chain(&post.spl)
            .all(|value| value.is_finite())
}

fn magnitude(
    pre: &CurveData,
    post: &CurveData,
    settings: &ViewSettings,
    provenance: &ViewProvenance,
    bands_per_octave: usize,
) -> MagnitudeView {
    let pre = autoeq_core::curve_transforms::smooth_one_over_n_octave(
        &pre.clone().into(),
        bands_per_octave,
    );
    let post = autoeq_core::curve_transforms::smooth_one_over_n_octave(
        &post.clone().into(),
        bands_per_octave,
    );
    MagnitudeView {
        provenance: provenance.clone(),
        settings_hash: settings.settings_hash(),
        freqs: pre.freq.to_vec(),
        pre_db: pre.spl.to_vec(),
        post_db: post.spl.to_vec(),
    }
}

/// Build bound diagnostics from the actual serialized workflow output.
///
/// Uses canonical quality bundles and existing smoothing kernels. Unsupported
/// acquisition-dependent views remain absent with explicit reasons. The returned
/// digest is an integrity check, never an acoustic or physical-headroom verdict.
///
/// # Errors
/// Returns a reason for invalid sample rate or an unserializable graph/payload.
pub fn graph_acceptance_evidence(
    graph: &DspGraph,
    sample_rate_hz: f64,
) -> Result<AcceptanceEvidence, String> {
    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err("acceptance diagnostics require a finite positive sample rate".into());
    }
    let mut graph_value = serde_json::to_value(graph).map_err(|e| e.to_string())?;
    graph_value
        .as_object_mut()
        .ok_or("graph is not an object")?
        .remove("correction_decisions");
    let identity = canonical_value_identity(&graph_value);
    let mut channels = BTreeMap::new();
    for (name, channel) in &graph.channels {
        let mut unavailable = BTreeMap::from([
            ("ir_step", "raw, common-reference pre/post captures are not retained in this output; display reconstructions are not capture evidence".to_owned()),
            ("etc", "octave ETC needs raw common-reference IRs and a documented acquisition window".to_owned()),
            ("decay", "frequency-resolved decay needs raw IRs, noise support, and fit evidence".to_owned()),
            ("ambient_noise", "no calibrated silent-playback noise capture is retained".to_owned()),
            ("headroom", "no complete routed true-peak trial and calibrated physical demand/limits are retained".to_owned()),
        ]);
        let mut settings = ViewSettings {
            sample_rate_hz,
            bass_detail: BassDetail::Fine24,
            normalization: "retained channel response levels; no additional level matching; absolute SPL unverified".into(),
            window: "retained magnitude data; raw acquisition window unavailable".into(),
            ..ViewSettings::default()
        };
        let mut bundle = AcceptanceBundle::default();
        let mut fine_magnitude = None;
        if let (Some(pre), Some(post)) = (&channel.initial_curve, &channel.final_curve)
            && supported_pair(pre, post, sample_rate_hz)
        {
            settings.freq_limits_hz = [pre.freq[0], pre.freq[pre.freq.len() - 1]];
            let source = serde_json::to_value(pre).map_err(|e| e.to_string())?;
            let provenance = ViewProvenance {
                measurement_ids: vec![format!(
                    "retained-curve:{}",
                    canonical_value_identity(&source).fingerprint
                )],
                graph_identity: identity.fingerprint.clone(),
                sample_rate_hz,
                calibration: "uncalibrated".into(),
                processing_chain: format!(
                    "retained-prediction:{name}; existing conditioning retained; additional log-frequency dB smoothing"
                ),
            };
            bundle.magnitude = Some(magnitude(pre, post, &settings, &provenance, 12));
            fine_magnitude = Some(magnitude(pre, post, &settings, &provenance, 24));
        } else {
            unavailable.insert("magnitude", "missing or invalid matched retained pre/post frequency grids/levels/normalization; no interpolation or calibration is invented".into());
        }
        let disposition_provenance = ViewProvenance {
            measurement_ids: vec![format!("serialized-graph:{}", identity.fingerprint)],
            graph_identity: identity.fingerprint.clone(),
            sample_rate_hz,
            calibration: "uncalibrated".into(),
            processing_chain: "serialized DSP controls; not measured physical output".into(),
        };
        match super::derive_dsp_dispositions(
            graph,
            disposition_provenance,
            settings.settings_hash(),
        ) {
            Ok(disposition) => bundle.disposition = Some(disposition),
            Err(reason) => {
                unavailable.insert("disposition", reason);
            }
        }
        bundle.settings = settings;
        let gate = super::evaluate_bundle_gate(&bundle);
        channels.insert(name.clone(), serde_json::json!({
            "evidence_kind": "retained_prediction",
            "source_id": name,
            "seat_ids": [],
            "seat_provenance": "unavailable; channel response is not an identified individual-seat capture",
            "bundle": bundle,
            "fine_magnitude": fine_magnitude,
            "fine_magnitude_bands_per_octave": 24,
            "unavailable": unavailable,
            "internal_gate": {"passed": gate.passed, "missing_views": gate.missing_views, "failures": gate.failures},
        }));
    }
    Ok(AcceptanceEvidence::new(
        serde_json::json!({
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
            "scope": "retained per-channel predictions and serialized DSP controls; not independent routed replay or acoustic verification",
        }),
        &identity.fingerprint,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> DspGraph {
        let mut graph = DspGraph::new("1");
        graph.add_channel("L", Vec::new());
        let channel = graph.channels.get_mut("L").unwrap();
        let curve = |level| CurveData {
            freq: vec![100.0, 200.0, 400.0, 800.0],
            spl: vec![level; 4],
            phase: None,
            norm_range: None,
            ..Default::default()
        };
        channel.initial_curve = Some(curve(4.0));
        channel.final_curve = Some(curve(7.0));
        graph
    }

    #[test]
    fn roadmap_correction_production_views_preserve_levels_and_unknown_capture_evidence() {
        let graph = graph();
        let evidence = graph_acceptance_evidence(&graph, 48_000.0).unwrap();
        assert!(evidence.matches(&evidence.binding.graph_identity));
        let channel = &evidence.payload["channels"]["L"];
        let bundle: AcceptanceBundle = serde_json::from_value(channel["bundle"].clone()).unwrap();
        assert_eq!(bundle.magnitude.as_ref().unwrap().pre_db, vec![4.0; 4]);
        assert_eq!(bundle.magnitude.as_ref().unwrap().post_db, vec![7.0; 4]);
        assert_eq!(
            channel["fine_magnitude"]["post_db"],
            serde_json::json!(vec![7.0; 4])
        );
        assert!(!super::super::evaluate_bundle_gate(&bundle).passed);
        assert!(bundle.ir_step.is_none() && bundle.headroom.is_none());
        assert!(
            !channel["unavailable"]["headroom"]
                .as_str()
                .unwrap()
                .is_empty()
        );
        assert!(channel["seat_ids"].as_array().unwrap().is_empty());
    }

    #[test]
    fn roadmap_correction_production_views_refuse_incompatible_retained_curves() {
        for case in 0..4 {
            let mut graph = graph();
            let post = graph
                .channels
                .get_mut("L")
                .unwrap()
                .final_curve
                .as_mut()
                .unwrap();
            match case {
                0 => post.freq[1] = 210.0,
                1 => post.spl.pop().map(|_| ()).unwrap(),
                2 => post.norm_range = Some((100.0, 800.0)),
                _ => post.spl[0] = f64::NAN,
            }
            let evidence = graph_acceptance_evidence(&graph, 48_000.0).unwrap();
            let channel = &evidence.payload["channels"]["L"];
            assert!(channel["bundle"]["magnitude"].is_null());
            assert!(channel["fine_magnitude"].is_null());
            assert!(channel["unavailable"]["magnitude"].is_string());
        }
    }
}
