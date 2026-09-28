//! Apply attenuation budgets to mains while retaining unlimited subwoofer cuts.

// Rust guideline compliant 2026-02-21
use super::*;
use std::collections::{BTreeMap, BTreeSet};

fn sub_outputs(result: &RoomOptimizationResult, config: &RoomConfig) -> BTreeSet<String> {
    // Routing owns physical output identity. Fall back to the existing channel
    // classifier only for configurations without a routed graph.
    if result
        .metadata
        .bass_management
        .as_ref()
        .and_then(|bass| bass.routing_graph.as_ref())
        .is_some()
    {
        return sub_output_limiter::sub_outputs(result);
    }
    crate::electrical_headroom::independent_graph_output_ports(&result.channels)
        .into_iter()
        .filter(|((channel, _), _)| super::super::misc::is_subwoofer_channel(config, channel))
        .map(|(_, output)| output)
        .collect()
}

pub(super) fn sub_requirements(
    result: &RoomOptimizationResult,
    config: &RoomConfig,
    required: &BTreeMap<String, f64>,
) -> BTreeMap<String, f64> {
    let subs = sub_outputs(result, config);
    required
        .iter()
        .filter(|(name, cut)| {
            subs.contains(*name) && **cut > config.optimizer.finalization.max_attenuation_db + 1e-6
        })
        .map(|(name, value)| (name.clone(), *value))
        .collect()
}

pub(super) fn check(
    result: &RoomOptimizationResult,
    config: &RoomConfig,
    required: &BTreeMap<String, f64>,
    common_cut: Option<f64>,
    context: &str,
) -> Result<()> {
    let subs = sub_outputs(result, config);
    let limit = config.optimizer.finalization.max_attenuation_db;
    let worst = required
        .iter()
        .filter(|(name, _)| !subs.contains(*name))
        .map(|(name, cut)| (name, common_cut.unwrap_or(*cut)))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((name, cut)) = worst
        && cut > limit + 1e-6
    {
        let details = required
            .iter()
            .map(|(name, cut)| {
                let role = if subs.contains(name) {
                    "subwoofer, exempt"
                } else {
                    "main"
                };
                format!("{name}={cut:.3} dB ({role})")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let treatment = if common_cut.is_some() {
            "common attenuation"
        } else {
            "attenuation"
        };
        return Err(failed(format!(
            "{context} needs {cut:.3} dB {treatment} on non-subwoofer output '{name}' beyond {limit:.3} dB limit (required physical-output cuts: {details})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_cut_is_unbounded_but_main_and_common_cuts_are_bounded() {
        let (result, config) = super::super::tests::implicit_lfe_fixture();
        let required = BTreeMap::from([("L".into(), 0.0), ("Sub1".into(), 30.0)]);
        check(&result, &config, &required, None, "candidate").unwrap();
        assert_eq!(
            sub_requirements(&result, &config, &required),
            BTreeMap::from([("Sub1".into(), 30.0)])
        );
        assert!(
            sub_requirements(&result, &config, &BTreeMap::from([("Sub1".into(), 6.0)])).is_empty()
        );
        let error = check(&result, &config, &required, Some(30.0), "candidate")
            .unwrap_err()
            .to_string();
        assert!(error.contains("output 'L'"));
        assert!(error.contains("Sub1=30.000 dB (subwoofer, exempt)"));
        let required = BTreeMap::from([("L".into(), 13.0), ("Sub1".into(), 30.0)]);
        assert!(check(&result, &config, &required, None, "baseline").is_err());
    }

    #[test]
    fn structural_fallback_allows_large_sub_cut_and_rechecks_ceiling() {
        let (mut result, config) = super::super::tests::implicit_lfe_fixture();
        result
            .channels
            .get_mut("Sub1")
            .unwrap()
            .plugins
            .push(PluginConfigWrapper {
                plugin_type: "gain".into(),
                parameters: serde_json::json!({"gain_db": 30.0, "room_eq_stage": "post_route"}),
            });
        let dir = tempfile::tempdir().unwrap();
        let captures = seat_replay::capture_training(&config).unwrap();
        rebuild(&mut result, &config, &HashMap::new(), 48_000.0, dir.path()).unwrap();
        publish_baseline(
            &mut result,
            &captures,
            &HashMap::new(),
            &config,
            48_000.0,
            dir.path(),
            &autoeq_artifacts::FsArtifactStore::new(),
            "no_candidate_within_electrical_acoustic_limits",
            Vec::new(),
        )
        .unwrap();
        let outputs = crate::electrical_headroom::assess_final_graph(
            &result.to_dsp_chain_output(),
            48_000.0,
            dir.path(),
            &config.optimizer.finalization,
        )
        .unwrap();
        assert!(
            outputs
                .iter()
                .all(|output| output.peak_dbfs.unwrap() <= 1e-6)
        );
        assert!(
            result
                .metadata
                .stage_outcomes
                .iter()
                .any(|stage| stage.stage == "final_correction_selection")
        );
    }

    #[test]
    fn sub_attenuation_preserves_main_transfer_and_enforces_output_ceiling() {
        let (mut result, config) = super::super::tests::implicit_lfe_fixture();
        result
            .channels
            .get_mut("Sub1")
            .unwrap()
            .plugins
            .push(PluginConfigWrapper {
                plugin_type: "gain".into(),
                parameters: serde_json::json!({"gain_db": 30.0, "room_eq_stage": "post_route"}),
            });
        let dir = tempfile::tempdir().unwrap();
        let assess = |result: &RoomOptimizationResult| {
            crate::electrical_headroom::assess_final_graph(
                &result.to_dsp_chain_output(),
                48_000.0,
                dir.path(),
                &config.optimizer.finalization,
            )
            .unwrap()
        };
        let before = assess(&result);
        let required = physical_drive::combined_attenuations(&before, &BTreeMap::new(), 0.0);
        assert!(required["Sub1"] > 12.0);
        check(&result, &config, &required, None, "candidate").unwrap();
        install_output_attenuation_requirements(&mut result, &required).unwrap();
        let after = assess(&result);
        assert!(after.iter().all(|output| output.peak_dbfs.unwrap() <= 1e-6));
        let main = |outputs: &[roomeq_engine::quality::electrical_headroom::SampledElectricalOutputPeak]| outputs.iter().find(|output| output.output == "L").unwrap().peak_amplitude;
        assert!((main(&before) - main(&after)).abs() < 1e-10);
    }
}
