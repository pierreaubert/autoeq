#[test]
fn test_export_camilladsp() {
    let output = make_test_output();
    let result = export_camilladsp(&output, 48000.0).unwrap();

    assert!(result.contains("samplerate: 48000"));
    assert!(result.contains("left_gain:"));
    assert!(result.contains("left_delay:"));
    assert!(result.contains("left_peq_0:"));
    assert!(result.contains("left_peq_1:"));
    assert!(result.contains("left_peq_2:"));
    assert!(result.contains("right_gain:"));
    assert!(result.contains("right_peq_0:"));
    assert!(result.contains("type: Biquad"));
    assert!(result.contains("type: Peaking"));
    assert!(result.contains("type: Highshelf"));
    assert!(result.contains("type: Gain"));
    assert!(result.contains("type: Delay"));
    assert!(result.contains("unit: samples"));
    assert!(result.contains("pipeline:"));
}

#[test]
fn camilladsp_package_retains_typed_backend_latency_contract() {
    let mut output = make_test_output();
    for chain in output.channels.values_mut() {
        for plugin in &mut chain.plugins {
            if plugin.plugin_type == "delay" {
                plugin.parameters["delay_ms"] = json!(0.0);
            }
        }
    }
    let plugin = output.channels.get_mut("left").unwrap().plugins.iter_mut()
        .find(|plugin| plugin.plugin_type == "delay").unwrap();
    plugin.parameters["delay_ms"] = json!(1.23456);
    let report = crate::camilladsp_delay_realization(&output, 48_000.0).unwrap();
    assert_eq!(report.serial_padding_samples, 5);
    assert_eq!(report.common_padding_samples, 5);
    assert_eq!(report.additional_latency_ms, 5.0 / 48.0);
    assert!(report.fractional_delay_present);
    assert_eq!(report.usable_band_upper_hz, 22_080.0);
    let package = crate::build_export_package(&output, ExportFormat::CamillaDsp,
        std::path::Path::new("room.yaml"), 48_000.0, &[],
        &std::collections::BTreeSet::new(), &std::collections::HashMap::new()).unwrap();
    let packaged = package.camilladsp_delay_realization().unwrap().unwrap();
    assert!((packaged.additional_latency_ms - report.additional_latency_ms).abs() < 1e-12);
    let normalized = crate::CamillaDspDelayRealization {
        additional_latency_ms: report.additional_latency_ms, ..packaged
    };
    assert_eq!(normalized, report);
    assert!(crate::camilladsp_delay_realization(&output, 48_000.5).is_err());
}

#[test]
fn test_export_camilladsp_routed_bass_management_graph() {
    let output = make_routed_bass_output();
    let result = export_camilladsp(&output, 48000.0).unwrap();

    assert!(result.contains("# Routed bass-management graph export"));
    assert!(result.contains("capture:\n    type: Stdin\n    channels: 3"));
    assert!(result.contains("playback:\n    type: Stdout\n    channels: 3"));
    assert!(result.contains("mixers:"));
    assert!(result.contains("  roomeq_route_matrix:"));
    assert!(result.contains("  roomeq_route_sum:"));
    assert!(result.contains("  name: roomeq_route_matrix\n  type: Mixer"));
    assert!(result.contains("  name: roomeq_route_sum\n  type: Mixer"));
    assert!(result.contains("route_0_L_to_L_crossover:"));
    assert!(result.contains("type: LinkwitzRileyHighpass"));
    assert!(result.contains("route_1_L_to_LFE_crossover:"));
    assert!(result.contains("type: LinkwitzRileyLowpass"));
    assert!(result.contains("route_1_L_to_LFE_delay:"));
    assert!(result.contains("delay: 120\n      unit: samples"));
    assert!(result.contains("gain: -6.000000"));
    assert!(result.contains("inverted: true"));
    assert!(result.contains("post_LFE_peq_0:"));
    assert!(result.contains("  - post_LFE_peq_0"));
}

#[test]
fn tool_contract_camilladsp_routed_export_can_be_validated_locally() {
    let output = make_routed_bass_output();
    let result = export_camilladsp(&output, 48000.0).unwrap();

    let route_matrix = result.find("  roomeq_route_matrix:").unwrap();
    let route_sum = result.find("  roomeq_route_sum:").unwrap();
    let pipeline = result.find("pipeline:").unwrap();
    assert!(route_matrix < route_sum);
    assert!(route_sum < pipeline);
    assert_eq!(result.matches("type: Mixer").count(), 2);
    assert_eq!(result.matches("type: BiquadCombo").count(), 5);
    assert!(result.contains("channels:\n      in: 3\n      out: 5"));
    assert!(result.contains("channels:\n      in: 5\n      out: 3"));

    run_optional_export_validator("ROOMEQ_CAMILLADSP_VALIDATE_CMD", "yaml", &result);
}

#[test]
fn test_export_camilladsp_writes_crossover_plugins() {
    let mut output = make_test_output();
    output
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "crossover".to_string(),
            parameters: json!({
                "type": "LR48",
                "frequency": 95.0,
                "output": "high",
            }),
        });

    let result = export_camilladsp(&output, 48000.0).unwrap();
    assert!(result.contains("left_crossover:"));
    assert!(result.contains("type: BiquadCombo"));
    assert!(result.contains("type: LinkwitzRileyHighpass"));
    assert!(result.contains("order: 8"));
    assert!(result.contains("  - left_crossover"));
}

#[test]
fn test_camilladsp_pipeline_uses_gui_friendly_steps() {
    let output = make_test_output();
    let result = export_camilladsp(&output, 48000.0).unwrap();

    assert!(
        result.contains("pipeline:\n- bypassed: null\n  channels:\n  - 0\n  names:\n  - left_gain"),
        "Expected pipeline entries to start with bypassed null, got:\n{result}"
    );
    assert!(
        result.contains("  type: Filter\n- bypassed: null"),
        "Expected type line inside the pipeline step, got:\n{result}"
    );
    assert!(
        !result.contains("  - type: Filter"),
        "Pipeline step should not start with a dashed type line"
    );
    assert!(result.contains("  - left_delay"));
    assert!(result.contains("  - left_peq_0"));
    assert!(result.contains("  - right_gain"));
}
