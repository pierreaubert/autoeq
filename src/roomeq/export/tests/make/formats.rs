#[test]
fn test_export_equalizer_apo() {
    let output = make_test_output();
    let result = export_equalizer_apo(&output).unwrap();

    assert!(result.contains("Channel: L"));
    assert!(result.contains("Channel: R"));
    assert!(result.contains("Preamp: -2.5 dB"));
    assert!(result.contains("Delay: 1.500 ms"));
    assert!(result.contains("Filter  1: ON PK Fc 100 Hz Gain -5.00 dB Q 2.0000"));
    assert!(result.contains("Filter  3: ON HSC Fc 8000 Hz Gain -2.00 dB Q 0.7000"));
    assert!(result.contains("Filter  1: ON PK Fc 200 Hz Gain -3.00 dB Q 1.0000"));
    assert!(result.contains("Filter  2: ON LSC Fc 80 Hz Gain +4.00 dB Q 0.7100"));
}

#[test]
fn tool_contract_equalizer_apo_text_has_channel_scoped_filters() {
    let output = make_test_output();
    let result = export_equalizer_apo(&output).unwrap();

    let mut current_channel = None;
    let mut left_filters = 0;
    let mut right_filters = 0;
    for line in result.lines() {
        if line == "Channel: L" {
            current_channel = Some("L");
        } else if line == "Channel: R" {
            current_channel = Some("R");
        } else if line.starts_with("Filter") {
            assert!(line.contains(" ON "));
            assert!(line.contains(" Fc "));
            match current_channel {
                Some("L") => left_filters += 1,
                Some("R") => right_filters += 1,
                _ => panic!("filter emitted before channel header: {line}"),
            }
        }
    }
    assert_eq!(left_filters, 3);
    assert_eq!(right_filters, 2);

    run_optional_export_validator("ROOMEQ_EQUALIZER_APO_VALIDATE_CMD", "txt", &result);
}

#[test]
fn equalizer_apo_routed_export_uses_channel_and_copy_for_supported_static_mix() {
    let mut output = make_routed_bass_output();
    let graph = output
        .metadata
        .as_mut()
        .unwrap()
        .bass_management
        .as_mut()
        .unwrap()
        .routing_graph
        .as_mut()
        .unwrap();
    // A low-pass mix with one route per source has no fan-out and can be
    // represented by APO's in-place Channel/Copy model.
    graph.routes.retain(|route| {
        route.destination == "LFE" && matches!(route.source_channel.as_str(), "L" | "R")
    });
    graph.input_channels = vec!["L".to_string(), "R".to_string()];
    graph.output_channels = vec!["LFE".to_string()];

    let result = export_equalizer_apo(&output).unwrap();
    assert!(result.contains("# Static RoomEQ routing graph (Channel/Copy)"));
    assert!(result.contains("Channel: L"));
    assert!(result.contains("Channel: R"));
    assert!(result.contains("Filter  1: ON LP Fc 80 Hz Q 0.7071"));
    assert!(result.contains("Copy:\n  LFE = 0.501187234*L + -0.501187234*R"));
    assert!(result.contains("Channel: LFE"));
    run_optional_export_validator("ROOMEQ_EQUALIZER_APO_VALIDATE_CMD", "txt", &result);
}

#[test]
fn equalizer_apo_routed_export_explains_unsupported_fan_out() {
    let output = make_routed_bass_output();
    let err = export_equalizer_apo(&output).unwrap_err();
    assert!(err.to_string().contains("cannot preserve fan-out"));
    assert!(err.to_string().contains("Use CamillaDSP or Apply as Graph"));
}

#[test]
fn equalizer_apo_routed_export_rejects_intermediate_destination_bus() {
    let mut output = make_routed_bass_output();
    let graph = output
        .metadata
        .as_mut()
        .unwrap()
        .bass_management
        .as_mut()
        .unwrap()
        .routing_graph
        .as_mut()
        .unwrap();
    graph.routes.retain(|route| {
        (route.destination == "LFE" && matches!(route.source_channel.as_str(), "L" | "R"))
            || (route.source_channel == "LFE" && route.destination == "LFE")
    });
    graph.input_channels = vec!["L".to_string(), "R".to_string(), "LFE".to_string()];
    graph.output_channels = vec!["LFE".to_string()];

    let error = export_equalizer_apo(&output).unwrap_err();

    assert!(error.to_string().contains("routed destination 'LFE'"));
    assert!(error.to_string().contains("intermediate bus"));
}

#[test]
fn test_export_easyeffects() {
    let output = make_test_output();
    let result = export_easyeffects(&output).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let eq = &parsed["output"]["equalizer#0"];
    assert_eq!(eq["num-bands"].as_u64().unwrap(), 5);
    assert!(eq["left"]["band0"]["frequency"].as_f64().unwrap() > 0.0);

    // Check filter types
    let band0_type = eq["left"]["band0"]["type"].as_str().unwrap();
    assert_eq!(band0_type, "Bell");
}

#[test]
fn tool_contract_easyeffects_json_has_mirrored_stereo_preset() {
    let output = make_test_output();
    let result = export_easyeffects(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let eq = &parsed["output"]["equalizer#0"];

    assert_eq!(eq["split-channels"], serde_json::json!(false));
    assert_eq!(eq["num-bands"].as_u64().unwrap(), 5);
    assert_eq!(eq["left"], eq["right"]);
    for band_idx in 0..eq["num-bands"].as_u64().unwrap() {
        let band = &eq["left"][format!("band{band_idx}")];
        assert!(band["frequency"].as_f64().unwrap().is_finite());
        assert!(band["q"].as_f64().unwrap().is_finite());
        assert!(band["gain"].as_f64().unwrap().is_finite());
        assert_eq!(band["solo"], serde_json::json!(false));
        assert_eq!(band["mute"], serde_json::json!(false));
    }

    run_optional_export_validator("ROOMEQ_EASYEFFECTS_VALIDATE_CMD", "json", &result);
}

#[test]
fn test_export_wavelet() {
    let output = make_test_output();
    let result = export_wavelet(&output, 48000.0).unwrap();

    assert!(result.contains("GraphicEQ:"));
    // Should have 9 frequency/gain pairs
    let line = result
        .lines()
        .find(|l| l.starts_with("GraphicEQ:"))
        .unwrap();
    let parts: Vec<&str> = line.trim_start_matches("GraphicEQ:").split(';').collect();
    assert_eq!(parts.len(), 9);
}

#[test]
fn tool_contract_wavelet_graphiceq_has_numeric_band_pairs() {
    let output = make_test_output();
    let result = export_wavelet(&output, 48000.0).unwrap();
    let line = result
        .lines()
        .find(|line| line.starts_with("GraphicEQ:"))
        .unwrap();
    let mut previous_freq = 0.0;
    for pair in line.trim_start_matches("GraphicEQ:").split(';') {
        let fields: Vec<_> = pair.split_whitespace().collect();
        assert_eq!(fields.len(), 2, "unexpected Wavelet band pair: {pair}");
        let freq: f64 = fields[0].parse().unwrap();
        let gain: f64 = fields[1].parse().unwrap();
        assert!(freq > previous_freq);
        assert!(gain.is_finite());
        previous_freq = freq;
    }

    run_optional_export_validator("ROOMEQ_WAVELET_VALIDATE_CMD", "txt", &result);
}

#[test]
fn test_export_pipewire() {
    let output = make_test_output();
    let result = export_pipewire(&output, 48000.0).unwrap();

    assert!(result.contains("libpipewire-module-filter-chain"));
    assert!(result.contains("bq_peaking"));
    assert!(result.contains("bq_highshelf"));
    assert!(result.contains("filter.graph"));
    assert!(result.contains("nodes ="));
    assert!(result.contains("links ="));
    assert!(result.contains("audio.channels = 2"));
    assert!(result.contains("\"FL\""));
    assert!(result.contains("\"FR\""));
}

#[test]
fn tool_contract_pipewire_filter_chain_has_nodes_links_and_positions() {
    let output = make_test_output();
    let result = export_pipewire(&output, 48000.0).unwrap();

    assert!(result.contains("filter.graph = {"));
    assert!(result.contains("nodes = ["));
    assert!(result.contains("links = ["));
    assert!(result.contains("inputs  = ["));
    assert!(result.contains("outputs = ["));
    assert!(result.contains("audio.position = [ \"FL\", \"FR\" ]"));
    assert!(result.contains("label = delay"));
    assert!(result.contains("config = { \"max-delay\" = 0.001500 }"));
    assert!(result.contains("control = { \"Delay (s)\" = 0.001500 }"));
    assert!(result.contains("label = bq_peaking"));
    assert!(result.contains("label = bq_highshelf"));
    assert!(result.contains("\"ch0_left_gain:In\""));
    assert!(result.contains("\"ch0_left_eq_2:Out\""));

    run_optional_export_validator("ROOMEQ_PIPEWIRE_VALIDATE_CMD", "conf", &result);
}

#[test]
fn test_export_roon() {
    let output = make_test_output();
    let result = export_roon(&output).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let channels = &parsed["channels"];

    // Left channel
    let left = &channels["left"];
    assert!(left["headroom_gain_db"].as_f64().unwrap() < 0.0);
    assert!((left["delay_ms"].as_f64().unwrap() - 1.5).abs() < 0.01);

    let left_bands = left["parametric_eq"]["bands"].as_array().unwrap();
    assert_eq!(left_bands.len(), 3);
    assert_eq!(left_bands[0]["type"].as_str().unwrap(), "Peak/Dip");
    assert_eq!(left_bands[0]["frequency"].as_f64().unwrap(), 100.0);
    assert_eq!(left_bands[2]["type"].as_str().unwrap(), "High Shelf");

    // Right channel
    let right = &channels["right"];
    assert!(right["headroom_gain_db"].as_f64().unwrap() < 0.0);
    assert!(right.get("delay_ms").is_none()); // no delay on right

    let right_bands = right["parametric_eq"]["bands"].as_array().unwrap();
    assert_eq!(right_bands.len(), 2);
    assert_eq!(right_bands[1]["type"].as_str().unwrap(), "Low Shelf");
    assert!(right_bands[0]["enabled"].as_bool().unwrap());
}

#[test]
fn tool_contract_roon_json_keeps_per_channel_manual_setup_data() {
    let output = make_test_output();
    let result = export_roon(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let channels = parsed["channels"].as_object().unwrap();

    assert_eq!(channels.len(), 2);
    for (name, channel) in channels {
        assert!(channel["headroom_gain_db"].as_f64().unwrap().is_finite());
        if name == "left" {
            assert!(channel["delay_ms"].as_f64().unwrap().is_finite());
        }
        let bands = channel["parametric_eq"]["bands"].as_array().unwrap();
        assert!(!bands.is_empty());
        assert!(bands.len() <= 20);
        for band in bands {
            assert_eq!(band["enabled"], serde_json::json!(true));
            assert!(band["frequency"].as_f64().unwrap().is_finite());
            assert!(band["q"].as_f64().unwrap().is_finite());
        }
    }

    run_optional_export_validator("ROOMEQ_ROON_VALIDATE_CMD", "json", &result);
}

#[test]
fn test_wavelet_rejects_unknown_filter_type() {
    let output = make_single_filter_output("lowsehlf", 3.0);

    let err = export_wavelet(&output, 48_000.0).unwrap_err();

    assert!(
        err.to_string().contains("Unsupported biquad filter type"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pipewire_rejects_unknown_filter_type() {
    let output = make_single_filter_output("lowsehlf", 3.0);

    let err = export_pipewire(&output, 48_000.0).unwrap_err();

    assert!(
        err.to_string()
            .contains("Unsupported PipeWire biquad filter type"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_pipewire_highpassvariableq_omits_gain_control() {
    let output = make_single_filter_output("highpassvariableq", -6.0);

    let conf = export_pipewire(&output, 48_000.0).unwrap();

    assert!(conf.contains("label = bq_highpass"));
    assert!(
        conf.contains("control = { \"Freq\" = 80.0  \"Q\" = 0.7070 }"),
        "highpassvariableq should emit only Freq/Q controls:\n{conf}"
    );
    assert!(
        !conf.contains("\"Gain\" = -6.00"),
        "PipeWire highpassvariableq must not emit unsupported Gain control:\n{conf}"
    );
}
