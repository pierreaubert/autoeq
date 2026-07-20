#[test]
fn test_export_equalizer_apo() {
    let output = make_test_output();
    let result = export_equalizer_apo(&output).unwrap();

    assert!(result.contains("Channel: L"));
    assert!(result.contains("Channel: R"));
    assert!(result.contains("Preamp: -2.500000000 dB"));
    assert!(result.contains("Delay: 1.500000000 ms"));
    assert!(
        result.contains("Filter  1: ON PK Fc 100.000000000 Hz Gain -5.000000000 dB Q 2.000000000")
    );
    assert!(
        result
            .contains("Filter  3: ON HSC Fc 8000.000000000 Hz Gain -2.000000000 dB Q 0.700000000")
    );
    assert!(
        result.contains("Filter  1: ON PK Fc 200.000000000 Hz Gain -3.000000000 dB Q 1.000000000")
    );
    assert!(
        result.contains("Filter  2: ON LSC Fc 80.000000000 Hz Gain +4.000000000 dB Q 0.710000000")
    );
}

#[test]
fn rew_export_is_single_channel_generic_eq_text_and_fails_closed() {
    let mut output = make_test_output();
    output.channels.remove("right");
    output.channels.get_mut("left").unwrap().plugins.retain(|plugin| {
        plugin.plugin_type != "delay"
    });

    let text = export_rew(&output).expect("serial PEQ chain should export to REW");
    assert!(text.starts_with("Filter Settings file\n\nEqualiser: Generic\n"));
    assert!(text.contains("Channel: L"));
    assert!(text.contains("Preamp: -2.500000000 dB"));
    assert!(
        text.contains("Filter  1: ON PK Fc 100.000000000 Hz Gain -5.000000000 dB Q 2.000000000")
    );
    assert!(
        text.contains("Filter  3: ON HS Fc 8000.000000000 Hz Gain -2.000000000 dB Q 0.700000000")
    );

    let mut multichannel = output.clone();
    let mut right = multichannel.channels["left"].clone();
    right.channel = "right".to_string();
    multichannel.channels.insert("right".to_string(), right);
    let error = export_rew(&multichannel).unwrap_err().to_string();
    assert!(error.contains("exactly one channel"), "unexpected error: {error}");

    let mut convolution = output;
    convolution.channels.get_mut("left").unwrap().plugins.push(PluginConfigWrapper {
        plugin_type: "convolution".to_string(),
        parameters: json!({"ir_file": "left.wav"}),
    });
    let error = export_rew(&convolution).unwrap_err().to_string();
    assert!(error.contains("does not support"), "unexpected error: {error}");
}

#[test]
fn normalized_biquad_export_round_trips_canonical_transfer_functions() {
    let output = make_test_output();
    let rendered = export_normalized_biquad_coefficients(&output, 48_000.0)
        .expect("serial channel chains should export normalized coefficients");
    let document: serde_json::Value = serde_json::from_str(&rendered).unwrap();

    assert_eq!(document["format"], "roomeq_normalized_biquad_coefficients");
    assert_eq!(document["version"], 1);
    assert_eq!(document["sample_rate_hz"], 48_000.0);
    let channels = document["channels"].as_array().unwrap();
    assert_eq!(channels.len(), 2);
    assert_eq!(channels[0]["channel"], "L");
    assert_eq!(channels[0]["preamp_gain_db"], -2.5);
    assert_eq!(channels[0]["delay_ms"], 1.5);

    let source_filters = extract_eq_filters(&output.channels["left"].plugins);
    let sections = channels[0]["sections"].as_array().unwrap();
    assert_eq!(sections.len(), source_filters.len());
    for (index, (source, section)) in source_filters.iter().zip(sections).enumerate() {
        assert_eq!(section["section_index"], index);
        assert_eq!(section["order"], 2);
        assert_eq!(section["filter_type"], source.filter_type);
        assert_eq!(section["a0"], 1.0);

        let coefficients = ["b0", "b1", "b2", "a1", "a2"]
            .map(|name| section[name].as_f64().unwrap());
        let canonical = math_audio_iir_fir::Biquad::new(
            parse_biquad_filter_type(&source.filter_type).unwrap(),
            source.freq,
            48_000.0,
            source.q,
            source.gain_db,
        );
        for frequency in [20.0, 80.0, 1_000.0, 10_000.0, 20_000.0] {
            let omega = 2.0 * std::f64::consts::PI * frequency / 48_000.0;
            let z1 = num_complex::Complex64::from_polar(1.0, -omega);
            let z2 = z1 * z1;
            let numerator = coefficients[0] + coefficients[1] * z1 + coefficients[2] * z2;
            let denominator = 1.0 + coefficients[3] * z1 + coefficients[4] * z2;
            let exported_db = 20.0 * (numerator / denominator).norm().log10();
            assert!(
                (exported_db - canonical.log_result(frequency)).abs() < 1e-10,
                "section {index} mismatch at {frequency} Hz"
            );
        }
    }
}

#[test]
fn normalized_biquad_export_supports_every_canonical_filter_type() {
    for filter_type in [
        "lowpass",
        "highpass",
        "highpassvariableq",
        "bandpass",
        "peak",
        "notch",
        "lowshelf",
        "highshelf",
        "allpass",
        "lowshelforf",
        "highshelforf",
        "peakmatched",
    ] {
        let output = make_single_filter_output(filter_type, -3.0);
        let rendered = export_normalized_biquad_coefficients(&output, 48_000.0)
            .unwrap_or_else(|error| panic!("{filter_type} coefficient export failed: {error}"));
        let document: serde_json::Value = serde_json::from_str(&rendered).unwrap();
        assert_eq!(document["channels"][0]["sections"][0]["filter_type"], filter_type);
    }

    for coefficient_only_type in ["lowshelforf", "highshelforf", "peakmatched"] {
        let output = make_single_filter_output(coefficient_only_type, -3.0);
        let error = export_rew(&output).unwrap_err().to_string();
        assert!(
            error.contains("does not support filter type"),
            "REW unexpectedly accepted {coefficient_only_type}: {error}"
        );
    }
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
        route.destination == "LFE" && matches!(route.source_channel.as_str(), "L" | "R" | "LFE")
    });
    graph.input_channels = vec!["L".to_string(), "R".to_string(), "LFE".to_string()];
    graph.output_channels = vec!["LFE".to_string()];

    let error = export_equalizer_apo(&output).unwrap_err();

    assert!(error.to_string().contains("routed destination 'LFE'"));
    assert!(error.to_string().contains("intermediate bus"));
}

#[test]
fn test_export_easyeffects() {
    let output = make_systemwide_test_output();
    let result = export_easyeffects(&output).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let eq = &parsed["output"]["equalizer#0"];
    assert_eq!(eq["num-bands"].as_u64().unwrap(), 3);
    assert!(eq["left"]["band0"]["frequency"].as_f64().unwrap() > 0.0);

    // Check filter types
    let band0_type = eq["left"]["band0"]["type"].as_str().unwrap();
    assert_eq!(band0_type, "Bell");
}

#[test]
fn tool_contract_easyeffects_json_has_mirrored_stereo_preset() {
    let output = make_systemwide_test_output();
    let result = export_easyeffects(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let eq = &parsed["output"]["equalizer#0"];

    assert_eq!(eq["split-channels"], serde_json::json!(false));
    assert_eq!(eq["num-bands"].as_u64().unwrap(), 3);
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
    let output = make_systemwide_test_output();
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
    let output = make_systemwide_test_output();
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
    assert!(result.contains("config = { \"max-delay\" = 0.001500000 }"));
    assert!(result.contains("control = { \"Delay (s)\" = 0.001500000 }"));
    assert!(result.contains("label = bq_peaking"));
    assert!(result.contains("label = mixer"));
    assert!(result.contains("\"ch0_left_plugin_0_gain:In 1\""));
    assert!(result.contains("\"ch0_left_plugin_2_eq_2:Out\""));

    run_optional_export_validator("ROOMEQ_PIPEWIRE_VALIDATE_CMD", "conf", &result);
}

#[test]
fn test_export_roon() {
    let output = make_test_output();
    let result = export_roon(&output).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let channels = &parsed["channels"];
    assert_eq!(parsed["manifest_version"], serde_json::json!(1));
    assert_eq!(parsed["artifact_type"], "roon_manual_iir_setup");
    assert_eq!(parsed["importable_preset"], false);

    // Left channel
    let left = &channels["left"];
    assert!(left["headroom_gain_db"].as_f64().unwrap() < 0.0);
    assert!((left["delay_ms"].as_f64().unwrap() - 1.5).abs() < 0.01);

    let left_bands = left["parametric_eq"]["bands"].as_array().unwrap();
    assert_eq!(left_bands.len(), 3);
    assert_eq!(left_bands[0]["type"].as_str().unwrap(), "Peak/Dip");
    assert_eq!(left_bands[0]["frequency"].as_f64().unwrap(), 100.0);
    assert_eq!(left_bands[2]["type"].as_str().unwrap(), "High Shelf");
    let operations = left["procedural_eq"]["operations"].as_array().unwrap();
    assert_eq!(operations[0]["type"], "Volume");
    assert_eq!(operations[1]["type"], "Delay");
    assert_eq!(operations[2]["type"], "Parametric EQ");

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
fn roon_manifest_schema_and_json_round_trip_are_versioned() {
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../docs/roon_manual_iir_manifest.schema.json"
    ))
    .unwrap();
    assert_eq!(schema["properties"]["manifest_version"]["const"], json!(1));
    assert_eq!(
        schema["properties"]["artifact_type"]["const"],
        json!("roon_manual_iir_setup")
    );

    let original: serde_json::Value =
        serde_json::from_str(&export_roon(&make_test_output()).unwrap()).unwrap();
    let round_tripped: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&original).unwrap()).unwrap();
    assert_eq!(round_tripped, original);
}

#[test]
fn test_wavelet_rejects_unknown_filter_type() {
    let output = make_single_filter_output("lowsehlf", 3.0);

    let err = export_wavelet(&output, 48_000.0).unwrap_err();

    assert!(
        err.to_string().contains("does not support filter type"),
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
        conf.contains("control = { \"Freq\" = 80.000000000  \"Q\" = 0.707000000 }"),
        "highpassvariableq should emit only Freq/Q controls:\n{conf}"
    );
    assert!(
        !conf.contains("\"Gain\" = -6.00"),
        "PipeWire highpassvariableq must not emit unsupported Gain control:\n{conf}"
    );
}

#[test]
fn pipewire_preserves_plugin_order_and_emits_convolution() {
    let mut output = make_test_output();
    let left = output.channels.get_mut("left").unwrap();
    left.plugins.insert(
        1,
        PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": "/tmp/room eq.wav"}),
        },
    );

    let conf = export_pipewire(&output, 48_000.0).unwrap();
    let gain = conf.find("ch0_left_plugin_0_gain").unwrap();
    let convolver = conf.find("ch0_left_plugin_1_convolver").unwrap();
    let delay = conf.find("ch0_left_plugin_2_delay").unwrap();
    let eq = conf.find("ch0_left_plugin_3_eq_0").unwrap();
    assert!(gain < convolver && convolver < delay && delay < eq);
    assert!(conf.contains("label = convolver"));
    assert!(conf.contains("filename = \"/tmp/room eq.wav\""));
}

#[test]
fn pipewire_never_silently_omits_unsupported_processing() {
    let mut output = make_test_output();
    output
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "band_split".to_string(),
            parameters: json!({"type": "LR24", "frequency": 80.0}),
        });
    let error = export_pipewire(&output, 48_000.0).unwrap_err().to_string();
    assert!(error.contains("does not support channel 'left' plugin #3 ('band_split')"));

    let mut output = make_test_output();
    output.global_plugins.push(PluginConfigWrapper {
        plugin_type: "matrix".to_string(),
        parameters: json!({"label": "home_cinema_bass_management"}),
    });
    let error = export_pipewire(&output, 48_000.0).unwrap_err().to_string();
    assert!(error.contains("does not yet support global plugin #0 ('matrix')"));
}

#[test]
fn pipewire_renders_lr24_and_lr48_crossovers_as_complete_cascades() {
    for (kind, sections) in [("LR24", 2), ("LR48", 4)] {
        let mut output = make_test_output();
        output.channels.get_mut("left").unwrap().plugins = vec![PluginConfigWrapper {
            plugin_type: "crossover".to_string(),
            parameters: json!({"type": kind, "output": "low", "frequency": 80.0}),
        }];
        let conf = export_pipewire(&output, 48_000.0).unwrap();
        assert_eq!(conf.matches("label = bq_lowpass").count(), sections);
        for section in 0..sections {
            assert!(conf.contains(&format!("plugin_0_crossover_{section}")));
        }
    }
}

#[test]
fn pipewire_and_camilladsp_exports_preserve_the_same_serial_semantics() {
    let output = make_test_output();
    let pipewire = export_pipewire(&output, 48_000.0).unwrap();
    let camilla = export_camilladsp(&output, 48_000.0).unwrap();

    // The two backends use different gain units, so compare PipeWire's linear
    // multiplier with CamillaDSP's dB value through the defining conversion.
    let expected_linear = 10.0_f64.powf(-2.5 / 20.0);
    assert!(pipewire.contains(&format!("\"Gain 1\" = {expected_linear:.12}")));
    assert!(camilla.contains("gain: -2.50"));

    assert!(pipewire.contains("\"Delay (s)\" = 0.001500000"));
    assert!(camilla.contains("delay: 1.500"));
    assert_eq!(
        pipewire.matches("label = bq_").count(),
        camilla.matches("type: Biquad").count()
    );

    let pipewire_gain = pipewire.find("ch0_left_plugin_0_gain").unwrap();
    let pipewire_delay = pipewire.find("ch0_left_plugin_1_delay").unwrap();
    let pipewire_eq = pipewire.find("ch0_left_plugin_2_eq_0").unwrap();
    assert!(pipewire_gain < pipewire_delay && pipewire_delay < pipewire_eq);
    let camilla_gain = camilla.find("left_gain:").unwrap();
    let camilla_delay = camilla.find("left_delay:").unwrap();
    let camilla_eq = camilla.find("left_peq_0:").unwrap();
    assert!(camilla_gain < camilla_delay && camilla_delay < camilla_eq);
}

#[test]
fn pipewire_rendered_controls_match_the_canonical_magnitude_response() {
    let output = make_test_output();
    let conf = export_pipewire(&output, 48_000.0).unwrap();
    let chain = &output.channels["left"];
    let canonical_filters = extract_eq_filters(&chain.plugins);
    let canonical_biquads: Vec<_> = canonical_filters
        .iter()
        .map(|filter| {
            math_audio_iir_fir::Biquad::new(
                parse_biquad_filter_type(&filter.filter_type).unwrap(),
                filter.freq,
                48_000.0,
                filter.q,
                filter.gain_db,
            )
        })
        .collect();

    let left_node_lines: Vec<_> = conf
        .lines()
        .filter(|line| line.contains("ch0_left_plugin_"))
        .filter(|line| line.contains("type = builtin"))
        .collect();
    for frequency in [20.0, 80.0, 100.0, 500.0, 1000.0, 8000.0, 20_000.0] {
        let canonical_db = -2.5
            + canonical_biquads
                .iter()
                .map(|biquad| biquad.log_result(frequency))
                .sum::<f64>();
        let mut rendered_db = 0.0;
        for line in &left_node_lines {
            if line.contains("label = mixer") {
                rendered_db += 20.0 * control_value(line, "\"Gain 1\"").abs().log10();
            } else if let Some(filter_type) = pipewire_line_filter_type(line) {
                let gain = if line.contains("\"Gain\"") {
                    control_value(line, "\"Gain\"")
                } else {
                    0.0
                };
                let biquad = math_audio_iir_fir::Biquad::new(
                    filter_type,
                    control_value(line, "\"Freq\""),
                    48_000.0,
                    control_value(line, "\"Q\""),
                    gain,
                );
                rendered_db += biquad.log_result(frequency);
            }
        }
        assert!(
            (rendered_db - canonical_db).abs() < 1e-8,
            "{frequency} Hz: canonical {canonical_db} dB, PipeWire controls {rendered_db} dB"
        );
    }
}

fn control_value(line: &str, name: &str) -> f64 {
    line.split_once(&format!("{name} = "))
        .unwrap()
        .1
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

fn pipewire_line_filter_type(line: &str) -> Option<math_audio_iir_fir::BiquadFilterType> {
    let name = if line.contains("label = bq_peaking") {
        "peak"
    } else if line.contains("label = bq_lowshelf") {
        "lowshelf"
    } else if line.contains("label = bq_highshelf") {
        "highshelf"
    } else if line.contains("label = bq_lowpass") {
        "lowpass"
    } else if line.contains("label = bq_highpass") {
        "highpass"
    } else if line.contains("label = bq_notch") {
        "notch"
    } else if line.contains("label = bq_bandpass") {
        "bandpass"
    } else if line.contains("label = bq_allpass") {
        "allpass"
    } else {
        return None;
    };
    Some(parse_biquad_filter_type(name).unwrap())
}

#[test]
fn pipewire_represents_polarity_inversion_as_negative_linear_gain() {
    let mut output = make_test_output();
    output.channels.get_mut("left").unwrap().plugins[0].parameters["invert"] = json!(true);
    let conf = export_pipewire(&output, 48_000.0).unwrap();
    assert!(conf.contains("label = mixer"));
    assert!(conf.contains("\"Gain 1\" = -0.749894209332"));
}

#[test]
fn tool_contract_pipewire_accepts_every_native_biquad_and_convolver() {
    let temp = tempfile::tempdir().unwrap();
    let ir_path = temp.path().join("identity.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&ir_path, spec).unwrap();
    writer.write_sample(1.0_f32).unwrap();
    writer.finalize().unwrap();

    let mut output = make_test_output();
    let left = output.channels.get_mut("left").unwrap();
    left.plugins = vec![
        PluginConfigWrapper {
            plugin_type: "gain".to_string(),
            parameters: json!({"gain_db": -3.0, "invert": true}),
        },
        PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: json!({"filters": [
                {"filter_type": "peak", "freq": 100.0, "q": 0.7, "db_gain": 1.0},
                {"filter_type": "lowshelf", "freq": 120.0, "q": 0.7, "db_gain": 1.0},
                {"filter_type": "highshelf", "freq": 8000.0, "q": 0.7, "db_gain": -1.0},
                {"filter_type": "lowpass", "freq": 18000.0, "q": 0.7, "db_gain": 0.0},
                {"filter_type": "highpass", "freq": 20.0, "q": 0.7, "db_gain": 0.0},
                {"filter_type": "notch", "freq": 500.0, "q": 2.0, "db_gain": 0.0},
                {"filter_type": "bandpass", "freq": 1000.0, "q": 1.0, "db_gain": 0.0},
                {"filter_type": "allpass", "freq": 2000.0, "q": 0.7, "db_gain": 0.0}
            ]}),
        },
        PluginConfigWrapper {
            plugin_type: "delay".to_string(),
            parameters: json!({"delay_ms": 0.5}),
        },
        PluginConfigWrapper {
            plugin_type: "crossover".to_string(),
            parameters: json!({"type": "LR24", "output": "high", "frequency": 20.0}),
        },
        PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": ir_path}),
        },
    ];

    let conf = export_pipewire(&output, 48_000.0).unwrap();
    for label in [
        "mixer",
        "bq_peaking",
        "bq_lowshelf",
        "bq_highshelf",
        "bq_lowpass",
        "bq_highpass",
        "bq_notch",
        "bq_bandpass",
        "bq_allpass",
        "delay",
        "convolver",
    ] {
        assert!(
            conf.contains(&format!("label = {label}")),
            "missing {label}"
        );
    }
    run_optional_export_validator("ROOMEQ_PIPEWIRE_VALIDATE_CMD", "conf", &conf);
}

fn canonical_serial_db(chain: &ChannelDspChain, frequency: f64, sample_rate: f64) -> f64 {
    extract_gain_db(&chain.plugins)
        + extract_eq_filters(&chain.plugins)
            .iter()
            .map(|filter| {
                math_audio_iir_fir::Biquad::new(
                    parse_biquad_filter_type(&filter.filter_type).unwrap(),
                    filter.freq,
                    sample_rate,
                    filter.q,
                    filter.gain_db,
                )
                .log_result(frequency)
            })
            .sum::<f64>()
}

#[test]
fn easyeffects_export_reconstructs_the_canonical_response() {
    let output = make_systemwide_test_output();
    let preset: serde_json::Value =
        serde_json::from_str(&export_easyeffects(&output).unwrap()).unwrap();
    let eq = &preset["output"]["equalizer#0"];
    let gain = eq["input-gain"].as_f64().unwrap();
    let bands = eq["left"].as_object().unwrap();

    for frequency in [32.0, 100.0, 1000.0, 8000.0, 20_000.0] {
        let mut exported_db = gain;
        for band in bands.values() {
            let filter_type = match band["type"].as_str().unwrap() {
                "Bell" => "peak",
                "Lo Shelf" => "lowshelf",
                "Hi Shelf" => "highshelf",
                "Lo-pass" => "lowpass",
                "Hi-pass" => "highpass",
                "Notch" => "notch",
                "Bandpass" => "bandpass",
                "Allpass" => "allpass",
                other => panic!("unexpected EasyEffects filter type {other}"),
            };
            exported_db += math_audio_iir_fir::Biquad::new(
                parse_biquad_filter_type(filter_type).unwrap(),
                band["frequency"].as_f64().unwrap(),
                48_000.0,
                band["q"].as_f64().unwrap(),
                band["gain"].as_f64().unwrap(),
            )
            .log_result(frequency);
        }
        let canonical = canonical_serial_db(&output.channels["left"], frequency, 48_000.0);
        assert!((exported_db - canonical).abs() < 1e-10);
    }
}

#[test]
fn roon_export_reconstructs_the_canonical_response() {
    let output = make_test_output();
    let preset: serde_json::Value = serde_json::from_str(&export_roon(&output).unwrap()).unwrap();
    let left = &preset["channels"]["left"];
    for frequency in [32.0, 100.0, 1000.0, 8000.0, 20_000.0] {
        let mut exported_db = left["headroom_gain_db"].as_f64().unwrap();
        for band in left["parametric_eq"]["bands"].as_array().unwrap() {
            let filter_type = match band["type"].as_str().unwrap() {
                "Peak/Dip" => "peak",
                "Low Shelf" => "lowshelf",
                "High Shelf" => "highshelf",
                "Low Pass" => "lowpass",
                "High Pass" => "highpass",
                "Band Pass" => "bandpass",
                "Band Stop" => "notch",
                other => panic!("unexpected Roon filter type {other}"),
            };
            exported_db += math_audio_iir_fir::Biquad::new(
                parse_biquad_filter_type(filter_type).unwrap(),
                band["frequency"].as_f64().unwrap(),
                48_000.0,
                band["q"].as_f64().unwrap(),
                band["gain"].as_f64().unwrap(),
            )
            .log_result(frequency);
        }
        let canonical = canonical_serial_db(&output.channels["left"], frequency, 48_000.0);
        assert!((exported_db - canonical).abs() < 1e-10);
    }
    assert_eq!(left["delay_ms"], json!(1.5));
}

#[test]
fn wavelet_export_matches_canonical_values_at_its_band_centres() {
    let output = make_systemwide_test_output();
    let text = export_wavelet(&output, 48_000.0).unwrap();
    let line = text
        .lines()
        .find(|line| line.starts_with("GraphicEQ:"))
        .unwrap();
    for pair in line.trim_start_matches("GraphicEQ:").split(';') {
        let fields: Vec<_> = pair.split_whitespace().collect();
        let frequency: f64 = fields[0].parse().unwrap();
        let exported_db: f64 = fields[1].parse().unwrap();
        let canonical = canonical_serial_db(&output.channels["left"], frequency, 48_000.0);
        assert!(
            (exported_db - canonical).abs() <= 0.051,
            "{frequency} Hz: Wavelet {exported_db} dB, canonical {canonical} dB"
        );
    }
}

#[test]
fn equalizer_apo_export_reconstructs_the_canonical_response() {
    let output = make_test_output();
    let text = export_equalizer_apo(&output).unwrap();
    let mut in_left = false;
    let mut gain = 0.0;
    let mut filters = Vec::new();
    for line in text.lines() {
        if line == "Channel: L" {
            in_left = true;
            continue;
        }
        if line.starts_with("Channel:") && in_left {
            break;
        }
        if !in_left {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.first() == Some(&"Preamp:") {
            gain = fields[1].parse().unwrap();
        } else if fields.first() == Some(&"Filter") {
            let filter_type = match fields[3] {
                "PK" => "peak",
                "LSC" => "lowshelf",
                "HSC" => "highshelf",
                "LP" => "lowpass",
                "HP" => "highpass",
                "NO" => "notch",
                "BP" => "bandpass",
                "AP" => "allpass",
                other => panic!("unexpected APO filter type {other}"),
            };
            let frequency: f64 = fields[5].parse().unwrap();
            let q_index = fields.iter().position(|field| *field == "Q").unwrap();
            let q: f64 = fields[q_index + 1].parse().unwrap();
            let filter_gain = fields
                .iter()
                .position(|field| *field == "Gain")
                .map(|index| fields[index + 1].parse().unwrap())
                .unwrap_or(0.0);
            filters.push(math_audio_iir_fir::Biquad::new(
                parse_biquad_filter_type(filter_type).unwrap(),
                frequency,
                48_000.0,
                q,
                filter_gain,
            ));
        }
    }
    for frequency in [32.0, 100.0, 1000.0, 8000.0, 20_000.0] {
        let exported = gain
            + filters
                .iter()
                .map(|filter| filter.log_result(frequency))
                .sum::<f64>();
        let canonical = canonical_serial_db(&output.channels["left"], frequency, 48_000.0);
        assert!((exported - canonical).abs() < 1e-8);
    }
}
