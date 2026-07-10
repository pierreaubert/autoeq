#[test]
fn external_exports_reject_routed_bass_management() {
    let output = make_routed_bass_output();
    assert!(external_export_supported(&output, ExportFormat::CamillaDsp).is_ok());

    for format in [
        ExportFormat::EqualizerApo,
        ExportFormat::EasyEffects,
        ExportFormat::Wavelet,
        ExportFormat::PipeWire,
        ExportFormat::RoonDsp,
    ] {
        let err = external_export_supported(&output, format).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot represent routed home-cinema bass management safely"),
            "unexpected error for {format:?}: {err}"
        );
    }
}

#[test]
fn package_convolution_sidecars_copies_and_rewrites_relative_paths() {
    let source_dir = tempfile::tempdir().unwrap();
    let dest_dir = tempfile::tempdir().unwrap();
    std::fs::write(source_dir.path().join("L_fir_96000hz.wav"), b"wav").unwrap();

    let mut output = make_test_output();
    output
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": "L_fir_96000hz.wav"}),
        });

    let packaged =
        package_convolution_sidecars(&output, source_dir.path(), dest_dir.path()).unwrap();

    assert_eq!(
        std::fs::read(dest_dir.path().join("L_fir_96000hz.wav")).unwrap(),
        b"wav"
    );
    let ir_file = packaged.channels["left"]
        .plugins
        .iter()
        .find(|plugin| plugin.plugin_type == "convolution")
        .unwrap()
        .parameters
        .get("ir_file")
        .and_then(|value| value.as_str())
        .unwrap();
    assert_eq!(ir_file, "L_fir_96000hz.wav");
}

#[test]
fn package_convolution_sidecars_avoids_destination_collisions() {
    let source_dir = tempfile::tempdir().unwrap();
    let dest_dir = tempfile::tempdir().unwrap();
    std::fs::write(source_dir.path().join("L_fir_96000hz.wav"), b"new").unwrap();
    std::fs::write(dest_dir.path().join("L_fir_96000hz.wav"), b"old").unwrap();

    let mut output = make_test_output();
    output
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": "L_fir_96000hz.wav"}),
        });

    let packaged =
        package_convolution_sidecars(&output, source_dir.path(), dest_dir.path()).unwrap();

    assert_eq!(
        std::fs::read(dest_dir.path().join("L_fir_96000hz_002.wav")).unwrap(),
        b"new"
    );
    let ir_file = packaged.channels["left"]
        .plugins
        .iter()
        .find(|plugin| plugin.plugin_type == "convolution")
        .unwrap()
        .parameters
        .get("ir_file")
        .and_then(|value| value.as_str())
        .unwrap();
    assert_eq!(ir_file, "L_fir_96000hz_002.wav");
}

#[test]
fn export_with_convolution_sidecars_uses_selected_sample_rate() {
    let source_dir = tempfile::tempdir().unwrap();
    let dest_dir = tempfile::tempdir().unwrap();
    std::fs::write(source_dir.path().join("L_fir_96000hz.wav"), b"wav").unwrap();

    let mut output = make_test_output();
    output
        .channels
        .get_mut("left")
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": "L_fir_96000hz.wav"}),
        });

    let export_path = dest_dir.path().join("room_eq_cdsp.yaml");
    export_dsp_chain_with_convolution_sidecars(
        &output,
        ExportFormat::CamillaDsp,
        &export_path,
        96_000.0,
        source_dir.path(),
    )
    .unwrap();

    let yaml = std::fs::read_to_string(&export_path).unwrap();
    assert!(yaml.contains("samplerate: 96000"));
    assert!(yaml.contains("filename: \"L_fir_96000hz.wav\""));
    assert!(dest_dir.path().join("L_fir_96000hz.wav").is_file());
}
