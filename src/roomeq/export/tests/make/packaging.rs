#[test]
fn external_exports_reject_routed_bass_management() {
    let output = make_routed_bass_output();
    assert!(external_export_supported(&output, ExportFormat::CamillaDsp).is_ok());

    for format in [
        ExportFormat::EasyEffects,
        ExportFormat::Wavelet,
        ExportFormat::PipeWire,
        ExportFormat::RoonDsp,
        ExportFormat::Rew,
        ExportFormat::BiquadCoefficients,
    ] {
        let err = external_export_supported(&output, format).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot represent routed home-cinema bass management safely"),
            "unexpected error for {format:?}: {err}"
        );
    }

    // APO decides graph representability during rendering: some static
    // Channel/Copy routings work, while this fixture's source fan-out must
    // fail instead of silently changing the DSP.
    assert!(external_export_supported(&output, ExportFormat::EqualizerApo).is_ok());
    let err = export_equalizer_apo(&output).unwrap_err();
    assert!(err.to_string().contains("cannot preserve fan-out"));
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

fn write_test_wav(path: &std::path::Path, sample_rate: u32, channels: u16, frames: usize) {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for frame in 0..frames {
        for channel in 0..channels {
            writer
                .write_sample(if frame == 0 && channel == 0 { 1.0_f32 } else { 0.0 })
                .unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn add_convolution(output: &mut DspChainOutput, channel: &str, path: &str) {
    output
        .channels
        .get_mut(channel)
        .unwrap()
        .plugins
        .push(PluginConfigWrapper {
            plugin_type: "convolution".to_string(),
            parameters: json!({"ir_file": path}),
        });
}

#[test]
fn roon_export_writes_deterministic_routed_convolver_archive() {
    use std::io::Read;

    let source_dir = tempfile::tempdir().unwrap();
    let dest_dir = tempfile::tempdir().unwrap();
    write_test_wav(&source_dir.path().join("left.wav"), 48_000, 1, 64);

    let mut output = make_test_output();
    add_convolution(&mut output, "left", "left.wav");
    let manifest_path = dest_dir.path().join("room_eq.json");
    export_dsp_chain_with_convolution_sidecars(
        &output,
        ExportFormat::RoonDsp,
        &manifest_path,
        48_000.0,
        source_dir.path(),
    )
    .unwrap();

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["manifest_version"], json!(1));
    assert_eq!(manifest["importable_preset"], json!(false));
    assert_eq!(
        manifest["convolution_archive"]["file"],
        json!("room_eq_convolution.zip")
    );
    assert_eq!(
        manifest["convolution_archive"]["wave_channel_mask_hex"],
        json!("3")
    );

    let archive_path = dest_dir.path().join("room_eq_convolution.zip");
    let first = std::fs::read(&archive_path).unwrap();
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(&first)).unwrap();
    let names: Vec<_> = (0..archive.len())
        .map(|index| archive.by_index(index).unwrap().name().to_string())
        .collect();
    assert_eq!(
        names,
        [
            "room_eq_48000_2ch.cfg",
            "filters/00_L.wav",
            "filters/01_R.wav"
        ]
    );
    let mut cfg = String::new();
    archive
        .by_name("room_eq_48000_2ch.cfg")
        .unwrap()
        .read_to_string(&mut cfg)
        .unwrap();
    assert_eq!(
        cfg,
        "48000 2 2 3\n0 0 0 0\nfilters/00_L.wav\n0\n0.0\n0.0\nfilters/01_R.wav\n0\n1.0\n1.0\n"
    );
    let right = archive.by_name("filters/01_R.wav").unwrap();
    assert_eq!(right.size(), 44 + 64 * 4);
    drop(right);
    drop(archive);

    export_dsp_chain_with_convolution_sidecars(
        &output,
        ExportFormat::RoonDsp,
        &manifest_path,
        48_000.0,
        source_dir.path(),
    )
    .unwrap();
    assert_eq!(first, std::fs::read(&archive_path).unwrap());
}

#[test]
fn roon_convolver_rejects_unsafe_malformed_and_mismatched_wavs() {
    let source_dir = tempfile::tempdir().unwrap();
    let dest_dir = tempfile::tempdir().unwrap();
    let archive = dest_dir.path().join("room_eq_convolution.zip");

    let cases = [
        ("stereo.wav", 48_000, 2, 64, "must be mono"),
        ("wrong_rate.wav", 44_100, 1, 64, "has sample rate"),
    ];
    for (name, rate, channels, frames, expected) in cases {
        write_test_wav(&source_dir.path().join(name), rate, channels, frames);
        let mut output = make_test_output();
        add_convolution(&mut output, "left", name);
        let error = package_roon_convolution_archive(
            &output,
            source_dir.path(),
            &archive,
            48_000.0,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains(expected), "unexpected error: {error}");
    }

    std::fs::write(source_dir.path().join("broken.wav"), b"not a wave file").unwrap();
    let mut malformed = make_test_output();
    add_convolution(&mut malformed, "left", "broken.wav");
    assert!(
        package_roon_convolution_archive(
            &malformed,
            source_dir.path(),
            &archive,
            48_000.0,
        )
        .unwrap_err()
        .to_string()
        .contains("not a valid WAV")
    );

    let mut unsafe_path = make_test_output();
    add_convolution(&mut unsafe_path, "left", "../outside.wav");
    assert!(
        package_roon_convolution_archive(
            &unsafe_path,
            source_dir.path(),
            &archive,
            48_000.0,
        )
        .unwrap_err()
        .to_string()
        .contains("safe relative path")
    );
}

#[test]
fn roon_convolver_rejects_unequal_lengths_and_unknown_channels() {
    let source_dir = tempfile::tempdir().unwrap();
    let dest_dir = tempfile::tempdir().unwrap();
    let archive = dest_dir.path().join("room_eq_convolution.zip");
    write_test_wav(&source_dir.path().join("left.wav"), 48_000, 1, 64);
    write_test_wav(&source_dir.path().join("right.wav"), 48_000, 1, 128);

    let mut output = make_test_output();
    add_convolution(&mut output, "left", "left.wav");
    add_convolution(&mut output, "right", "right.wav");
    assert!(
        package_roon_convolution_archive(
            &output,
            source_dir.path(),
            &archive,
            48_000.0,
        )
        .unwrap_err()
        .to_string()
        .contains("equal impulse-response lengths")
    );

    let mut unknown = output.clone();
    let mut chain = unknown.channels.remove("right").unwrap();
    chain.channel = "aux".to_string();
    unknown.channels.insert("aux".to_string(), chain);
    assert!(
        package_roon_convolution_archive(
            &unknown,
            source_dir.path(),
            &archive,
            48_000.0,
        )
        .unwrap_err()
        .to_string()
        .contains("does not know the WAVE channel mapping")
    );
}
