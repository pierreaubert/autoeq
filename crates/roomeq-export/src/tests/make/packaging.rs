use std::collections::BTreeSet;
use std::io::{Cursor, Read};
use std::path::Path;

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
        let error = external_export_supported(&output, format).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cannot represent routed home-cinema bass management safely"),
            "unexpected error for {format:?}: {error}"
        );
    }

    assert!(external_export_supported(&output, ExportFormat::EqualizerApo).is_ok());
    let error = export_equalizer_apo(&output).unwrap_err();
    assert!(error.to_string().contains("cannot preserve fan-out"));
}

#[test]
fn package_convolution_sidecars_returns_hashed_member_and_rewritten_graph() {
    let mut output = make_test_output();
    add_convolution(&mut output, "left", "L_fir_96000hz.wav");
    let resources = [resource("L_fir_96000hz.wav", b"wav".to_vec())];

    let (packaged, members) =
        package_convolution_sidecars(
            &output,
            &resources,
            &BTreeSet::new(),
            &HashMap::new(),
        )
        .unwrap();

    assert_eq!(members.len(), 1);
    assert_eq!(members[0].relative_path, Path::new("L_fir_96000hz.wav"));
    assert_eq!(members[0].bytes.as_ref(), b"wav");
    assert_eq!(members[0].sha256.len(), 64);
    assert_eq!(convolution_path(&packaged, "left"), "L_fir_96000hz.wav");
}

#[test]
fn package_convolution_sidecars_avoids_explicit_destination_collisions() {
    let mut output = make_test_output();
    add_convolution(&mut output, "left", "L_fir_96000hz.wav");
    let resources = [resource("L_fir_96000hz.wav", b"new".to_vec())];
    let occupied = BTreeSet::from(["L_fir_96000hz.wav".to_string()]);

    let (packaged, members) =
        package_convolution_sidecars(&output, &resources, &occupied, &HashMap::new()).unwrap();

    assert_eq!(members[0].relative_path, Path::new("L_fir_96000hz_002.wav"));
    assert_eq!(members[0].bytes.as_ref(), b"new");
    assert_eq!(
        convolution_path(&packaged, "left"),
        "L_fir_96000hz_002.wav"
    );
}

#[test]
fn package_convolution_sidecars_reuses_existing_identical_member() {
    let mut output = make_test_output();
    add_convolution(&mut output, "left", "L_fir_96000hz.wav");
    let resources = [resource("L_fir_96000hz.wav", b"wav".to_vec())];
    let occupied = BTreeSet::from(["L_fir_96000hz.wav".to_string()]);
    let reusable = HashMap::from([(
        "L_fir_96000hz.wav".to_string(),
        "L_fir_96000hz.wav".to_string(),
    )]);

    let (packaged, members) =
        package_convolution_sidecars(&output, &resources, &occupied, &reusable).unwrap();

    assert!(members.is_empty());
    assert_eq!(convolution_path(&packaged, "left"), "L_fir_96000hz.wav");
}

#[test]
fn package_convolution_sidecars_deduplicates_reference_aliases_by_content() {
    let mut output = make_test_output();
    add_convolution(&mut output, "left", "relative/shared.wav");
    add_convolution(&mut output, "right", "/absolute/shared.wav");
    let resources = [
        resource("relative/shared.wav", b"same wav".to_vec()),
        resource("/absolute/shared.wav", b"same wav".to_vec()),
    ];

    let (packaged, members) = package_convolution_sidecars(
        &output,
        &resources,
        &BTreeSet::new(),
        &HashMap::new(),
    )
    .unwrap();

    assert_eq!(members.len(), 1);
    assert_eq!(convolution_path(&packaged, "left"), "shared.wav");
    assert_eq!(convolution_path(&packaged, "right"), "shared.wav");
}

#[test]
fn export_package_uses_selected_sample_rate_and_explicit_sidecar() {
    let mut output = make_test_output();
    add_convolution(&mut output, "left", "L_fir_96000hz.wav");
    let resources = [resource("L_fir_96000hz.wav", b"wav".to_vec())];

    let package = build_export_package(
        &output,
        ExportFormat::CamillaDsp,
        Path::new("room_eq_cdsp.yaml"),
        96_000.0,
        &resources,
        &BTreeSet::new(),
        &HashMap::new(),
    )
    .unwrap();

    let yaml = String::from_utf8(
        package
            .member(Path::new("room_eq_cdsp.yaml"))
            .unwrap()
            .bytes
            .to_vec(),
    )
    .unwrap();
    assert!(yaml.contains("samplerate: 96000"));
    assert!(yaml.contains("filename: \"L_fir_96000hz.wav\""));
    assert!(package.member(Path::new("L_fir_96000hz.wav")).is_some());
}

#[test]
fn roon_export_builds_deterministic_routed_convolver_archive() {
    let mut output = make_test_output();
    add_convolution(&mut output, "left", "left.wav");
    let resources = [resource("left.wav", test_wav(48_000, 1, 64))];

    let first = build_export_package(
        &output,
        ExportFormat::RoonDsp,
        Path::new("room_eq.json"),
        48_000.0,
        &resources,
        &BTreeSet::new(),
        &HashMap::new(),
    )
    .unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(
        &first.member(Path::new("room_eq.json")).unwrap().bytes,
    )
    .unwrap();
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

    let archive_member = first
        .member(Path::new("room_eq_convolution.zip"))
        .unwrap();
    let mut archive = zip::ZipArchive::new(Cursor::new(&archive_member.bytes)).unwrap();
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
    let mut config = String::new();
    archive
        .by_name("room_eq_48000_2ch.cfg")
        .unwrap()
        .read_to_string(&mut config)
        .unwrap();
    assert_eq!(
        config,
        "48000 2 2 3\n0 0\n0 0\nfilters/00_L.wav\n0\n0.0\n0.0\nfilters/01_R.wav\n0\n1.0\n1.0\n"
    );
    let right = archive.by_name("filters/01_R.wav").unwrap();
    assert_eq!(right.size(), 44 + 64 * 4);
    drop(right);
    drop(archive);

    let second = build_export_package(
        &output,
        ExportFormat::RoonDsp,
        Path::new("room_eq.json"),
        48_000.0,
        &resources,
        &BTreeSet::new(),
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(first, second);
}

#[test]
fn roon_convolver_rejects_unsafe_malformed_and_mismatched_wavs() {
    let cases = [
        ("stereo.wav", 48_000, 2, 64, "must be mono"),
        ("wrong_rate.wav", 44_100, 1, 64, "has sample rate"),
    ];
    for (name, rate, channels, frames, expected) in cases {
        let mut output = make_test_output();
        add_convolution(&mut output, "left", name);
        let resources = [resource(name, test_wav(rate, channels, frames))];
        let error = build_roon_convolution_archive(&output, &resources, 48_000.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "unexpected error: {error}");
    }

    let mut malformed = make_test_output();
    add_convolution(&mut malformed, "left", "broken.wav");
    let malformed_resources = [resource("broken.wav", b"not a wave file".to_vec())];
    assert!(
        build_roon_convolution_archive(&malformed, &malformed_resources, 48_000.0)
            .unwrap_err()
            .to_string()
            .contains("not a valid WAV")
    );

    let mut unsafe_path = make_test_output();
    add_convolution(&mut unsafe_path, "left", "../outside.wav");
    let unsafe_resources = [resource("../outside.wav", test_wav(48_000, 1, 64))];
    assert!(
        build_roon_convolution_archive(&unsafe_path, &unsafe_resources, 48_000.0)
            .unwrap_err()
            .to_string()
            .contains("safe relative path")
    );
}

#[test]
fn roon_convolver_rejects_unequal_lengths_and_unknown_channels() {
    let mut output = make_test_output();
    add_convolution(&mut output, "left", "left.wav");
    add_convolution(&mut output, "right", "right.wav");
    let resources = [
        resource("left.wav", test_wav(48_000, 1, 64)),
        resource("right.wav", test_wav(48_000, 1, 128)),
    ];
    assert!(
        build_roon_convolution_archive(&output, &resources, 48_000.0)
            .unwrap_err()
            .to_string()
            .contains("equal impulse-response lengths")
    );

    let mut unknown = output.clone();
    let mut chain = unknown.channels.remove("right").unwrap();
    chain.channel = "aux".to_string();
    unknown.channels.insert("aux".to_string(), chain);
    assert!(
        build_roon_convolution_archive(&unknown, &resources, 48_000.0)
            .unwrap_err()
            .to_string()
            .contains("does not know the WAVE channel mapping")
    );
}

fn add_convolution(output: &mut DspGraph, channel: &str, path: &str) {
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

fn convolution_path<'a>(output: &'a DspGraph, channel: &str) -> &'a str {
    output.channels[channel]
        .plugins
        .iter()
        .find(|plugin| plugin.plugin_type == "convolution")
        .unwrap()
        .parameters
        .get("ir_file")
        .and_then(serde_json::Value::as_str)
        .unwrap()
}

fn resource(reference: &str, bytes: Vec<u8>) -> ConvolutionResource {
    ConvolutionResource {
        reference: reference.to_string(),
        bytes: bytes.into(),
    }
}

fn test_wav(sample_rate: u32, channels: u16, frames: usize) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
        for frame in 0..frames {
            for channel in 0..channels {
                writer
                    .write_sample(if frame == 0 && channel == 0 {
                        1.0_f32
                    } else {
                        0.0
                    })
                    .unwrap();
            }
        }
        writer.finalize().unwrap();
    }
    cursor.into_inner()
}
