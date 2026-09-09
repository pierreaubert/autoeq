//! Analytic pressure-transfer fixtures; not measured spatial validation.
use super::*;
use roomeq_model::DriverDspChain;

fn driver(name: &str, index: usize, gain: f64, delay: f64) -> DriverDspChain {
    DriverDspChain {
        name: name.into(),
        index,
        plugins: vec![
            roomeq_engine::output::create_gain_plugin(gain),
            roomeq_engine::output::create_delay_plugin(delay),
        ],
        initial_curve: None,
    }
}

fn capture(low: f64, high: f64, level: f64, delay: f64) -> Curve {
    let freq = ndarray::Array1::logspace(10.0, low.log10(), high.log10(), 301);
    Curve {
        spl: ndarray::Array1::from_elem(freq.len(), level),
        phase: Some(freq.mapv(|f| -360.0 * f * delay)),
        freq,
        ..Default::default()
    }
}

fn fixture(
    tweeter_low: f64,
) -> (
    RoomOptimizationResult,
    BTreeMap<String, Vec<Curve>>,
    RoomConfig,
) {
    let mut result = crate::test_fixtures::single_channel_room_result("left");
    let chain = result.channels.get_mut("left").unwrap();
    chain.plugins.clear();
    chain.drivers = Some(vec![
        driver("woofer", 0, -1.0, 0.0),
        driver("tweeter", 1, -3.0, 0.1),
    ]);
    let physical = BTreeMap::from([
        ("woofer".into(), vec![capture(20.0, 20_000.0, 80.0, 0.003)]),
        (
            "tweeter".into(),
            vec![capture(tweeter_low, 20_000.0, 77.0, 0.0032)],
        ),
    ]);
    let mut config = RoomConfig::default();
    config.optimizer.min_freq = 20.0;
    config.optimizer.max_freq = 20_000.0;
    (result, physical, config)
}

#[test]
fn missing_low_driver_support_cannot_silently_remove_measured_bass() {
    let (result, physical, config) = fixture(200.0);
    let replay = replay_final_physical_seat(
        &result,
        &physical,
        "left",
        0,
        &config,
        48_000.0,
        Path::new("."),
        "training",
    );
    let error = replay
        .err()
        .expect("unmeasured tweeter bass must not silently shrink final playback");
    assert!(
        error
            .to_string()
            .contains("insufficient summation evidence"),
        "{error}"
    );
    assert!(error.to_string().contains("tweeter"), "{error}");
}

#[test]
fn driver_fir_is_not_mistaken_for_parent_retained_fir() {
    let (mut result, mut physical, config) = fixture(20.0);
    let directory = tempfile::tempdir().unwrap();
    let convolution = |path: &str| PluginConfigWrapper {
        plugin_type: "convolution".into(),
        parameters: serde_json::json!({"ir_file": path}),
    };
    let chain = result.channels.get_mut("left").unwrap();
    chain.plugins = vec![convolution("shared.wav")];
    chain.drivers = Some(vec![DriverDspChain {
        name: "woofer".into(),
        index: 0,
        plugins: vec![convolution("driver.wav")],
        initial_curve: None,
    }]);
    result.channel_results.get_mut("left").unwrap().fir_coeffs = Some(vec![0.25]);
    physical.insert("woofer".into(), vec![capture(20.0, 20_000.0, 80.0, 0.0)]);
    physical.remove("tweeter");
    for (name, tap) in [("shared.wav", 0.25_f32), ("driver.wav", 0.5_f32)] {
        let mut writer = hound::WavWriter::create(
            directory.path().join(name),
            hound::WavSpec {
                channels: 1,
                sample_rate: 48_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        writer.write_sample(tap).unwrap();
        writer.finalize().unwrap();
    }
    let replay = replay_final_physical_seat(
        &result,
        &physical,
        "left",
        0,
        &config,
        48_000.0,
        directory.path(),
        "training",
    )
    .unwrap();
    let expected = 80.0 + 20.0 * (0.25_f64 * 0.5).log10();
    assert!(
        replay
            .delivered
            .spl
            .iter()
            .all(|db| (db - expected).abs() < 1e-9)
    );
    assert!(
        replay
            .baseline
            .spl
            .iter()
            .all(|db| (db - 80.0).abs() < 1e-9)
    );
    // In-memory replay may use the retained parent taps for that exact declared
    // channel resource; the distinct driver still needs its own sidecar.
    std::fs::remove_file(directory.path().join("shared.wav")).unwrap();
    let retained = replay_final_physical_seat(
        &result,
        &physical,
        "left",
        0,
        &config,
        48_000.0,
        directory.path(),
        "training",
    )
    .unwrap();
    assert!(
        retained
            .delivered
            .spl
            .iter()
            .all(|db| (db - expected).abs() < 1e-9)
    );
    // An absent driver resource must not be replaced with the parent's taps.
    std::fs::remove_file(directory.path().join("driver.wav")).unwrap();
    assert!(
        replay_final_physical_seat(
            &result,
            &physical,
            "left",
            0,
            &config,
            48_000.0,
            directory.path(),
            "training"
        )
        .is_err()
    );
}

#[test]
fn explicit_requested_lower_bound_can_use_common_driver_support() {
    let (result, physical, mut config) = fixture(200.0);
    config.optimizer.min_freq = 250.0;
    let replay = replay_final_physical_seat(
        &result,
        &physical,
        "left",
        0,
        &config,
        48_000.0,
        Path::new("."),
        "training",
    )
    .unwrap();
    assert!(replay.delivered.freq[0] >= 250.0);
    assert!(*replay.delivered.freq.last().unwrap() > 19_999.0);
}

fn acoustic(branch: usize, seat: usize, f: f64) -> num_complex::Complex64 {
    use num_complex::Complex64 as C;
    // RMS pressure relative to 20 uPa, e^(+jwt). Different acoustic poles,
    // output levels and seat gains; no ideal acoustic complementarity assumed.
    if branch == 0 {
        let s = C::new(0.0, f / 400.0);
        10.0_f64.powf((80.0 + seat as f64) / 20.0) / (1.0 + s).powu(2)
    } else {
        let s = C::new(0.0, f / 800.0);
        10.0_f64.powf((77.0 - seat as f64) / 20.0) * s / (1.0 + s)
    }
}

fn lr24(f: f64, cutoff: f64, fs: f64, high: bool) -> num_complex::Complex64 {
    // Bilinear-transform reference: two cascaded second-order Butterworth
    // sections. Independent of production Biquad/electrical replay helpers.
    let s = num_complex::Complex64::new(
        0.0,
        (std::f64::consts::PI * f / fs).tan() / (std::f64::consts::PI * cutoff / fs).tan(),
    );
    let denominator = (s * s + std::f64::consts::SQRT_2 * s + 1.0).powu(2);
    if high {
        s.powu(4) / denominator
    } else {
        1.0 / denominator
    }
}

#[test]
fn asymmetric_driver_replay_matches_independent_complex_crossover_sum() {
    use num_complex::Complex64 as C;
    for fs in [44_100.0, 48_000.0, 96_000.0] {
        let (mut result, _, config) = fixture(20.0);
        let chain = result.channels.get_mut("left").unwrap();
        chain.plugins = vec![
            roomeq_engine::output::create_gain_plugin(-2.0),
            roomeq_engine::output::create_delay_plugin(0.07),
        ];
        let drivers = chain.drivers.as_mut().unwrap();
        drivers[0]
            .plugins
            .push(roomeq_engine::output::create_crossover_plugin(
                "LR24", 950.0, "low",
            ));
        drivers[1].plugins[0] = roomeq_engine::output::create_gain_plugin_with_invert(-3.0, true);
        drivers[1]
            .plugins
            .push(roomeq_engine::output::create_crossover_plugin(
                "LR24", 1250.0, "high",
            ));
        // Replay serialized plugin ownership, not a parallel parameter list.
        let graph: roomeq_model::DspGraph =
            serde_json::from_slice(&serde_json::to_vec(&result.to_dsp_chain_output()).unwrap())
                .unwrap();
        result.channels = graph.channels;
        let mut physical = BTreeMap::new();
        for (branch, name) in ["woofer", "tweeter"].into_iter().enumerate() {
            let mut seats = Vec::new();
            for seat in 0..2 {
                let freq = ndarray::Array1::logspace(
                    10.0,
                    20.0_f64.log10(),
                    20_000.0_f64.log10(),
                    1001 + branch * 100,
                );
                seats.push(Curve {
                    spl: freq.mapv(|f| 20.0 * acoustic(branch, seat, f).norm().log10()),
                    phase: Some(freq.mapv(|f| acoustic(branch, seat, f).arg().to_degrees())),
                    freq,
                    ..Default::default()
                });
            }
            physical.insert(name.into(), seats);
        }
        for seat in 0..2 {
            let replay = replay_final_physical_seat(
                &result,
                &physical,
                "left",
                seat,
                &config,
                fs,
                Path::new("."),
                "training",
            )
            .unwrap();
            assert_eq!(replay.physical_outputs, ["woofer", "tweeter"]);
            assert!(replay.delivered.freq[0] < 20.001);
            assert!(*replay.delivered.freq.last().unwrap() > 19_999.0);
            let mut max_relative_error = 0.0_f64;
            for (i, &f) in replay.delivered.freq.iter().enumerate() {
                let woofer =
                    acoustic(0, seat, f) * lr24(f, 950.0, fs, false) * 10.0_f64.powf(-1.0 / 20.0);
                let tweeter = -acoustic(1, seat, f)
                    * lr24(f, 1250.0, fs, true)
                    * 10.0_f64.powf(-3.0 / 20.0)
                    * C::from_polar(1.0, -std::f64::consts::TAU * f * 0.0001);
                let expected = (woofer + tweeter)
                    * 10.0_f64.powf(-2.0 / 20.0)
                    * C::from_polar(1.0, -std::f64::consts::TAU * f * 0.00007);
                let actual = C::from_polar(
                    10.0_f64.powf(replay.delivered.spl[i] / 20.0),
                    replay.delivered.phase.as_ref().unwrap()[i].to_radians(),
                );
                let error = (actual - expected).norm() / expected.norm().max(1e-9);
                max_relative_error = max_relative_error.max(error);
                assert!(
                    error < 0.002,
                    "{fs} Hz, seat {seat}, {f} Hz: relative complex error {error}"
                );
            }
            eprintln!(
                "asymmetric driver reference: fs={fs}, seat={seat}, max_relative_complex_error={max_relative_error}"
            );
            let original_channels = result.channels.clone();
            result
                .channels
                .get_mut("left")
                .unwrap()
                .drivers
                .as_mut()
                .unwrap()
                .reverse();
            let reordered = replay_final_physical_seat(
                &result,
                &physical,
                "left",
                seat,
                &config,
                fs,
                Path::new("."),
                "training",
            )
            .unwrap();
            assert_eq!(replay.delivered.spl, reordered.delivered.spl);
            assert_eq!(replay.delivered.phase, reordered.delivered.phase);
            result.channels = original_channels.clone();
            // Deliberate ownership fault: omit the serialized tweeter inversion.
            result
                .channels
                .get_mut("left")
                .unwrap()
                .drivers
                .as_mut()
                .unwrap()[1]
                .plugins[0] = roomeq_engine::output::create_gain_plugin(-3.0);
            let wrong = replay_final_physical_seat(
                &result,
                &physical,
                "left",
                seat,
                &config,
                fs,
                Path::new("."),
                "training",
            )
            .unwrap();
            let mut polarity_error = 0.0_f64;
            for i in 0..replay.delivered.freq.len() {
                let reference = C::from_polar(
                    10.0_f64.powf(replay.delivered.spl[i] / 20.0),
                    replay.delivered.phase.as_ref().unwrap()[i].to_radians(),
                );
                let mutant = C::from_polar(
                    10.0_f64.powf(wrong.delivered.spl[i] / 20.0),
                    wrong.delivered.phase.as_ref().unwrap()[i].to_radians(),
                );
                polarity_error = polarity_error.max((mutant - reference).norm() / reference.norm());
            }
            assert!(
                polarity_error > 0.5,
                "the phasor fixture must detect missing driver polarity"
            );
            result.channels = original_channels;
        }
    }
}
