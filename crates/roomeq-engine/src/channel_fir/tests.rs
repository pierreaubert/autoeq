use autoeq_core::Curve;
use ndarray::Array1;
use roomeq_model::{
    FirConfig, MixedPhaseSerdeConfig, OptimizerConfig, RoomConfig, TargetResponseConfig,
    UserPreference,
};

use super::*;
use crate::channel_preprocessing::PreprocessedFeatures;
use crate::channel_result::ConvolutionSidecarReference;
use crate::channel_target::build_target_context;
use crate::eq::EqResources;
use crate::{PreparedCea2034, PreparedChannelMeasurements};

fn curve(with_phase: bool) -> Curve {
    let frequency = Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 96);
    let spl = frequency
        .iter()
        .map(|frequency| 80.0 + 5.0 * (-((*frequency - 120.0) / 15.0).powi(2)).exp())
        .collect::<Vec<_>>();
    Curve {
        freq: frequency.clone(),
        spl: Array1::from(spl),
        phase: with_phase.then(|| Array1::zeros(frequency.len())),
        ..Curve::default()
    }
}

fn prepared(curve: Curve) -> PreparedChannelInput {
    PreparedChannelInput::new(
        PreparedChannelMeasurements::new(curve.clone(), vec![curve], false),
        None,
        PreparedCea2034::default(),
        EqResources::default(),
    )
}

fn preprocessed(curve: &Curve) -> PreprocessedFeatures {
    PreprocessedFeatures {
        curve: curve.clone(),
        curve_for_optim: curve.clone(),
        excursion_filters: Vec::new(),
        cea2034_filters: Vec::new(),
        cea2034_plugins: Vec::new(),
        optimizer_evidence: Vec::new(),
        broadband_plugins: Vec::new(),
        broadband_biquads: Vec::new(),
        broadband_mean_shift: 0.0,
        broadband_enabled: false,
        norm_range: Some((20.0, 20_000.0)),
        score_min_freq: 20.0,
    }
}

fn config() -> RoomConfig {
    RoomConfig {
        optimizer: OptimizerConfig {
            num_filters: 1,
            max_iter: 10,
            population: 6,
            min_freq: 20.0,
            max_freq: 500.0,
            psychoacoustic: false,
            refine: false,
            fir: Some(FirConfig {
                taps: 64,
                phase: "linear".to_string(),
                ..FirConfig::default()
            }),
            ..OptimizerConfig::default()
        },
        ..RoomConfig::default()
    }
}

fn reference(name: &str) -> ConvolutionSidecarReference {
    ConvolutionSidecarReference::new(name).unwrap()
}

#[test]
fn spatial_fir_native_stop_does_not_return_coefficients() {
    use roomeq_model::{MultiMeasurementConfig, MultiMeasurementStrategy};
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    let curve = curve(false);
    let prepared = PreparedChannelInput::new(
        PreparedChannelMeasurements::new(curve.clone(), vec![curve.clone(), curve.clone()], true),
        None, PreparedCea2034::default(), EqResources::default(),
    );
    for phase in ["linear", "minimum"] {
        for algorithm in ["autoeq:de", "autoeq:cmaes"] {
            let mut room_config = config();
            room_config.optimizer.algorithm = algorithm.into();
            room_config.optimizer.strategy = "rand1bin".into();
            room_config.optimizer.max_iter = 1000;
            room_config.optimizer.seed = Some(7);
            room_config.optimizer.fir.as_mut().unwrap().phase = phase.into();
            room_config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
                strategy: MultiMeasurementStrategy::WeightedSum,
                weights: Some(vec![0.5, 0.5]), ..Default::default()
            });
            let target = build_target_context("left", &room_config, &curve, None);
            let features = preprocessed(&curve);
            let resources = EqResources::default();
            let request = FirChannelRequest {
                mode: FirChannelMode::Hybrid, channel_name: "left", prepared: &prepared,
                room_config: &room_config, sample_rate: 48_000.0, target: &target,
                preprocessed: &features, optimizer: &room_config.optimizer,
                eq_resources: &resources, sidecar_reference: reference("stop.wav"), callback: None,
            };
            // Two seats plus neutral, and the additional representative in the
            // linear basis. Continue through every vertex, then stop natively.
            let vertices = if phase == "linear" { 4 } else { 3 };
            let calls = Arc::new(AtomicUsize::new(0));
            let observed = calls.clone();
            let progress = progress::FirProgress::new(Some(Box::new(move |_, _, _| {
                if observed.fetch_add(1, Ordering::Relaxed) >= vertices {
                    autoeq_optim::de::CallbackAction::Stop
                } else {
                    autoeq_optim::de::CallbackAction::Continue
                }
            })));
            let result = if phase == "linear" {
                let mut representative = vec![0.0; 64];
                representative[32] = 1.0;
                spatial_linear::optimize(&request, &curve, &[], representative, &progress)
            } else {
                spatial_realized::optimize(&request, &curve, &[], &progress)
            };
            let error = result.expect_err("native Stop must not deliver FIR coefficients");
            assert!(error.to_string().contains("scalar optimization stopped by user"), "{phase}/{algorithm}: {error}");
            assert_eq!(calls.load(Ordering::Relaxed), vertices + 1);
        }
    }
}

#[test]
fn hybrid_callback_survives_iir_into_spatial_fir() {
    use roomeq_model::{MultiMeasurementConfig, MultiMeasurementStrategy};
    let curve = curve(false);
    let prepared = PreparedChannelInput::new(
        PreparedChannelMeasurements::new(curve.clone(), vec![curve.clone(), curve.clone()], true),
        None, PreparedCea2034::default(), EqResources::default(),
    );
    for phase in ["linear", "minimum"] {
        let mut room_config = config();
        room_config.optimizer.algorithm = "autoeq:de".into();
        room_config.optimizer.strategy = "rand1bin".into();
        room_config.optimizer.max_iter = 10;
        room_config.optimizer.seed = Some(7);
        room_config.optimizer.fir.as_mut().unwrap().phase = phase.into();
        room_config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::WeightedSum,
            weights: Some(vec![0.5, 0.5]), ..Default::default()
        });
        let target = build_target_context("left", &room_config, &curve, None);
        let features = preprocessed(&curve);
        let resources = EqResources::default();
        let mut previous = 0;
        let result = process_fir_channel(FirChannelRequest {
            mode: FirChannelMode::Hybrid, channel_name: "left", prepared: &prepared,
            room_config: &room_config, sample_rate: 48_000.0, target: &target,
            preprocessed: &features, optimizer: &room_config.optimizer,
            eq_resources: &resources, sidecar_reference: reference("stop_after_iir.wav"),
            callback: Some(Box::new(move |iteration, _, _| {
                let action = if iteration < previous {
                    autoeq_optim::de::CallbackAction::Stop
                } else { autoeq_optim::de::CallbackAction::Continue };
                previous = iteration;
                action
            })),
        });
        let error = result.err().expect("Stop at the first FIR vertex must survive IIR dispatch");
        assert!(error.to_string().contains("Hybrid FIR stopped by progress callback"), "{phase}: {error}");
    }
}

#[test]
fn hybrid_narrow_band_preserves_fixed_target_in_direct_dtft() {
    // Matrix-shaped LFE evidence through real channel preparation. The request
    // starts below measured support; dispatch must clamp it before designing.
    // Routed system acceptance is separately exercised by the matrix shell test.
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/narrow_lfe.json")).unwrap();
    let curve: Curve = serde_json::from_value(fixture["measurement"].clone()).unwrap();
    let mut room_config = config();
    room_config.optimizer = serde_json::from_value(fixture["optimizer"].clone()).unwrap();
    room_config.optimizer.min_freq = 20.0;
    let prepared = prepared(curve.clone());
    let resources = EqResources::default();
    let execution = crate::channel_execution::prepare_channel_execution(
        "LFE",
        &prepared,
        &room_config,
        96_000.0,
        None,
    )
    .unwrap();
    let result = crate::channel_execution::execute_prepared_channel(
        "LFE",
        &prepared,
        &room_config,
        96_000.0,
        &execution,
        &resources,
        Some(reference("lfe_probe.wav")),
        None,
    )
    .unwrap();
    assert_eq!(result.fir_coeffs.as_ref().unwrap().len(), 1920);
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed_calls = std::sync::Arc::clone(&calls);
    let observed = crate::channel_execution::execute_prepared_channel(
        "LFE",
        &prepared,
        &room_config,
        96_000.0,
        &execution,
        &resources,
        Some(reference("lfe_observed.wav")),
        Some(Box::new(move |_, _, _| {
            observed_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            autoeq_optim::de::CallbackAction::Continue
        })),
    )
    .unwrap();
    assert!(calls.load(std::sync::atomic::Ordering::Relaxed) > 0);
    assert_eq!(
        observed.fir_coeffs, result.fir_coeffs,
        "progress observation must preserve adaptive Hybrid output"
    );
    assert_eq!(observed.filters.len(), result.filters.len());
    let observed_iir =
        response::compute_peq_complex_response(&observed.filters, &curve.freq, 96_000.0);
    let unobserved_iir =
        response::compute_peq_complex_response(&result.filters, &curve.freq, 96_000.0);
    assert!(
        observed_iir
            .iter()
            .zip(unobserved_iir.iter())
            .all(|(a, b)| (*a - *b).norm() < 1e-12)
    );
    let stopped = crate::channel_execution::execute_prepared_channel(
        "LFE",
        &prepared,
        &room_config,
        96_000.0,
        &execution,
        &resources,
        Some(reference("lfe_stopped.wav")),
        Some(Box::new(|_, _, _| autoeq_optim::de::CallbackAction::Stop)),
    );
    assert!(
        stopped
            .err()
            .expect("cancellation must stop adaptive selection")
            .to_string()
            .contains("stopped")
    );
    let target = crate::fir::prepared_fir_target_curve(&curve, &room_config.optimizer, &resources);
    let published = result
        .channel
        .target_curve
        .as_ref()
        .expect("FIR design target");
    assert_eq!(published.freq, curve.freq.to_vec());
    assert!(
        published
            .spl
            .iter()
            .zip(&target.spl)
            .all(|(a, b)| (a - b).abs() < 1e-12),
        "published target must match the FIR design reference"
    );
    let iir = response::apply_complex_response(
        &curve,
        &response::compute_peq_complex_response(&result.filters, &curve.freq, 96_000.0),
    );
    for (index, &f) in curve.freq.iter().enumerate().filter(|(_, f)| **f <= 160.0) {
        let (re, im) = result.fir_coeffs.as_ref().unwrap().iter().enumerate().fold(
            (0.0, 0.0),
            |(re, im), (n, h)| {
                let phase = -std::f64::consts::TAU * f * n as f64 / 96_000.0;
                (re + h * phase.cos(), im + h * phase.sin())
            },
        );
        let fir_db = 20.0 * re.hypot(im).log10();
        let final_error = iir.spl[index] + fir_db - target.spl[index];
        assert!(
            final_error.is_finite() && final_error.abs() < 0.1,
            "{f} Hz: delivered target error {final_error} dB"
        );
    }
}

#[test]
fn phase_linear_returns_required_sidecar_and_in_memory_coefficients() {
    let curve = curve(false);
    let prepared = prepared(curve.clone());
    let room_config = config();
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let mut features = preprocessed(&curve);
    features
        .broadband_plugins
        .push(crate::output::create_gain_plugin(-1.0));
    let result = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::PhaseLinear,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_fir_48000hz.wav"),
        callback: None,
    })
    .unwrap();

    let coefficients = result.fir_coeffs.as_ref().unwrap();
    assert_eq!(coefficients.len(), 64);
    let peak_index = coefficients
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(index, _)| index)
        .unwrap();
    assert!(
        coefficients
            .iter()
            .enumerate()
            .any(|(index, coefficient)| index != peak_index && coefficient.abs() > 1.0e-6),
        "a non-flat measurement must not produce an identity FIR"
    );
    let sidecar = result.convolution_sidecar.unwrap();
    assert!(sidecar.required);
    assert_eq!(sidecar.reference.filename(), "left_fir_48000hz.wav");
    assert_eq!(result.channel.plugins[0].plugin_type, "gain");
    assert_eq!(result.channel.plugins[1].plugin_type, "convolution");
}

#[test]
fn hybrid_returns_iir_filters_and_required_residual_sidecar() {
    let curve = curve(false);
    let prepared = prepared(curve.clone());
    let mut room_config = config();
    room_config.optimizer.seed = Some(42);
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);
    let result = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::Hybrid,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_residual_fir_48000hz.wav"),
        callback: None,
    })
    .unwrap();

    let residual = response::apply_complex_response(
        &curve,
        &response::compute_peq_complex_response(&result.filters, &curve.freq, 48_000.0),
    );
    let original_target =
        crate::fir::prepared_fir_target_curve(&curve, &room_config.optimizer, &resources);
    let candidate_target =
        crate::fir::prepared_fir_target_curve(&residual, &room_config.optimizer, &resources);
    assert!(
        (&original_target.spl - &candidate_target.spl)
            .iter()
            .any(|delta| delta.abs() > 1e-3),
        "fixture must challenge a candidate-induced target-level change"
    );
    let expected = crate::fir::generate_fir_correction_prepared(
        &residual,
        &room_config.optimizer,
        &original_target,
        48_000.0,
    )
    .unwrap();
    assert_eq!(
        result.fir_coeffs.as_ref().unwrap(),
        &expected,
        "hybrid FIR must keep the pre-IIR target reference"
    );
    assert!(result.convolution_sidecar.unwrap().required);
    assert!(result.channel.plugins.iter().any(|plugin| {
        plugin.plugin_type == "convolution"
            && plugin.parameters["ir_file"] == "left_residual_fir_48000hz.wav"
    }));
}

#[test]
fn hybrid_honours_multi_measurement_weights() {
    // F04: Hybrid must optimize the configured multi-measurement objective,
    // not just the representative curve. Opposed seats with swapped weights
    // must yield different IIR solutions.
    use roomeq_model::{MultiMeasurementConfig, MultiMeasurementStrategy};
    let frequency = Array1::logspace(10.0, f64::log10(20.0), f64::log10(500.0), 48);
    let seat_peak = Curve {
        freq: frequency.clone(),
        spl: Array1::from(
            frequency
                .iter()
                .map(|f| 80.0 + 6.0 * (-((f - 120.0) / 10.0).powi(2)).exp())
                .collect::<Vec<_>>(),
        ),
        ..Curve::default()
    };
    let seat_flat = Curve {
        freq: frequency.clone(),
        spl: Array1::from_elem(frequency.len(), 80.0),
        ..Curve::default()
    };
    let representative = Curve {
        freq: frequency.clone(),
        spl: (&seat_peak.spl + &seat_flat.spl) / 2.0,
        ..Curve::default()
    };
    let run_with_weights = |weights: Vec<f64>| {
        let prepared = PreparedChannelInput::new(
            PreparedChannelMeasurements::new(
                representative.clone(),
                vec![seat_peak.clone(), seat_flat.clone()],
                true,
            ),
            None,
            PreparedCea2034::default(),
            EqResources::default(),
        );
        let mut room_config = config();
        room_config.optimizer.num_filters = 1;
        room_config.optimizer.max_iter = 50;
        room_config.optimizer.population = 12;
        room_config.optimizer.seed = Some(42);
        room_config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::WeightedSum,
            weights: Some(weights),
            ..MultiMeasurementConfig::default()
        });
        let resources = EqResources::default();
        let target = build_target_context("left", &room_config, &representative, None);
        let features = preprocessed(&representative);
        process_fir_channel(FirChannelRequest {
            mode: FirChannelMode::Hybrid,
            channel_name: "left",
            prepared: &prepared,
            room_config: &room_config,
            sample_rate: 48_000.0,
            target: &target,
            preprocessed: &features,
            optimizer: &room_config.optimizer,
            eq_resources: &resources,
            sidecar_reference: reference("left_residual_fir_48000hz.wav"),
            callback: None,
        })
        .unwrap()
    };
    let peak_first = run_with_weights(vec![1.0, 0.0]);
    let flat_first = run_with_weights(vec![0.0, 1.0]);
    let response_at_120 = |filters: &[Biquad]| {
        let response = autoeq_core::response::compute_peq_complex_response(
            filters,
            &Array1::from(vec![120.0]),
            48_000.0,
        );
        20.0 * response[0].norm().log10()
    };
    let peak_cut = response_at_120(&peak_first.filters);
    let flat_cut = response_at_120(&flat_first.filters);
    assert!(
        (peak_cut - flat_cut).abs() > 2.0,
        "hybrid ignored multi-measurement weights: peak-weighted {peak_cut:.2} dB vs flat-weighted {flat_cut:.2} dB"
    );
    // The callback path must use the same multi-measurement dispatch.
    let prepared = PreparedChannelInput::new(
        PreparedChannelMeasurements::new(
            representative.clone(),
            vec![seat_peak.clone(), seat_flat.clone()],
            true,
        ),
        None,
        PreparedCea2034::default(),
        EqResources::default(),
    );
    let mut room_config = config();
    room_config.optimizer.num_filters = 1;
    room_config.optimizer.max_iter = 50;
    room_config.optimizer.population = 12;
    room_config.optimizer.seed = Some(42);
    room_config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
        strategy: MultiMeasurementStrategy::WeightedSum,
        weights: Some(vec![1.0, 0.0]),
        ..MultiMeasurementConfig::default()
    });
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &representative, None);
    let features = preprocessed(&representative);
    let callback: autoeq_optim::optim::OptimProgressCallback =
        Box::new(|_, _, _| autoeq_optim::de::CallbackAction::Continue);
    let with_callback = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::Hybrid,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_residual_fir_48000hz.wav"),
        callback: Some(callback),
    })
    .unwrap();
    let callback_cut = response_at_120(&with_callback.filters);
    assert!(
        (callback_cut - peak_cut).abs() < 0.5,
        "hybrid callback path diverged from multi-measurement objective: {callback_cut:.2} vs {peak_cut:.2} dB"
    );
}

/// Audit F04: the residual FIR must not erase the IIR's spatial objective.
/// Analytic seats, not physical held-outs or a claim about workflow acceptance.
#[test]
fn hybrid_complete_chain_preserves_seat_weight_choice() {
    use roomeq_model::{MultiMeasurementConfig, MultiMeasurementStrategy};
    let frequencies = Array1::logspace(10.0, 20.0_f64.log10(), 500.0_f64.log10(), 192);
    let flat = Curve { freq: frequencies.clone(), spl: Array1::from_elem(192, 80.0), ..Curve::default() };
    let mut peak = flat.clone();
    peak.spl = frequencies.mapv(|f| 80.0 + 6.0 * (-((f - 120.0) / 30.0).powi(2)).exp());
    let mut representative = flat.clone();
    representative.spl = (&peak.spl + &flat.spl) / 2.0;
    let prepared = PreparedChannelInput::new(
        PreparedChannelMeasurements::new(representative.clone(), vec![peak, flat], true),
        None, PreparedCea2034::default(), EqResources::default(),
    );
    let mut outcomes = Vec::new();
    for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
    for (strategy, weights) in [
        (MultiMeasurementStrategy::WeightedSum, Some(vec![1.0, 0.0])),
        (MultiMeasurementStrategy::WeightedSum, Some(vec![0.0, 1.0])),
        (MultiMeasurementStrategy::Minimax, None),
    ] {
        let mut room_config = config();
        room_config.optimizer.num_filters = 1;
        room_config.optimizer.max_iter = 120;
        room_config.optimizer.population = 12;
        room_config.optimizer.seed = Some(42);
        room_config.optimizer.fir.as_mut().unwrap().taps = 8192;
        room_config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
            strategy, weights: weights.clone(), ..MultiMeasurementConfig::default()
        });
        let resources = EqResources::default();
        let target = build_target_context("left", &room_config, &representative, None);
        let features = preprocessed(&representative);
        let result = process_fir_channel(FirChannelRequest {
            mode: FirChannelMode::Hybrid, channel_name: "left", prepared: &prepared,
            room_config: &room_config, sample_rate, target: &target,
            preprocessed: &features, optimizer: &room_config.optimizer,
            eq_resources: &resources, sidecar_reference: reference("spatial_residual.wav"),
            callback: None,
        }).unwrap();
        let iir = response::compute_peq_complex_response(&result.filters, &Array1::from_vec(vec![120.0]), sample_rate)[0];
        // Independent direct DTFT of the actual in-memory FIR coefficients.
        let taps = result.fir_coeffs.as_ref().unwrap();
        let fir = taps.iter().enumerate().fold(num_complex::Complex64::new(0.0, 0.0), |sum, (n, tap)| {
            sum + num_complex::Complex64::from_polar(*tap, -std::f64::consts::TAU * 120.0 * n as f64 / sample_rate)
        });
        // Serialize the actual chain and round-trip the taps through a mono
        // float WAV before canonical replay. Compare against an independent
        // direct DTFT; this is not an external playback-backend test.
        let chain = serde_json::from_slice(&serde_json::to_vec(&result.channel).unwrap()).unwrap();
        let mut bytes = std::io::Cursor::new(Vec::new());
        {
            let mut wav = hound::WavWriter::new(&mut bytes, hound::WavSpec {
                channels: 1, sample_rate: sample_rate as u32, bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            }).unwrap();
            for tap in taps { wav.write_sample(*tap as f32).unwrap(); }
            wav.finalize().unwrap();
        }
        let mut wav = hound::WavReader::new(std::io::Cursor::new(bytes.into_inner())).unwrap();
        struct Sidecar(Vec<f64>, u32);
        impl crate::dsp_realization::ConvolutionIrProvider for Sidecar {
            fn taps(&mut self, name: &str, rate: u32) -> Result<&[f64]> {
                assert_eq!(name, "spatial_residual.wav");
                assert_eq!(rate, self.1);
                Ok(&self.0)
            }
        }
        let mut sidecar = Sidecar(wav.samples::<f32>().map(|v| v.unwrap() as f64).collect(), sample_rate as u32);
        let mut replay = crate::dsp_realization::RealizedDsp::new(&chain, sample_rate, &mut sidecar).unwrap();
        let verification_grid = Array1::logspace(10.0, 20.0_f64.log10(), 500.0_f64.log10(), 257);
        let iir_grid = response::compute_peq_complex_response(&result.filters, &verification_grid, sample_rate);
        let mut maximum_roundtrip_error = 0.0_f64;
        let mut maximum_fir_group_delay_error_samples = 0.0_f64;
        let native_peak = &prepared.measurements().individual()[0];
        let native_levels: Vec<_> = native_peak.freq.iter().zip(&native_peak.spl)
            .filter(|(frequency, _)| **frequency >= 20.0 && **frequency <= 500.0)
            .map(|(_, level)| *level - 80.0).collect();
        let peak_reference = native_levels.iter().sum::<f64>() / native_levels.len() as f64;
        let mut peak_errors = Vec::new();
        let mut flat_errors = Vec::new();
        for (bin, &frequency) in verification_grid.iter().enumerate() {
            let reference_fir: num_complex::Complex64 = taps.iter().enumerate().map(|(n, tap)|
                num_complex::Complex64::from_polar(*tap, -std::f64::consts::TAU * frequency * n as f64 / sample_rate)
            ).sum();
            let expected = iir_grid[bin] * reference_fir;
            // Exact DTFT derivative, not differencing wrapped phase:
            // group delay in samples = Re(sum n*h[n]*exp(-j*w*n) / H).
            let moment: num_complex::Complex64 = taps.iter().enumerate().map(|(n, tap)|
                num_complex::Complex64::from_polar(n as f64 * tap,
                    -std::f64::consts::TAU * frequency * n as f64 / sample_rate)
            ).sum();
            assert!(reference_fir.norm() > 1e-8, "group delay needs nonzero transfer");
            let delay_samples = (moment / reference_fir).re;
            maximum_fir_group_delay_error_samples = maximum_fir_group_delay_error_samples
                .max((delay_samples - (taps.len() / 2) as f64).abs());
            let correction_db = 20.0 * expected.norm().log10();
            peak_errors.push(6.0 * (-((frequency - 120.0) / 30.0).powi(2)).exp() - peak_reference + correction_db);
            flat_errors.push(correction_db);
            let error = (replay.response_at(frequency).unwrap() - expected).norm();
            maximum_roundtrip_error = maximum_roundtrip_error.max(error);
            assert!(error < 2e-6, "serialized Hybrid at {frequency} Hz: {error}");
        }
        // Independent log-frequency quadrature with the original native
        // reference level fixed before correction, matching the declared
        // normalized shape task without fitting away candidate gain changes.
        let rms = |errors: &[f64]| {
            let integral: f64 = (1..errors.len()).map(|i|
                (verification_grid[i] / verification_grid[i - 1]).ln()
                    * (errors[i - 1].powi(2) + errors[i].powi(2)) / 2.0
            ).sum();
            (integral / (500.0_f64 / 20.0).ln()).sqrt()
        };
        let peak_rms = rms(&peak_errors);
        let flat_rms = rms(&flat_errors);
        outcomes.push(serde_json::json!({
            "strategy": strategy, "weights": weights, "sample_rate_hz": sample_rate, "actual_fir_taps": taps.len(),
            "iir_db_at_120hz": 20.0 * iir.norm().log10(),
            "fir_db_at_120hz": 20.0 * fir.norm().log10(),
            "complete_chain_db_at_120hz": 20.0 * (iir * fir).norm().log10(),
            "json_and_float_wav_roundtrip_max_complex_error": maximum_roundtrip_error,
            "roundtrip_frequency_count": 257,
            "fir_duration_ms": 1000.0 * taps.len() as f64 / sample_rate,
            "peak_seat_shape_rms_db": peak_rms,
            "flat_seat_shape_rms_db": flat_rms,
            "worst_seat_shape_rms_db": peak_rms.max(flat_rms),
            "optimizer_evidence": result.optimizer_evidence,
            "expected_fir_delay_samples": taps.len() / 2,
            "max_fir_group_delay_error_samples": maximum_fir_group_delay_error_samples,
        }));
    }
    }
    let evidence_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qa");
    std::fs::create_dir_all(&evidence_directory).unwrap();
    std::fs::write(evidence_directory.join("hybrid-spatial-complete-chain.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "scope": "analytic_training_seat_in_memory_iir_plus_fir_not_workflow_acceptance_or_backend_render",
        "frequency_hz": 120.0, "peak_seat_db": 86.0, "flat_seat_db": 80.0, "outcomes": outcomes,
    })).unwrap()).unwrap();
    assert_eq!(outcomes.len(), 9);
    for pair in outcomes.chunks_exact(3) {
    for outcome in pair {
        assert!(outcome["max_fir_group_delay_error_samples"].as_f64().unwrap() < 0.05,
            "linear FIR timing changed over supported band: {outcome}");
    }
    assert_eq!(pair[0]["actual_fir_taps"], 8192);
    assert_eq!(pair[1]["actual_fir_taps"], 8192);
    assert_eq!(pair[2]["actual_fir_taps"], 8192);
    let peak_weighted = pair[0]["complete_chain_db_at_120hz"].as_f64().unwrap();
    let flat_weighted = pair[1]["complete_chain_db_at_120hz"].as_f64().unwrap();
    assert!(flat_weighted - peak_weighted > 2.0,
        "complete hybrid erased seat-weight choice: peak {peak_weighted:.3} dB, flat {flat_weighted:.3} dB");
    let minimax = pair[2]["worst_seat_shape_rms_db"].as_f64().unwrap();
    for single_seat in &pair[..2] {
        assert!(minimax + 0.05 < single_seat["worst_seat_shape_rms_db"].as_f64().unwrap(),
            "minimax must improve worst-seat error over a single-seat choice: {pair:?}");
    }
    }
}

#[test]
fn fir_assembly_exports_every_preprocessing_and_preference_stage() {
    let curve = curve(false);
    let prepared = prepared(curve.clone());
    let mut room_config = config();
    room_config.optimizer.target_response = Some(TargetResponseConfig {
        preference: UserPreference {
            bass_shelf_db: 2.0,
            ..Default::default()
        },
        ..Default::default()
    });
    let mut target = build_target_context("left", &room_config, &curve, None);
    target.cea2034_active = true;
    let mut features = preprocessed(&curve);
    let cea_filter = Biquad::new(
        math_audio_iir_fir::BiquadFilterType::Peak,
        1_000.0,
        48_000.0,
        1.0,
        -1.0,
    );
    features.cea2034_filters.push(cea_filter.clone());
    features
        .cea2034_plugins
        .push(crate::output::create_labeled_eq_plugin(
            &[cea_filter],
            "cea2034",
        ));
    features
        .broadband_plugins
        .push(crate::output::create_gain_plugin(-1.0));
    features.excursion_filters.push(Biquad::new(
        math_audio_iir_fir::BiquadFilterType::Highpass,
        30.0,
        48_000.0,
        std::f64::consts::FRAC_1_SQRT_2,
        0.0,
    ));
    let request = FirChannelRequest {
        mode: FirChannelMode::PhaseLinear,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &EqResources::default(),
        sidecar_reference: reference("left_fir_48000hz.wav"),
        callback: None,
    };
    let result = super::assemble::assemble_fir_result(
        &request,
        FirOptimizerOutput::PhaseLinear {
            coefficients: vec![1.0],
            sidecar_reference: request.sidecar_reference.clone(),
        },
        Vec::new(),
        None,
    )
    .unwrap();

    let labels: Vec<_> = result
        .channel
        .plugins
        .iter()
        .filter_map(|plugin| {
            plugin
                .parameters
                .get("label")
                .and_then(|value| value.as_str())
        })
        .collect();
    assert!(labels.contains(&"cea2034"));
    assert!(labels.contains(&"excursion_protection"));
    assert!(labels.contains(&"user_preference"));
    assert!(
        result
            .channel
            .plugins
            .iter()
            .any(|plugin| plugin.plugin_type == "gain")
    );
    assert!(
        result
            .channel
            .plugins
            .iter()
            .any(|plugin| plugin.plugin_type == "convolution")
    );
}

#[test]
fn minimum_phase_hybrid_preserves_seat_weights_without_linear_delay() {
    nonlinear_spatial_weight_fixture("minimum", false);
}

#[test]
fn kirkeby_hybrid_preserves_seat_weights_with_reference_phase() {
    nonlinear_spatial_weight_fixture("kirkeby", true);
}

#[test]
fn kirkeby_hybrid_preserves_seat_weights_without_phase_correction() {
    nonlinear_spatial_weight_fixture("kirkeby", false);
}

#[test]
fn kirkeby_hybrid_cannot_use_iir_phase_as_missing_acoustic_phase() {
    let measurement = curve(false);
    let prepared = prepared(measurement.clone());
    let features = preprocessed(&measurement);
    let mut room_config = config();
    let fir = room_config.optimizer.fir.as_mut().unwrap();
    fir.phase = "kirkeby".into();
    fir.correct_excess_phase = true;
    let target = build_target_context("left", &room_config, &measurement, None);
    let resources = EqResources::default();
    let result = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::Hybrid, channel_name: "left", prepared: &prepared,
        room_config: &room_config, sample_rate: 48_000.0, target: &target,
        preprocessed: &features, optimizer: &room_config.optimizer, eq_resources: &resources,
        sidecar_reference: reference("missing_phase.wav"), callback: None,
    });
    assert!(result.err().expect("missing acoustic phase must fail before IIR generation")
        .to_string().contains("requires acoustic phase"));
}

#[test]
fn hybrid_complete_transfer_is_invariant_to_weighted_seat_permutation() {
    use roomeq_model::{MultiMeasurementConfig, MultiMeasurementStrategy};
    let frequencies = Array1::logspace(10.0, 20.0_f64.log10(), 500.0_f64.log10(), 96);
    let flat = Curve { freq: frequencies.clone(), spl: Array1::from_elem(96, 80.0), ..Default::default() };
    let peak = Curve { freq: frequencies.clone(),
        spl: frequencies.mapv(|f| 80.0 + 6.0 * (-((f - 120.0) / 30.0).powi(2)).exp()), ..Default::default() };
    let representative = Curve { freq: frequencies, spl: (&peak.spl + &flat.spl) / 2.0, ..Default::default() };
    let replay_grid = Array1::logspace(10.0, 20.0_f64.log10(), 500.0_f64.log10(), 257);
    let mut records = Vec::new();
    let mut failures = Vec::new();
    for phase_kind in ["linear", "minimum"] {
        for sample_rate in [44_100.0, 48_000.0, 96_000.0] {
            let mut transfers = Vec::new();
            for reversed in [false, true] {
                let seats = if reversed { vec![flat.clone(), peak.clone()] } else { vec![peak.clone(), flat.clone()] };
                let weights = if reversed { vec![0.2, 0.8] } else { vec![0.8, 0.2] };
                let prepared = PreparedChannelInput::new(
                    PreparedChannelMeasurements::new(representative.clone(), seats, true),
                    None, PreparedCea2034::default(), EqResources::default());
                let mut room_config = config();
                room_config.optimizer.num_filters = 1;
                room_config.optimizer.algorithm = "autoeq:cobyla".into();
                room_config.optimizer.max_iter = 120;
                room_config.optimizer.seed = Some(42);
                let fir = room_config.optimizer.fir.as_mut().unwrap();
                fir.phase = phase_kind.into();
                fir.taps = 2048;
                room_config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
                    strategy: MultiMeasurementStrategy::WeightedSum, weights: Some(weights), ..Default::default()
                });
                let target = build_target_context("left", &room_config, &representative, None);
                let features = preprocessed(&representative);
                let resources = EqResources::default();
                let result = process_fir_channel(FirChannelRequest {
                    mode: FirChannelMode::Hybrid, channel_name: "left", prepared: &prepared,
                    room_config: &room_config, sample_rate, target: &target,
                    preprocessed: &features, optimizer: &room_config.optimizer,
                    eq_resources: &resources, sidecar_reference: reference("permuted.wav"), callback: None,
                }).unwrap();
                let iir = response::compute_peq_complex_response(&result.filters, &replay_grid, sample_rate);
                let taps = result.fir_coeffs.as_ref().unwrap();
                // H(exp(j*omega)) = sum_n h[n] exp(-j*omega*n), no FFT normalization.
                let transfer: Vec<_> = replay_grid.iter().zip(iir.iter()).map(|(&f, &iir)| {
                    let fir: num_complex::Complex64 = taps.iter().enumerate().map(|(n, &tap)|
                        num_complex::Complex64::from_polar(tap, -std::f64::consts::TAU * f * n as f64 / sample_rate)).sum();
                    iir * fir
                }).collect();
                records.push(serde_json::json!({"phase": phase_kind, "rate": sample_rate,
                    "reversed": reversed, "optimizer": result.optimizer_evidence}));
                transfers.push(transfer);
            }
            // Optimization tolerance, not float32 serialization tolerance: same
            // physical weighted problem must agree within 0.1 dB / 0.02 complex.
            let mut max_db: f64 = 0.0;
            let mut max_complex: f64 = 0.0;
            for (a, b) in transfers[0].iter().zip(&transfers[1]) {
                assert!(a.re.is_finite() && a.im.is_finite() && a.norm() > 0.0);
                assert!(b.re.is_finite() && b.im.is_finite() && b.norm() > 0.0);
                max_db = max_db.max((20.0 * (a.norm() / b.norm()).log10()).abs());
                max_complex = max_complex.max((a - b).norm());
            }
            records.push(serde_json::json!({"phase": phase_kind, "rate": sample_rate,
                "max_magnitude_difference_db": max_db, "max_complex_difference": max_complex}));
            if !(max_db.is_finite() && max_db <= 0.1 && max_complex <= 0.02) {
                failures.push(format!("{phase_kind} {sample_rate}: permutation changed transfer by {max_db} dB, {max_complex} complex"));
            }
        }
    }
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qa");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("hybrid-seat-permutation.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "scope": "analytic_training_seats_direct_finite_transfer_not_physical_holdouts_or_backend",
        "records": records,
    })).unwrap()).unwrap();
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

#[test]
fn fixed_budget_hybrid_stop_does_not_continue_into_fir() {
    let curve = curve(false);
    let prepared = prepared(curve.clone());
    let mut room_config = config();
    room_config.optimizer.num_filters = 1;
    room_config.optimizer.algorithm = "autoeq:de".into();
    room_config.optimizer.max_iter = 40;
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);
    let resources = EqResources::default();
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = calls.clone();
    let result = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::Hybrid, channel_name: "left", prepared: &prepared,
        room_config: &room_config, sample_rate: 48_000.0, target: &target,
        preprocessed: &features, optimizer: &room_config.optimizer, eq_resources: &resources,
        sidecar_reference: reference("stopped.wav"),
        callback: Some(Box::new(move |_, _, _| {
            observed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            autoeq_optim::de::CallbackAction::Stop
        })),
    });
    assert!(calls.load(std::sync::atomic::Ordering::Relaxed) > 0);
    let error = result.err().expect("stopped IIR search must not produce a completed Hybrid artifact");
    assert!(error.to_string().contains("stopped by progress callback"), "{error}");
}

fn nonlinear_spatial_weight_fixture(phase_kind: &str, correct_excess_phase: bool) {
    use roomeq_model::{MultiMeasurementConfig, MultiMeasurementStrategy};
    let frequencies = Array1::logspace(10.0, 20.0_f64.log10(), 500.0_f64.log10(), 96);
    // Stable first-order digital all-pass plus common arrival delay. The
    // all-pass creates dispersion that cannot be explained by delay alone.
    let allpass_pole = (-std::f64::consts::TAU * 150.0 / 48_000.0).exp();
    let phase = (phase_kind == "kirkeby").then(|| frequencies.mapv(|f| {
        let z = num_complex::Complex64::from_polar(1.0, -std::f64::consts::TAU * f / 48_000.0);
        ((z - allpass_pole) / (1.0 - allpass_pole * z)).arg().to_degrees() - 360.0 * f * 0.01
    }));
    let flat = Curve { freq: frequencies.clone(), spl: Array1::from_elem(96, 80.0), phase: phase.clone(), ..Default::default() };
    let peak = Curve { freq: frequencies.clone(),
        spl: frequencies.mapv(|f| 80.0 + 6.0 * (-((f - 120.0) / 30.0).powi(2)).exp()), phase: phase.clone(), ..Default::default() };
    let representative = Curve { freq: frequencies.clone(), spl: (&peak.spl + &flat.spl) / 2.0, phase, ..Default::default() };
    let prepared = PreparedChannelInput::new(PreparedChannelMeasurements::new(
        representative.clone(), vec![peak, flat], true), None, PreparedCea2034::default(), EqResources::default());
    let mut outcomes = Vec::new();
    for weights in [vec![1.0, 0.0], vec![0.0, 1.0]] {
        let mut room_config = config();
        room_config.optimizer.algorithm = "autoeq:cobyla".into();
        room_config.optimizer.max_iter = 40;
        room_config.optimizer.seed = Some(42);
        let fir_config = room_config.optimizer.fir.as_mut().unwrap();
        fir_config.phase = phase_kind.into();
        fir_config.correct_excess_phase = correct_excess_phase;
        fir_config.taps = 2048;
        room_config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::WeightedSum, weights: Some(weights.clone()), ..Default::default()
        });
        let target = build_target_context("left", &room_config, &representative, None);
        let features = preprocessed(&representative);
        let resources = EqResources::default();
        let result = process_fir_channel(FirChannelRequest {
            mode: FirChannelMode::Hybrid, channel_name: "left", prepared: &prepared,
            room_config: &room_config, sample_rate: 48_000.0, target: &target,
            preprocessed: &features, optimizer: &room_config.optimizer, eq_resources: &resources,
            sidecar_reference: reference("minimum_spatial.wav"), callback: None,
        }).unwrap();
        let taps = result.fir_coeffs.as_ref().unwrap();
        assert_eq!(taps.len(), 2048);
        let iir = response::compute_peq_complex_response(&result.filters, &Array1::from_vec(vec![120.0]), 48_000.0)[0];
        let fir: num_complex::Complex64 = taps.iter().enumerate().map(|(n, tap)|
            num_complex::Complex64::from_polar(*tap, -std::f64::consts::TAU * 120.0 * n as f64 / 48_000.0)).sum();
        let energy: f64 = taps.iter().map(|tap| tap * tap).sum();
        let centroid = taps.iter().enumerate().map(|(n, tap)| n as f64 * tap * tap).sum::<f64>() / energy;
        let mut pre_delays = Vec::new();
        let mut post_delays = Vec::new();
        let mut complete_delays = Vec::new();
        let mut iir_delays = Vec::new();
        for frequency in [40.0, 100.0, 200.0, 400.0] {
            let omega = std::f64::consts::TAU * frequency / 48_000.0;
            let allpass_delay = (1.0 - allpass_pole.powi(2)) /
                (1.0 + allpass_pole.powi(2) - 2.0 * allpass_pole * omega.cos());
            let h: num_complex::Complex64 = taps.iter().enumerate().map(|(n, tap)|
                num_complex::Complex64::from_polar(*tap, -omega * n as f64)).sum();
            let moment: num_complex::Complex64 = taps.iter().enumerate().map(|(n, tap)|
                num_complex::Complex64::from_polar(n as f64 * tap, -omega * n as f64)).sum();
            pre_delays.push(allpass_delay);
            post_delays.push(allpass_delay + (moment / h).re);
            // Differentiate the actual IIR transfer, not only the FIR. Check
            // two frequency steps so the numerical oracle is not step-sensitive.
            let iir_delay = |step: f64| {
                let grid = Array1::from_vec(vec![frequency - step, frequency + step]);
                let response = response::compute_peq_complex_response(&result.filters, &grid, 48_000.0);
                -(response[1] / response[0]).arg() * 48_000.0
                    / (std::f64::consts::TAU * 2.0 * step)
            };
            let delay = iir_delay(0.01);
            assert!((delay - iir_delay(0.001)).abs() < 1e-3,
                "IIR delay oracle is step-sensitive at {frequency} Hz");
            assert!(delay.is_finite() && (moment / h).re.is_finite());
            iir_delays.push(delay);
            let acoustic_delay = if phase_kind == "kirkeby" { 480.0 + allpass_delay } else { 0.0 };
            complete_delays.push(acoustic_delay + delay + (moment / h).re);
        }
        let spread = |values: &[f64]| values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - values.iter().copied().fold(f64::INFINITY, f64::min);
        if phase_kind == "minimum" {
            assert!(centroid < 100.0, "minimum-phase design acquired linear-phase latency: {centroid}");
        }
        let expected_kind = if phase_kind == "minimum" { "minimum-phase" } else { "kirkeby reference-phase" };
        assert!(result.optimizer_evidence.iter().any(|evidence| evidence.status.contains(&format!("{expected_kind} realized dB-basis"))));
        outcomes.push(serde_json::json!({"weights": weights, "correction_db_at_120hz": 20.0 * (iir * fir).norm().log10(),
            "fir_taps": taps.len(), "energy_centroid_samples": centroid, "optimizer_evidence": result.optimizer_evidence,
            "analytic_allpass_delay_spread_samples": spread(&pre_delays),
            "allpass_plus_fir_delay_spread_samples": spread(&post_delays),
            "complete_acoustic_iir_fir_delay_spread_samples": spread(&complete_delays),
            "complete_acoustic_iir_fir_delay_samples": complete_delays,
            "iir_delay_samples": iir_delays,
            "delay_probe_frequencies_hz": [40.0, 100.0, 200.0, 400.0],
            "common_arrival_samples": if phase_kind == "kirkeby" { 480.0 } else { 0.0 }}));
    }
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/qa");
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join(format!("hybrid-{phase_kind}-spatial-phase-{correct_excess_phase}.json")), serde_json::to_vec_pretty(&serde_json::json!({
        "scope": "analytic_training_seats_in_memory_not_backend_or_minimum_phase_zero_certificate",
        "sample_rate_hz": 48000, "phase_kind": phase_kind, "correct_excess_phase": correct_excess_phase, "outcomes": outcomes,
    })).unwrap()).unwrap();
    let peak = outcomes[0]["correction_db_at_120hz"].as_f64().unwrap();
    let flat = outcomes[1]["correction_db_at_120hz"].as_f64().unwrap();
    assert!(flat - peak > 2.0, "{phase_kind} Hybrid erased weight choice: {peak} vs {flat}");
    if phase_kind == "kirkeby" && correct_excess_phase {
        let flat_case = &outcomes[1];
        assert!(flat_case["complete_acoustic_iir_fir_delay_spread_samples"].as_f64().unwrap()
            < 0.5 * flat_case["analytic_allpass_delay_spread_samples"].as_f64().unwrap(),
            "Kirkeby did not correct reference all-pass dispersion: {flat_case}");
    }
}

#[test]
fn fir_post_score_does_not_count_intended_target_tilt() {
    let mut tilted = curve(false);
    let tilt = tilted.freq.mapv(|frequency| -(frequency / 1_000.0).log2());
    tilted.spl = &tilt + 80.0;
    let prepared = prepared(tilted.clone());
    let room_config = config();
    let mut features = preprocessed(&tilted);
    features.curve_for_optim = tilted.clone();
    let target = crate::channel_target::TargetContext {
        target_tilt_curve: Some(Curve {
            freq: tilted.freq.clone(),
            spl: tilt,
            ..Default::default()
        }),
        min_freq: 20.0,
        max_freq: 20_000.0,
        pre_score: 0.0,
        mean_spl: 80.0,
        cea2034_active: false,
    };
    let resources = EqResources::default();
    let request = FirChannelRequest {
        mode: FirChannelMode::PhaseLinear,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_fir_48000hz.wav"),
        callback: None,
    };

    let result = super::assemble::assemble_fir_result(
        &request,
        FirOptimizerOutput::PhaseLinear {
            coefficients: vec![1.0],
            sidecar_reference: request.sidecar_reference.clone(),
        },
        Vec::new(),
        None,
    )
    .unwrap();
    assert!(result.post_score < 1e-4, "post_score={}", result.post_score);
}

#[test]
fn mixed_phase_without_phase_data_returns_iir_only() {
    let curve = curve(false);
    let prepared = prepared(curve.clone());
    let room_config = config();
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);
    let result = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::MixedPhase,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_excess_phase_fir_48000hz.wav"),
        callback: None,
    })
    .unwrap();

    assert!(result.fir_coeffs.is_none());
    assert!(result.convolution_sidecar.is_none());
    assert!(
        result
            .channel
            .plugins
            .iter()
            .all(|plugin| plugin.plugin_type != "convolution")
    );
}

#[test]
fn mixed_phase_with_phase_data_returns_optional_sidecar() {
    let curve = curve(true);
    let prepared = prepared(curve.clone());
    let mut room_config = config();
    room_config.optimizer.mixed_phase = Some(MixedPhaseSerdeConfig {
        max_fir_length_ms: 5.0,
        pre_ringing_threshold_db: -30.0,
        min_spatial_depth: 0.5,
        phase_smoothing_octaves: 1.0 / 6.0,
    });
    let resources = EqResources::default();
    let target = build_target_context("left", &room_config, &curve, None);
    let features = preprocessed(&curve);
    let result = process_fir_channel(FirChannelRequest {
        mode: FirChannelMode::MixedPhase,
        channel_name: "left",
        prepared: &prepared,
        room_config: &room_config,
        sample_rate: 48_000.0,
        target: &target,
        preprocessed: &features,
        optimizer: &room_config.optimizer,
        eq_resources: &resources,
        sidecar_reference: reference("left_excess_phase_fir_48000hz.wav"),
        callback: None,
    })
    .unwrap();

    assert!(result.fir_coeffs.is_some());
    let sidecar = result.convolution_sidecar.unwrap();
    assert!(!sidecar.required);
    assert_eq!(
        sidecar.reference.filename(),
        "left_excess_phase_fir_48000hz.wav"
    );
    assert!(
        result
            .channel
            .plugins
            .iter()
            .any(|plugin| plugin.plugin_type == "convolution")
    );
}

#[test]
fn mixed_phase_adapts_depth_to_preserve_phase_only_magnitude() {
    let curve = curve(true);
    let residual_phase_deg = Array1::from_iter(
        (0..curve.freq.len()).map(|index| 170.0 * ((index as f64) * 1.618_033_988_75).sin()),
    );
    let config = crate::mixed_phase::MixedPhaseConfig {
        max_fir_length_ms: 10.0,
        pre_ringing_threshold_db: -30.0,
        min_spatial_depth: 0.5,
        phase_smoothing_octaves: 1.0 / 6.0,
    };

    let full_coefficients = crate::mixed_phase::generate_excess_phase_fir_with_depth(
        &curve.freq,
        &residual_phase_deg,
        &config,
        48_000.0,
        None,
    );
    let full_deviation = max_phase_only_magnitude_deviation_db(
        &full_coefficients,
        &curve.freq,
        &curve.spl,
        48_000.0,
    );
    assert!(
        full_deviation > MAX_PHASE_ONLY_MAGNITUDE_DEVIATION_DB,
        "synthetic full-depth correction must exercise adaptation, deviation={full_deviation}"
    );

    let (_coefficients, applied_depth, deviation) = generate_magnitude_safe_excess_phase_fir(
        &curve.freq,
        &curve.spl,
        &residual_phase_deg,
        &config,
        48_000.0,
        None,
    )
    .expect("a partial phase correction should satisfy the magnitude contract");

    assert!(applied_depth > 0.0 && applied_depth < 1.0);
    assert!(deviation <= MAX_PHASE_ONLY_MAGNITUDE_DEVIATION_DB);
}
