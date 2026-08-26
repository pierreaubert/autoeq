use crate::measurement::load_source_with_frequency_samples;
use crate::{dba, multisub as multisub_resources};
use autoeq_measurements::read::interpolate_log_space;
use log::info;
use roomeq_engine::Curve;
use roomeq_engine::bass_management::{SubDriverInfo, SubPreprocessResult};
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_engine::topology::{is_valid_frequency_grid, same_frequency_grid};
use roomeq_model::{
    CardioidConfig, DBAConfig, MultiSubGroup, OptimizerConfig, SpeakerConfig, SubwooferStrategy,
};

/// Preprocess the LFE channel's SpeakerConfig into a combined curve and per-driver info.
///
/// Dispatches by SpeakerConfig variant:
/// - Single: load curve, no drivers
/// - MultiSub + Mso: run MSO optimization, return combined + per-sub gains/delays
/// - MultiSub + Single: power-sum all subs, return combined + per-sub info (zero gains/delays)
/// - MultiSub + Dba: error (should use SpeakerConfig::Dba)
/// - Cardioid: simulate combined response from front + delayed/inverted rear
/// - Dba: run DBA optimization, return combined + front/rear info
/// - Group: error (handled by generic path)
pub(in super::super) fn preprocess_sub_with_frequency_samples(
    lfe_config: &SpeakerConfig,
    strategy: &SubwooferStrategy,
    optimizer: &OptimizerConfig,
    sample_rate: f64,
    frequency_samples: usize,
) -> Result<SubPreprocessResult> {
    match lfe_config {
        SpeakerConfig::Single(source) => {
            let curve =
                load_source_with_frequency_samples(source, frequency_samples).map_err(|e| {
                    AutoeqError::InvalidMeasurement {
                        message: e.to_string(),
                    }
                })?;
            Ok(SubPreprocessResult {
                combined_curve: curve,
                drivers: None,
            })
        }
        SpeakerConfig::MultiSub(ms) => match strategy {
            SubwooferStrategy::Mso => preprocess_multisub_mso_with_frequency_samples(
                ms,
                optimizer,
                sample_rate,
                frequency_samples,
            ),
            SubwooferStrategy::Single => {
                preprocess_multisub_independent_with_frequency_samples(ms, frequency_samples)
            }
            SubwooferStrategy::Dba => Err(AutoeqError::InvalidConfiguration {
                message: "SubwooferStrategy::Dba requires SpeakerConfig::Dba, not MultiSub"
                    .to_string(),
            }),
        },
        SpeakerConfig::Cardioid(c) => {
            preprocess_cardioid_with_frequency_samples(c, frequency_samples)
        }
        SpeakerConfig::Dba(d) => {
            preprocess_dba_with_frequency_samples(d, optimizer, sample_rate, frequency_samples)
        }
        SpeakerConfig::Group(_) | SpeakerConfig::Topology(_) => {
            Err(AutoeqError::InvalidConfiguration {
                message:
                    "Group speaker config should not reach stereo sub workflow; use generic path"
                        .to_string(),
            })
        }
        SpeakerConfig::SupportingSource(_) => Err(AutoeqError::InvalidConfiguration {
            message: "Supporting source config cannot be used as an LFE/subwoofer channel"
                .to_string(),
        }),
    }
}

/// MSO: optimize inter-sub gains/delays, return combined curve + per-sub info
pub(in super::super) fn preprocess_multisub_mso_with_frequency_samples(
    ms: &MultiSubGroup,
    optimizer: &OptimizerConfig,
    sample_rate: f64,
    frequency_samples: usize,
) -> Result<SubPreprocessResult> {
    info!("  MSO optimization for {} subwoofers", ms.subwoofers.len());

    let optimized = multisub_resources::optimize_multisub_with_frequency_samples(
        &ms.subwoofers,
        optimizer,
        sample_rate,
        frequency_samples,
    )
    .map_err(|e| AutoeqError::OptimizationFailed {
        message: format!("MSO optimization failed: {}", e),
    })?;
    // Keep the spatial magnitude used for multi-seat EQ, but carry the
    // primary-seat complex phase/coherence into crossover timing. The legacy
    // curve alone is intentionally magnitude-only.
    let mut combined = optimized.combined_response.legacy_combined_curve();
    if let Some(primary) = optimized.combined_response.primary_seat_complex.as_ref() {
        combined.phase = primary.phase.clone();
        combined.coherence = primary.coherence.clone();
    }
    let result = optimized.base;

    info!(
        "  MSO result: gains={:?}, delays={:?}",
        result.gains, result.delays
    );

    // Load individual curves for driver info
    let mut drivers = Vec::new();
    for (i, source) in ms.subwoofers.iter().enumerate() {
        let curve = load_source_with_frequency_samples(source, frequency_samples).map_err(|e| {
            AutoeqError::InvalidMeasurement {
                message: e.to_string(),
            }
        })?;
        drivers.push(SubDriverInfo {
            name: format!("{}_{}", ms.name, i + 1),
            gain: result.gains.get(i).copied().unwrap_or(0.0),
            delay: result.delays.get(i).copied().unwrap_or(0.0),
            inverted: false,
            initial_curve: Some(curve),
        });
    }

    Ok(SubPreprocessResult {
        combined_curve: combined,
        drivers: Some(drivers),
    })
}

/// Independent subs: power-sum all sub curves, return combined + per-sub info (zero gains/delays)
pub(in super::super) fn preprocess_multisub_independent_with_frequency_samples(
    ms: &MultiSubGroup,
    frequency_samples: usize,
) -> Result<SubPreprocessResult> {
    info!(
        "  Independent sub averaging for {} subwoofers",
        ms.subwoofers.len()
    );

    let mut curves = Vec::new();
    for source in &ms.subwoofers {
        let curve = load_source_with_frequency_samples(source, frequency_samples).map_err(|e| {
            AutoeqError::InvalidMeasurement {
                message: e.to_string(),
            }
        })?;
        curves.push(curve);
    }

    // Power summation on the first sub's frequency grid:
    // Convert dB to linear power, sum, convert back to dB.
    // This correctly represents incoherent summation of multiple subs.
    let ref_freq = curves[0].freq.clone();
    let mut sum_power = ndarray::Array1::<f64>::zeros(ref_freq.len());
    for curve in &curves {
        let interp = interpolate_log_space(&ref_freq, curve);
        sum_power += &interp.spl.mapv(|db| 10.0_f64.powf(db / 10.0));
    }
    let avg_spl = sum_power.mapv(|p| 10.0 * p.log10());

    let combined = Curve {
        freq: ref_freq,
        spl: avg_spl,
        phase: None,
        ..Default::default()
    };

    let drivers: Vec<SubDriverInfo> = curves
        .into_iter()
        .enumerate()
        .map(|(i, curve)| SubDriverInfo {
            name: format!("{}_{}", ms.name, i + 1),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            initial_curve: Some(curve),
        })
        .collect();

    Ok(SubPreprocessResult {
        combined_curve: combined,
        drivers: Some(drivers),
    })
}

/// Cardioid: simulate combined response from front + delayed/inverted rear sub
pub(in super::super) fn preprocess_cardioid_with_frequency_samples(
    c: &CardioidConfig,
    frequency_samples: usize,
) -> Result<SubPreprocessResult> {
    let front_curve =
        load_source_with_frequency_samples(&c.front, frequency_samples).map_err(|e| {
            AutoeqError::InvalidMeasurement {
                message: format!("Cardioid front: {}", e),
            }
        })?;
    let rear_curve =
        load_source_with_frequency_samples(&c.rear, frequency_samples).map_err(|e| {
            AutoeqError::InvalidMeasurement {
                message: format!("Cardioid rear: {}", e),
            }
        })?;

    if !is_valid_frequency_grid(&front_curve.freq) || !is_valid_frequency_grid(&rear_curve.freq) {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Cardioid preprocessing requires valid frequency grids".to_string(),
        });
    }
    if front_curve.spl.len() != front_curve.freq.len()
        || rear_curve.spl.len() != rear_curve.freq.len()
        || front_curve
            .phase
            .as_ref()
            .is_some_and(|phase| phase.len() != front_curve.freq.len())
        || rear_curve
            .phase
            .as_ref()
            .is_some_and(|phase| phase.len() != rear_curve.freq.len())
    {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Cardioid preprocessing curve arrays must match frequency-grid length"
                .to_string(),
        });
    }
    if front_curve.phase.is_none() || rear_curve.phase.is_none() {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Cardioid preprocessing requires measured phase front rear drivers"
                .to_string(),
        });
    }
    if rear_curve.freq.first() > front_curve.freq.first()
        || rear_curve.freq.last() < front_curve.freq.last()
    {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Cardioid rear measurement must cover the full front frequency span"
                .to_string(),
        });
    }
    let rear_curve = if same_frequency_grid(&front_curve.freq, &rear_curve.freq) {
        rear_curve
    } else {
        interpolate_log_space(&front_curve.freq, &rear_curve)
    };
    let delay_ms = c.separation_meters / 343.0 * 1000.0;
    info!(
        "  Cardioid: separation={:.2}m, delay={:.2}ms",
        c.separation_meters, delay_ms
    );

    // Simulate combined response (complex sum of front + delayed/inverted rear)
    use num_complex::Complex;
    let n_points = front_curve.freq.len();
    let mut combined_spl = ndarray::Array1::zeros(n_points);
    let mut combined_phase = Vec::with_capacity(n_points);

    let front_phase = front_curve.phase.as_ref().expect("validated above");
    let rear_phase = rear_curve.phase.as_ref().expect("validated above");

    for i in 0..n_points {
        let f = front_curve.freq[i];
        let omega = 2.0 * std::f64::consts::PI * f;

        // Front
        let f_mag = 10.0_f64.powf(front_curve.spl[i] / 20.0);
        let f_phi = front_phase[i].to_radians();
        let f_c = Complex::from_polar(f_mag, f_phi);

        // Rear (Inverted + Delayed)
        let r_mag = 10.0_f64.powf(rear_curve.spl[i] / 20.0);
        let r_phi_meas = rear_phase[i].to_radians();
        let delay_s = delay_ms / 1000.0;
        let delay_phi = -omega * delay_s;
        let invert_phi = std::f64::consts::PI;
        let r_phi_total = r_phi_meas + delay_phi + invert_phi;
        let r_c = Complex::from_polar(r_mag, r_phi_total);

        let sum = f_c + r_c;
        combined_spl[i] = 20.0 * sum.norm().max(1e-12).log10();
        combined_phase.push(sum.arg().to_degrees());
    }

    let combined = Curve {
        freq: front_curve.freq.clone(),
        spl: combined_spl,
        phase: Some(ndarray::Array1::from_iter(combined_phase)),
        ..Default::default()
    };

    let drivers = vec![
        SubDriverInfo {
            name: "Front Sub".to_string(),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            initial_curve: Some(front_curve),
        },
        SubDriverInfo {
            name: "Rear Sub".to_string(),
            gain: 0.0,
            delay: delay_ms,
            inverted: true,
            initial_curve: Some(rear_curve),
        },
    ];

    Ok(SubPreprocessResult {
        combined_curve: combined,
        drivers: Some(drivers),
    })
}

/// DBA: run DBA optimization, return combined curve + front/rear driver info
pub(in super::super) fn preprocess_dba_with_frequency_samples(
    d: &DBAConfig,
    optimizer: &OptimizerConfig,
    sample_rate: f64,
    frequency_samples: usize,
) -> Result<SubPreprocessResult> {
    info!("  DBA optimization");

    let optimized =
        dba::optimize_dba_with_frequency_samples(d, optimizer, sample_rate, frequency_samples)
            .map_err(|e| AutoeqError::OptimizationFailed {
                message: format!("DBA optimization failed: {}", e),
            })?;
    let result = optimized.driver;
    let combined = optimized.combined_curve;

    info!(
        "  DBA result: gains={:?}, delays={:?}",
        result.gains, result.delays
    );

    // Load front and rear array responses for display
    let front_curve = dba::sum_array_response_with_frequency_samples(&d.front, frequency_samples)
        .map_err(|e| AutoeqError::InvalidMeasurement {
        message: format!("DBA front array: {}", e),
    })?;
    let rear_curve = dba::sum_array_response_with_frequency_samples(&d.rear, frequency_samples)
        .map_err(|e| AutoeqError::InvalidMeasurement {
            message: format!("DBA rear array: {}", e),
        })?;

    let drivers = vec![
        SubDriverInfo {
            name: "Front Array".to_string(),
            gain: result.gains.first().copied().unwrap_or(0.0),
            delay: result.delays.first().copied().unwrap_or(0.0),
            inverted: false,
            initial_curve: Some(front_curve),
        },
        SubDriverInfo {
            name: "Rear Array".to_string(),
            gain: result.gains.get(1).copied().unwrap_or(0.0),
            delay: result.delays.get(1).copied().unwrap_or(0.0),
            inverted: true,
            initial_curve: Some(rear_curve),
        },
    ];

    Ok(SubPreprocessResult {
        combined_curve: combined,
        drivers: Some(drivers),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_engine::Curve;
    use roomeq_model::{
        CardioidConfig, DBAConfig, MeasurementSource, MultiSubGroup, OptimizerConfig,
        SpeakerConfig, SpeakerGroup, SubwooferStrategy,
    };

    fn make_curve(freq_count: usize, spl_db: f64, phase_deg: Option<f64>) -> Curve {
        let freq = ndarray::Array1::logspace(10.0, f64::log10(20.0), f64::log10(200.0), freq_count);
        let spl = ndarray::Array1::from_elem(freq_count, spl_db);
        let phase = phase_deg.map(|p| ndarray::Array1::from_elem(freq_count, p));
        Curve {
            freq,
            spl,
            phase,
            ..Default::default()
        }
    }

    fn tiny_optimizer() -> OptimizerConfig {
        OptimizerConfig {
            algorithm: "autoeq:cobyla".to_string(),
            max_iter: 20,
            population: 6,
            min_freq: 20.0,
            max_freq: 200.0,
            seed: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn preprocess_sub_single_returns_finite_combined_curve() {
        let curve = make_curve(16, 80.0, None);
        let config = SpeakerConfig::Single(MeasurementSource::InMemory(curve));
        let result = preprocess_sub_with_frequency_samples(
            &config,
            &SubwooferStrategy::Single,
            &tiny_optimizer(),
            48000.0,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        );
        assert!(result.is_ok(), "expected Ok, got Err: {:?}", result.err());
        let result = result.unwrap();
        assert!(result.drivers.is_none());
        assert!(result.combined_curve.spl.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn preprocess_sub_multisub_mso_returns_drivers() {
        let subs = MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemory(make_curve(16, 80.0, Some(0.0))),
                MeasurementSource::InMemory(make_curve(16, 80.0, Some(0.0))),
            ],
            allpass_optimization: false,
        };
        let config = SpeakerConfig::MultiSub(subs);
        let result = preprocess_sub_with_frequency_samples(
            &config,
            &SubwooferStrategy::Mso,
            &tiny_optimizer(),
            48000.0,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        );
        assert!(result.is_ok(), "expected Ok, got Err: {:?}", result.err());
        let result = result.unwrap();
        assert!(result.drivers.is_some());
        let drivers = result.drivers.unwrap();
        assert!(!drivers.is_empty());
        assert!(result.combined_curve.spl.iter().all(|v| v.is_finite()));
        assert!(result.combined_curve.phase.is_some());
    }

    #[test]
    fn preprocess_sub_multisub_single_averages_subs() {
        let subs = MultiSubGroup {
            name: "subs".to_string(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemory(make_curve(16, 80.0, None)),
                MeasurementSource::InMemory(make_curve(16, 80.0, None)),
            ],
            allpass_optimization: false,
        };
        let config = SpeakerConfig::MultiSub(subs);
        let result = preprocess_sub_with_frequency_samples(
            &config,
            &SubwooferStrategy::Single,
            &tiny_optimizer(),
            48000.0,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        );
        assert!(result.is_ok(), "expected Ok, got Err: {:?}", result.err());
        let result = result.unwrap();
        assert!(result.drivers.is_some());
        let drivers = result.drivers.unwrap();
        assert_eq!(drivers.len(), 2);
        assert!(drivers.iter().all(|d| d.gain == 0.0 && d.delay == 0.0));
        assert!(result.combined_curve.spl.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn preprocess_sub_group_returns_error() {
        let group = SpeakerGroup {
            name: "group".to_string(),
            speaker_name: None,
            measurements: vec![MeasurementSource::InMemory(make_curve(16, 80.0, None))],
            crossover: None,
        };
        let config = SpeakerConfig::Group(group);
        let result = preprocess_sub_with_frequency_samples(
            &config,
            &SubwooferStrategy::Single,
            &tiny_optimizer(),
            48000.0,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        );
        assert!(result.is_err(), "expected Err for Group config");
    }

    #[test]
    fn preprocess_cardioid_happy_path_with_phase() {
        let front = make_curve(16, 80.0, Some(0.0));
        let rear = make_curve(16, 80.0, Some(0.0));
        let config = CardioidConfig {
            name: "cardioid".to_string(),
            speaker_name: None,
            front: MeasurementSource::InMemory(front),
            rear: MeasurementSource::InMemory(rear),
            separation_meters: 1.0,
        };
        let result =
            preprocess_cardioid_with_frequency_samples(&config, crate::DEFAULT_FREQUENCY_SAMPLES);
        assert!(result.is_ok(), "expected Ok, got Err: {:?}", result.err());
        let result = result.unwrap();
        assert!(result.drivers.is_some());
        assert_eq!(result.drivers.as_ref().unwrap().len(), 2);
        assert!(result.combined_curve.spl.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn preprocess_cardioid_interpolates_mismatched_frequency_grids() {
        let mut front = make_curve(16, 80.0, Some(0.0));
        let rear = make_curve(16, 80.0, Some(0.0));
        front.freq = ndarray::Array1::logspace(10.0, f64::log10(25.0), f64::log10(195.0), 16);
        let config = CardioidConfig {
            name: "cardioid".to_string(),
            speaker_name: None,
            front: MeasurementSource::InMemory(front),
            rear: MeasurementSource::InMemory(rear),
            separation_meters: 1.0,
        };
        let result =
            preprocess_cardioid_with_frequency_samples(&config, crate::DEFAULT_FREQUENCY_SAMPLES);
        assert!(
            result.is_ok(),
            "expected interpolation to accept mismatched grids"
        );
        assert!(
            result
                .unwrap()
                .combined_curve
                .spl
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn preprocess_cardioid_rejects_rear_span_extrapolation() {
        let front = make_curve(16, 80.0, Some(0.0));
        let mut rear = make_curve(16, 80.0, Some(0.0));
        rear.freq = ndarray::Array1::logspace(10.0, f64::log10(25.0), f64::log10(195.0), 16);
        let config = CardioidConfig {
            name: "cardioid".to_string(),
            speaker_name: None,
            front: MeasurementSource::InMemory(front),
            rear: MeasurementSource::InMemory(rear),
            separation_meters: 1.0,
        };

        let error =
            preprocess_cardioid_with_frequency_samples(&config, crate::DEFAULT_FREQUENCY_SAMPLES)
                .err()
                .expect("rear span must be rejected");
        assert!(error.to_string().contains("full front frequency span"));
    }

    #[test]
    fn preprocess_cardioid_errors_on_mismatched_spl_lengths() {
        let mut front = make_curve(16, 80.0, Some(0.0));
        front.spl = ndarray::Array1::from_elem(8, 80.0);
        let rear = make_curve(16, 80.0, Some(0.0));
        let config = CardioidConfig {
            name: "cardioid".to_string(),
            speaker_name: None,
            front: MeasurementSource::InMemory(front),
            rear: MeasurementSource::InMemory(rear),
            separation_meters: 1.0,
        };
        let result =
            preprocess_cardioid_with_frequency_samples(&config, crate::DEFAULT_FREQUENCY_SAMPLES);
        assert!(result.is_err(), "expected Err for mismatched SPL lengths");
    }

    #[test]
    fn preprocess_cardioid_errors_on_mismatched_phase_lengths() {
        let mut front = make_curve(16, 80.0, Some(0.0));
        front.phase = Some(ndarray::Array1::from_elem(8, 0.0));
        let rear = make_curve(16, 80.0, Some(0.0));
        let config = CardioidConfig {
            name: "cardioid".to_string(),
            speaker_name: None,
            front: MeasurementSource::InMemory(front),
            rear: MeasurementSource::InMemory(rear),
            separation_meters: 1.0,
        };
        let result =
            preprocess_cardioid_with_frequency_samples(&config, crate::DEFAULT_FREQUENCY_SAMPLES);
        assert!(result.is_err(), "expected Err for mismatched phase lengths");
    }

    #[test]
    fn preprocess_dba_happy_path() {
        let front_curve = make_curve(16, 80.0, Some(0.0));
        let rear_curve = make_curve(16, 80.0, Some(0.0));
        let config = DBAConfig {
            name: "dba".to_string(),
            speaker_name: None,
            front: vec![MeasurementSource::InMemory(front_curve)],
            rear: vec![MeasurementSource::InMemory(rear_curve)],
        };
        let result = preprocess_dba_with_frequency_samples(
            &config,
            &tiny_optimizer(),
            48000.0,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        );
        assert!(result.is_ok(), "expected Ok, got Err: {:?}", result.err());
        let result = result.unwrap();
        assert!(result.drivers.is_some());
        assert_eq!(result.drivers.as_ref().unwrap().len(), 2);
        assert!(result.combined_curve.spl.iter().all(|v| v.is_finite()));
    }
}
