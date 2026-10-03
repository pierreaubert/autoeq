use crate::measurement::load_source_with_frequency_samples;
use crate::{dba, multisub as multisub_resources};
use autoeq_measurements::read::interpolate_log_space;
use log::info;
use roomeq_engine::Curve;
use roomeq_engine::bass_management::{SubDriverInfo, SubPreprocessResult};
use roomeq_engine::error::{AutoeqError, Result};
use roomeq_engine::topology::is_valid_frequency_grid;
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
            let seats = crate::measurement::load_source_individual_with_frequency_samples(
                source,
                frequency_samples,
            )
            .map_err(|e| AutoeqError::InvalidMeasurement {
                message: e.to_string(),
            })?;
            let primary_seat = if seats.len() == 1 {
                0
            } else {
                optimizer
                    .multi_seat
                    .as_ref()
                    .map_or(0, |config| config.primary_seat)
            };
            let curve = seats.get(primary_seat).cloned().ok_or_else(|| {
                AutoeqError::InvalidMeasurement {
                    message: format!("primary seat {primary_seat} unavailable for single subwoofer with {} measurement(s)", seats.len()),
                }
            })?;
            Ok(SubPreprocessResult {
                // Spatial magnitude EQ needs every seat; crossover timing needs
                // the same measured primary seat as the mains, never their RMS.
                shared_eq_seats: (seats.len() > 1).then_some(seats),
                common_eq_complete: false,
                joint_sub: None,
                optimizer_evidence: Vec::new(),
                advisories: Vec::new(),
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
            SubwooferStrategy::Single if ms.joint_optimization => Err(AutoeqError::InvalidConfiguration {
                message: "joint_optimization requires routed subwoofer strategy 'mso'; 'single' selects independent subs".into(),
            }),
            SubwooferStrategy::Single => {
                preprocess_multisub_independent_with_frequency_samples(ms, frequency_samples)
            }
            SubwooferStrategy::Dba => Err(AutoeqError::InvalidConfiguration {
                message: "SubwooferStrategy::Dba requires SpeakerConfig::Dba, not MultiSub"
                    .to_string(),
            }),
        },
        SpeakerConfig::Cardioid(c) => preprocess_cardioid_with_frequency_samples(
            c,
            frequency_samples,
            optimizer
                .multi_seat
                .as_ref()
                .map_or(0, |seat| seat.primary_seat),
        ),
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
    if ms.joint_optimization
        || ms.allpass_optimization
        || optimizer
            .multi_seat
            .as_ref()
            .is_some_and(|seat| seat.enabled)
    {
        return preprocess_multisub_advanced(ms, optimizer, sample_rate, frequency_samples);
    }
    let measured = ms
        .subwoofers
        .iter()
        .map(|source| {
            load_source_with_frequency_samples(source, frequency_samples).map_err(|error| {
                AutoeqError::InvalidMeasurement {
                    message: error.to_string(),
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let bounded = roomeq_engine::group_processing::sub_optimizer_config(&measured, optimizer);
    let optimizer = &bounded;
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
    // Routing uses the complete primary-seat complex response when available.
    // Keep spatial EQ seats separately below; never attach primary-seat phase
    // to an averaged magnitude. The compatibility helper preserves this split.
    let combined = optimized.combined_response.legacy_combined_curve();
    let phase_controls_enabled = optimized.phase_controls_enabled;
    let mut advisories = optimized.advisories;
    let result = optimized.base;
    let seat_measurements =
        crate::group_measurements::load_multisub_seat_measurements_with_frequency_samples(
            ms,
            frequency_samples,
        )?;
    let shared_eq_seats = if phase_controls_enabled {
        seat_measurements
            .map(|seats| {
                roomeq_engine::multisub::render_mso_seat_responses(
                    &seats,
                    &result.gains,
                    &result.delays,
                )
            })
            .transpose()?
    } else {
        if seat_measurements.is_some() {
            let reason = if advisories
                .iter()
                .any(|advisory| advisory == "unverified_timing_gain_only")
            {
                "unverified_timing_shared_eq_seats_unavailable"
            } else {
                "missing_phase_shared_eq_seats_unavailable"
            };
            advisories.push(reason.into());
        }
        None
    };

    info!(
        "  MSO result: gains={:?}, delays={:?}",
        result.gains, result.delays
    );

    // Route alignment must use the same synchronous seat as the MSO solve.
    // `load_source` intentionally drops phase when averaging multiple seats;
    // that spatial magnitude is not a physical driver's complex response.
    let primary_seat = optimizer
        .multi_seat
        .as_ref()
        .map(|seat| seat.primary_seat)
        .unwrap_or(0);
    let primary_measurements =
        multisub_resources::load_primary_measurements_with_frequency_samples(
            &ms.subwoofers,
            primary_seat,
            frequency_samples,
        )
        .map_err(|error| AutoeqError::InvalidMeasurement {
            message: error.to_string(),
        })?;
    let mut drivers = Vec::new();
    for (i, curve) in primary_measurements.into_iter().enumerate() {
        drivers.push(SubDriverInfo {
            name: format!("{}_{}", ms.name, i + 1),
            gain: result.gains.get(i).copied().unwrap_or(0.0),
            delay: result.delays.get(i).copied().unwrap_or(0.0),
            inverted: false,
            processing: None,
            initial_curve: Some(curve),
        });
    }

    Ok(SubPreprocessResult {
        shared_eq_seats,
        common_eq_complete: false,
        joint_sub: None,
        optimizer_evidence: Vec::new(),
        advisories,
        combined_curve: combined,
        drivers: Some(drivers),
    })
}

/// Use the same multi-seat/all-pass engine as generic groups and retain its
/// per-driver transfer functions. Shared PEQ commutes with the array sum and
/// is included once on each driver before later routed correction.
fn preprocess_multisub_advanced(
    ms: &MultiSubGroup,
    optimizer: &OptimizerConfig,
    sample_rate: f64,
    frequency_samples: usize,
) -> Result<SubPreprocessResult> {
    use roomeq_engine::bass_management::SubDriverProcessing;
    let seats = crate::group_measurements::load_multisub_seat_measurements_with_frequency_samples(
        ms,
        frequency_samples,
    )?;
    let primary = optimizer
        .multi_seat
        .as_ref()
        .map(|seat| seat.primary_seat)
        .unwrap_or(0);
    let measurements = ms
        .subwoofers
        .iter()
        .enumerate()
        .map(|(index, source)| {
            if let Some(seats) = &seats {
                return seats[index].get(primary).cloned().ok_or_else(|| {
                    AutoeqError::InvalidMeasurement {
                        message: format!("Primary seat {primary} unavailable for sub {index}"),
                    }
                });
            }
            load_source_with_frequency_samples(source, frequency_samples).map_err(|error| {
                AutoeqError::InvalidMeasurement {
                    message: error.to_string(),
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let room = roomeq_model::RoomConfig {
        optimizer: roomeq_engine::group_processing::sub_optimizer_config(&measurements, optimizer),
        ..Default::default()
    };
    let resources = crate::prepare_eq_resources(&room.optimizer, None).map_err(|error| {
        AutoeqError::InvalidMeasurement {
            message: error.to_string(),
        }
    })?;
    let mut prepared = roomeq_engine::group_processing::PreparedMultiSubGroup {
        subwoofers: measurements.clone(),
        seat_measurements: seats,
        reference_scope: None,
    };
    let band_curves = prepared
        .seat_measurements
        .as_ref()
        .map(|seats| seats.iter().flatten().cloned().collect::<Vec<_>>())
        .unwrap_or_else(|| prepared.subwoofers.clone());
    let bounded =
        roomeq_engine::group_processing::sub_optimizer_config(&band_curves, &room.optimizer);
    prepared.reference_scope = crate::group_measurements::multisub_reference_scope(
        ms,
        [bounded.min_freq, bounded.max_freq],
    );
    let (chain, _, _, _, combined, _, _, _, _, evidence) =
        roomeq_engine::group_processing::process_multisub_group(
            &ms.name,
            ms,
            &room,
            sample_rate,
            &prepared,
            &resources,
            &resources,
        )?;
    let chains = chain
        .drivers
        .as_ref()
        .ok_or_else(|| AutoeqError::InvalidConfiguration {
            message: "Multi-sub engine omitted driver chains".into(),
        })?;
    let mut drivers = Vec::with_capacity(chains.len());
    for (driver_chain, measurement) in chains.iter().zip(measurements) {
        let mut gain = 0.0;
        let mut delay = 0.0;
        let mut inverted = false;
        let mut plugins = Vec::new();
        for plugin in &driver_chain.plugins {
            match plugin.plugin_type.as_str() {
                "gain" => {
                    gain += plugin.parameters["gain_db"].as_f64().unwrap_or(0.0);
                    inverted ^= plugin.parameters["invert"].as_bool().unwrap_or(false);
                }
                "delay" => delay += plugin.parameters["delay_ms"].as_f64().unwrap_or(0.0),
                _ => plugins.push(plugin.clone()),
            }
        }
        plugins.extend(chain.plugins.iter().cloned());
        let mut filter_chain = chain.clone();
        filter_chain.drivers = None;
        filter_chain.plugins = plugins.clone();
        let curve =
            crate::ctc::apply_channel_dsp_chain_to_curve(&filter_chain, &measurement, sample_rate)?;
        drivers.push(SubDriverInfo {
            name: driver_chain.name.clone(),
            gain,
            delay,
            inverted,
            initial_curve: Some(measurement),
            processing: Some(SubDriverProcessing { plugins, curve }),
        });
    }
    Ok(SubPreprocessResult {
        combined_curve: combined,
        shared_eq_seats: None,
        drivers: Some(drivers),
        common_eq_complete: true,
        joint_sub: chain.joint_sub.clone(),
        optimizer_evidence: evidence,
        advisories: vec![
            format!(
                "sub_alignment_strategy:{}",
                if chain.joint_sub.is_some() {
                    "joint_coherent".to_string()
                } else if ms.joint_optimization {
                    "joint_unavailable_detailed_fallback".to_string()
                } else {
                    optimizer
                        .multi_seat
                        .as_ref()
                        .filter(|seat| seat.enabled)
                        .map(|seat| format!("{:?}", seat.strategy))
                        .unwrap_or_else(|| "single_seat_allpass".into())
                }
            ),
            format!("sub_alignment_primary_seat:{primary}"),
        ],
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

    if curves.is_empty() {
        return Err(AutoeqError::InvalidMeasurement {
            message: "Independent sub array is empty".into(),
        });
    }
    let combined = if curves
        .iter()
        .all(roomeq_engine::topology::curve_has_usable_phase)
    {
        // These physical outputs carry the same logical input.
        roomeq_engine::dba::sum_array_response(&curves).map_err(|error| {
            AutoeqError::InvalidMeasurement {
                message: error.to_string(),
            }
        })?
    } else {
        log::warn!(
            "Independent subs lack measured phase: using an incoherent power approximation; coherent splice alignment is unavailable"
        );
        let ref_freq = curves[0].freq.clone();
        let mut sum_power = ndarray::Array1::<f64>::zeros(ref_freq.len());
        for curve in &curves {
            let interp = interpolate_log_space(&ref_freq, curve);
            sum_power += &interp.spl.mapv(|db| 10.0_f64.powf(db / 10.0));
        }
        Curve {
            freq: ref_freq,
            spl: sum_power.mapv(|power| 10.0 * power.max(1e-24).log10()),
            phase: None,
            ..Default::default()
        }
    };

    let drivers: Vec<SubDriverInfo> = curves
        .into_iter()
        .enumerate()
        .map(|(i, curve)| SubDriverInfo {
            name: format!("{}_{}", ms.name, i + 1),
            gain: 0.0,
            delay: 0.0,
            inverted: false,
            processing: None,
            initial_curve: Some(curve),
        })
        .collect();

    Ok(SubPreprocessResult {
        shared_eq_seats: None,
        common_eq_complete: false,
        joint_sub: None,
        optimizer_evidence: Vec::new(),
        advisories: Vec::new(),
        combined_curve: combined,
        drivers: Some(drivers),
    })
}

/// Render cardioid pairs at every seat, retaining primary-seat phase for routing.
///
/// # Errors
/// Rejects incomplete seat pairs, an unavailable primary seat, or invalid phase evidence.
pub(in super::super) fn preprocess_cardioid_with_frequency_samples(
    c: &CardioidConfig,
    frequency_samples: usize,
    primary_seat: usize,
) -> Result<SubPreprocessResult> {
    let load = |source, label| {
        crate::measurement::load_source_individual_with_frequency_samples(source, frequency_samples)
            .map_err(|error| AutoeqError::InvalidMeasurement {
                message: format!("Cardioid {label}: {error}"),
            })
    };
    let front = load(&c.front, "front")?;
    let rear = load(&c.rear, "rear")?;
    if front.is_empty() || front.len() != rear.len() {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!(
                "Cardioid requires paired front/rear seats, got {} and {}",
                front.len(),
                rear.len()
            ),
        });
    }
    if primary_seat >= front.len() {
        return Err(AutoeqError::InvalidMeasurement {
            message: format!("Cardioid primary seat {primary_seat} unavailable"),
        });
    }
    // Sum physical drivers within each synchronous seat before any spatial EQ
    // aggregation. Averaging driver magnitudes first destroys measured phase.
    let mut rendered = front
        .into_iter()
        .zip(rear)
        .map(|(front, rear)| render_cardioid_seat(c, front, rear))
        .collect::<Result<Vec<_>>>()?;
    let shared_eq_seats = (rendered.len() > 1).then(|| {
        rendered
            .iter()
            .map(|seat| seat.combined_curve.clone())
            .collect()
    });
    let mut result = rendered.swap_remove(primary_seat);
    result.shared_eq_seats = shared_eq_seats;
    Ok(result)
}

fn render_cardioid_seat(
    c: &CardioidConfig,
    front_curve: Curve,
    rear_curve: Curve,
) -> Result<SubPreprocessResult> {
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
    let grid = roomeq_engine::topology::shared_measurement_grid(&[&front_curve, &rear_curve])
        .ok_or_else(|| AutoeqError::InvalidMeasurement {
            message: "Cardioid measurements lack common frequency support".into(),
        })?;
    let front_curve = autoeq_core::interpolate_log_space(&grid, &front_curve);
    let rear_curve = autoeq_core::interpolate_log_space(&grid, &rear_curve);

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
            processing: None,
            initial_curve: Some(front_curve),
        },
        SubDriverInfo {
            name: "Rear Sub".to_string(),
            gain: 0.0,
            delay: delay_ms,
            inverted: true,
            processing: None,
            initial_curve: Some(rear_curve),
        },
    ];

    Ok(SubPreprocessResult {
        shared_eq_seats: None,
        common_eq_complete: false,
        joint_sub: None,
        optimizer_evidence: Vec::new(),
        advisories: Vec::new(),
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
            processing: None,
            initial_curve: Some(front_curve),
        },
        SubDriverInfo {
            name: "Rear Array".to_string(),
            gain: result.gains.get(1).copied().unwrap_or(0.0),
            delay: result.delays.get(1).copied().unwrap_or(0.0),
            inverted: true,
            processing: None,
            initial_curve: Some(rear_curve),
        },
    ];

    Ok(SubPreprocessResult {
        shared_eq_seats: None,
        common_eq_complete: false,
        joint_sub: None,
        optimizer_evidence: vec![optimized.optimizer_evidence],
        advisories: Vec::new(),
        combined_curve: combined,
        drivers: Some(drivers),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_engine::Curve;

    #[test]
    fn roadmap_correction_joint_routed_independent_strategy_cannot_silently_replace_selection() {
        let source = SpeakerConfig::MultiSub(MultiSubGroup {
            name: "subs".into(),
            speaker_name: None,
            subwoofers: vec![MeasurementSource::InMemory(make_curve(16, 80.0, Some(0.0))); 2],
            joint_optimization: true,
            allpass_optimization: false,
        });
        let result = preprocess_sub_with_frequency_samples(
            &source,
            &SubwooferStrategy::Single,
            &tiny_optimizer(),
            48_000.0,
            16,
        );
        let error = result
            .err()
            .expect("independent strategy must not silently replace joint mode");
        assert!(
            error
                .to_string()
                .contains("requires routed subwoofer strategy 'mso'")
        );
    }
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
    fn legacy_mso_retains_each_shared_eq_seat_separately_from_routing() {
        // Synthetic positive control with a declared common reference and
        // matching seat labels; plain in-memory phase arrays stay gain-only.
        let source = MeasurementSource::Multiple(autoeq_core::MeasurementMultiple {
            measurements: [("seat-0", 80.0, 37.0), ("seat-1", 90.0, -89.0)]
                .into_iter()
                .map(|(seat, level, phase)| autoeq_core::MeasurementRef::Loaded {
                    original: Box::new(autoeq_core::MeasurementRef::Named {
                        path: format!("synthetic-{seat}.csv").into(),
                        name: Some(seat.into()),
                    }),
                    loaded_response: Box::new(make_curve(16, level, Some(phase))),
                })
                .collect(),
            speaker_name: None,
            provenance: autoeq_core::MeasurementProvenance {
                capture_kind: autoeq_core::ProvenanceCaptureKind::StationaryIr,
                timing_reference_id: Some("synthetic-common-reference".into()),
                ..Default::default()
            },
        });
        let group = MultiSubGroup {
            name: "subs".into(),
            speaker_name: None,
            subwoofers: vec![source.clone(), source],
            allpass_optimization: false,
            joint_optimization: false,
        };
        let result =
            preprocess_multisub_mso_with_frequency_samples(&group, &tiny_optimizer(), 48_000.0, 16)
                .unwrap();
        let seats = result
            .shared_eq_seats
            .as_ref()
            .expect("shared EQ lost measured seats");
        assert_eq!(seats.len(), 2);
        for (first, second) in seats[0].spl.iter().zip(&seats[1].spl) {
            assert!((second - first - 10.0).abs() < 1e-8);
        }
        assert!(
            result.combined_curve.phase.is_some(),
            "routing lost complex phase"
        );
        assert!(!result.common_eq_complete);
        let routing = interpolate_log_space(&seats[0].freq, &result.combined_curve);
        for (actual, expected) in routing.spl.iter().zip(&seats[0].spl) {
            assert!(
                (actual - expected).abs() < 1e-8,
                "routing magnitude must belong to the same primary seat as its phase: {actual} vs {expected}"
            );
        }
        for (actual, expected) in routing
            .phase
            .as_ref()
            .unwrap()
            .iter()
            .zip(seats[0].phase.as_ref().unwrap())
        {
            assert!((actual - expected).to_radians().sin().abs() < 1e-8);
            assert!((actual - expected).to_radians().cos() > 0.0);
        }
        for driver in result.drivers.as_ref().unwrap() {
            let physical = driver.initial_curve.as_ref().unwrap();
            assert!(physical.spl.iter().all(|spl| (*spl - 80.0).abs() < 1e-8));
            assert!(
                physical
                    .phase
                    .as_ref()
                    .expect("routing lost primary-seat phase")
                    .iter()
                    .all(|phase| (*phase - 37.0).abs() < 1e-8)
            );
        }
    }

    #[test]
    fn legacy_mso_does_not_invent_coherent_shared_eq_seats_without_timing() {
        let source = MeasurementSource::InMemoryMultiple(vec![
            make_curve(16, 80.0, Some(37.0)),
            make_curve(16, 90.0, Some(-89.0)),
        ]);
        let group = MultiSubGroup {
            name: "subs".into(),
            speaker_name: None,
            subwoofers: vec![source.clone(), source],
            allpass_optimization: false,
            joint_optimization: false,
        };
        let result =
            preprocess_multisub_mso_with_frequency_samples(&group, &tiny_optimizer(), 48_000.0, 16)
                .unwrap();
        assert!(result.shared_eq_seats.is_none());
        assert!(result.combined_curve.phase.is_none());
        assert!(
            result
                .advisories
                .iter()
                .any(|reason| reason == "unverified_timing_shared_eq_seats_unavailable")
        );
        assert!(
            result
                .drivers
                .as_ref()
                .unwrap()
                .iter()
                .all(|driver| driver.delay == 0.0)
        );
    }

    #[test]
    fn legacy_mso_retains_gain_only_admission_reason() {
        let group = MultiSubGroup {
            name: "subs".into(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemory(make_curve(16, 80.0, None)),
                MeasurementSource::InMemory(make_curve(16, 82.0, None)),
            ],
            allpass_optimization: false,
            joint_optimization: false,
        };
        let result = preprocess_sub_with_frequency_samples(
            &SpeakerConfig::MultiSub(group),
            &SubwooferStrategy::Mso,
            &tiny_optimizer(),
            48_000.0,
            16,
        )
        .unwrap();
        assert!(
            result
                .advisories
                .iter()
                .any(|reason| reason == "missing_phase_gain_only")
        );
        assert!(result.combined_curve.phase.is_none());
        assert!(
            result
                .drivers
                .as_ref()
                .unwrap()
                .iter()
                .all(|driver| driver.delay == 0.0)
        );
    }

    #[test]
    fn preprocess_independent_subs_preserve_coherent_phase() {
        for (phase, expected) in [(0.0, 86.020599913), (180.0, -240.0)] {
            let group = MultiSubGroup {
                name: "subs".into(),
                speaker_name: None,
                subwoofers: vec![
                    MeasurementSource::InMemory(make_curve(16, 80.0, Some(0.0))),
                    MeasurementSource::InMemory(make_curve(16, 80.0, Some(phase))),
                ],
                allpass_optimization: false,
                joint_optimization: false,
            };
            let result =
                preprocess_multisub_independent_with_frequency_samples(&group, 16).unwrap();
            assert!(result.combined_curve.phase.is_some());
            assert!(
                result
                    .combined_curve
                    .spl
                    .iter()
                    .all(|&level| (level - expected).abs() < 2.0)
            );
            assert!(
                result
                    .drivers
                    .unwrap()
                    .iter()
                    .all(|driver| driver.gain == 0.0 && driver.delay == 0.0)
            );
        }
    }

    #[test]
    fn preprocess_allpass_retains_deployable_filters_and_replay() {
        // Synthetic positive control modeling declared same-seat capture
        // provenance, not evidence from physical measurement hardware.
        let declared = |curve| {
            MeasurementSource::Single(autoeq_core::MeasurementSingle {
                measurement: autoeq_core::MeasurementRef::Loaded {
                    original: Box::new(autoeq_core::MeasurementRef::Named {
                        path: "synthetic-allpass.csv".into(),
                        name: Some("seat-0".into()),
                    }),
                    loaded_response: Box::new(curve),
                },
                speaker_name: None,
                provenance: autoeq_core::MeasurementProvenance {
                    capture_kind: autoeq_core::ProvenanceCaptureKind::StationaryIr,
                    timing_reference_id: Some("synthetic-common-reference".into()),
                    ..Default::default()
                },
            })
        };
        let group = MultiSubGroup {
            name: "subs".into(),
            speaker_name: None,
            subwoofers: vec![
                declared(make_curve(16, 80.0, Some(0.0))),
                declared(make_curve(16, 78.0, Some(40.0))),
            ],
            allpass_optimization: true,
            joint_optimization: false,
        };
        let result =
            preprocess_multisub_mso_with_frequency_samples(&group, &tiny_optimizer(), 48000.0, 16)
                .unwrap();
        let mut realized = Vec::new();
        for driver in result.drivers.as_ref().unwrap() {
            let processing = driver
                .processing
                .as_ref()
                .expect("all-pass transfer retained");
            assert!(
                processing
                    .plugins
                    .iter()
                    .any(|plugin| plugin.parameters["label"] == "group_delay_allpass")
            );
            let mut curve = roomeq_engine::topology::apply_delay_and_polarity_to_curve(
                &processing.curve,
                driver.delay,
                driver.inverted,
            );
            curve.spl.mapv_inplace(|level| level + driver.gain);
            realized.push(curve);
        }
        let replay = roomeq_engine::dba::sum_array_response(&realized).unwrap();
        let expected = interpolate_log_space(&replay.freq, &result.combined_curve);
        assert!(
            replay
                .spl
                .iter()
                .zip(&expected.spl)
                .all(|(a, b)| (a - b).abs() < 0.1)
        );
    }

    #[test]
    fn preprocess_multiseat_uses_selected_primary_and_strategy() {
        let group = MultiSubGroup {
            name: "subs".into(),
            speaker_name: None,
            subwoofers: vec![
                MeasurementSource::InMemoryMultiple(vec![
                    make_curve(16, 80.0, Some(0.0)),
                    make_curve(16, 60.0, Some(0.0))
                ]);
                2
            ],
            allpass_optimization: false,
            joint_optimization: false,
        };
        let mut optimizer = tiny_optimizer();
        optimizer.multi_seat = Some(roomeq_model::MultiSeatConfig {
            enabled: true,
            strategy: roomeq_model::MultiSeatStrategy::Average,
            primary_seat: 1,
            per_sub_peq: false,
            global_eq: false,
            search: Some(roomeq_model::MultiSeatSearchConfig {
                evaluation_budget: Some(30),
                seed: Some(42),
                ..Default::default()
            }),
            ..Default::default()
        });
        let result =
            preprocess_multisub_mso_with_frequency_samples(&group, &optimizer, 48000.0, 16)
                .unwrap();
        assert!(
            result
                .advisories
                .iter()
                .any(|advisory| advisory == "sub_alignment_strategy:Average")
        );
        for driver in result.drivers.as_ref().unwrap() {
            assert!(
                driver
                    .initial_curve
                    .as_ref()
                    .unwrap()
                    .spl
                    .iter()
                    .all(|&level| level == 60.0)
            );
            assert!(driver.processing.as_ref().unwrap().plugins.is_empty());
        }
        let realized: Vec<_> = result
            .drivers
            .as_ref()
            .unwrap()
            .iter()
            .map(|driver| {
                let mut curve = roomeq_engine::topology::apply_delay_and_polarity_to_curve(
                    &driver.processing.as_ref().unwrap().curve,
                    driver.delay,
                    driver.inverted,
                );
                curve.spl.mapv_inplace(|level| level + driver.gain);
                curve
            })
            .collect();
        let replay = roomeq_engine::dba::sum_array_response(&realized).unwrap();
        let expected = interpolate_log_space(&replay.freq, &result.combined_curve);
        assert!(
            replay
                .spl
                .iter()
                .zip(&expected.spl)
                .all(|(a, b)| (a - b).abs() < 0.1)
        );
        assert!(result.combined_curve.spl.iter().all(|&level| level < 75.0));
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
    fn single_sub_keeps_spatial_eq_separate_from_primary_phase() {
        let first = make_curve(16, 80.0, Some(37.0));
        let second = make_curve(16, 90.0, Some(-89.0));
        let speaker =
            SpeakerConfig::Single(MeasurementSource::InMemoryMultiple(vec![first, second]));
        let mut optimizer = tiny_optimizer();
        optimizer.multi_seat = Some(roomeq_model::MultiSeatConfig {
            primary_seat: 1,
            ..Default::default()
        });
        let result = preprocess_sub_with_frequency_samples(
            &speaker,
            &SubwooferStrategy::Single,
            &optimizer,
            48000.0,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        )
        .unwrap();
        assert_eq!(result.shared_eq_seats.as_ref().unwrap().len(), 2);
        assert!(
            result
                .combined_curve
                .spl
                .iter()
                .all(|v| (*v - 90.0).abs() < 1e-8)
        );
        assert!(
            result
                .combined_curve
                .phase
                .unwrap()
                .iter()
                .all(|v| (*v + 89.0).abs() < 1e-8)
        );

        optimizer.multi_seat.as_mut().unwrap().primary_seat = 2;
        assert!(
            preprocess_sub_with_frequency_samples(
                &speaker,
                &SubwooferStrategy::Single,
                &optimizer,
                48000.0,
                crate::DEFAULT_FREQUENCY_SAMPLES,
            )
            .err()
            .unwrap()
            .to_string()
            .contains("primary seat 2 unavailable")
        );
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
            joint_optimization: false,
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
        assert!(result.combined_curve.phase.is_none());
        assert!(
            result
                .advisories
                .contains(&"unverified_timing_gain_only".into())
        );
        assert!(drivers.iter().all(|driver| driver.delay == 0.0));
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
            joint_optimization: false,
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
    fn preprocess_cardioid_preserves_paired_seats_and_primary_phase() {
        let front = make_curve(16, 80.0, Some(30.0));
        let mut rear = make_curve(16, 74.0, Some(0.0));
        // A one-millisecond propagation phase cancels the configured delay.
        // Seat zero subtracts the rear; seat one adds it after inversion.
        rear.phase = Some(rear.freq.mapv(|f| 30.0 + 360.0 * f * 0.001));
        let mut second_rear = rear.clone();
        second_rear
            .phase
            .as_mut()
            .unwrap()
            .mapv_inplace(|p| p + 180.0);
        let mut config = CardioidConfig {
            name: "paired cardioid".into(),
            speaker_name: None,
            front: MeasurementSource::InMemoryMultiple(vec![front.clone(), front]),
            rear: MeasurementSource::InMemoryMultiple(vec![rear, second_rear]),
            separation_meters: 0.343,
        };
        let mut optimizer = tiny_optimizer();
        optimizer.multi_seat = Some(roomeq_model::MultiSeatConfig {
            primary_seat: 1,
            ..Default::default()
        });
        let result = preprocess_sub_with_frequency_samples(
            &SpeakerConfig::Cardioid(Box::new(config.clone())),
            &SubwooferStrategy::Single,
            &optimizer,
            48000.0,
            crate::DEFAULT_FREQUENCY_SAMPLES,
        )
        .unwrap();
        let seats = result.shared_eq_seats.as_ref().unwrap();
        assert_eq!(seats.len(), 2);
        let front_amplitude = 10.0_f64.powf(80.0 / 20.0);
        let rear_amplitude = 10.0_f64.powf(74.0 / 20.0);
        for (seat, amplitude) in seats.iter().zip([
            front_amplitude - rear_amplitude,
            front_amplitude + rear_amplitude,
        ]) {
            let expected = 20.0 * amplitude.log10();
            assert!(seat.spl.iter().all(|value| (value - expected).abs() < 1e-8));
            assert!(
                seat.phase
                    .as_ref()
                    .unwrap()
                    .iter()
                    .all(|p| (p - 30.0).abs() < 1e-8)
            );
        }
        assert_eq!(result.combined_curve.spl, seats[1].spl);
        assert_eq!(result.combined_curve.phase, seats[1].phase);
        let drivers = result.drivers.as_ref().unwrap();
        let expected_rear = match &config.rear {
            MeasurementSource::InMemoryMultiple(curves) => curves[1].phase.as_ref().unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(
            drivers[1]
                .initial_curve
                .as_ref()
                .unwrap()
                .phase
                .as_ref()
                .unwrap(),
            expected_rear
        );

        let invalid_primary = preprocess_cardioid_with_frequency_samples(
            &config,
            crate::DEFAULT_FREQUENCY_SAMPLES,
            2,
        )
        .err()
        .unwrap();
        assert!(
            invalid_primary
                .to_string()
                .contains("primary seat 2 unavailable")
        );

        if let MeasurementSource::InMemoryMultiple(curves) = &mut config.rear {
            curves[1].phase = None;
        }
        let missing_phase = preprocess_cardioid_with_frequency_samples(
            &config,
            crate::DEFAULT_FREQUENCY_SAMPLES,
            0,
        )
        .err()
        .unwrap();
        assert!(
            missing_phase
                .to_string()
                .contains("requires measured phase")
        );

        if let MeasurementSource::InMemoryMultiple(curves) = &mut config.rear {
            curves.pop();
        }
        let missing_seat = preprocess_cardioid_with_frequency_samples(
            &config,
            crate::DEFAULT_FREQUENCY_SAMPLES,
            0,
        )
        .err()
        .unwrap();
        assert!(missing_seat.to_string().contains("paired front/rear seats"));
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
        let result = preprocess_cardioid_with_frequency_samples(
            &config,
            crate::DEFAULT_FREQUENCY_SAMPLES,
            0,
        );
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
        let result = preprocess_cardioid_with_frequency_samples(
            &config,
            crate::DEFAULT_FREQUENCY_SAMPLES,
            0,
        );
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

        let error = preprocess_cardioid_with_frequency_samples(
            &config,
            crate::DEFAULT_FREQUENCY_SAMPLES,
            0,
        )
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
        let result = preprocess_cardioid_with_frequency_samples(
            &config,
            crate::DEFAULT_FREQUENCY_SAMPLES,
            0,
        );
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
        let result = preprocess_cardioid_with_frequency_samples(
            &config,
            crate::DEFAULT_FREQUENCY_SAMPLES,
            0,
        );
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
