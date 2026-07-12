use super::types::{
    AcousticOracle, OracleParameters, ParallelSourceParameters, ProhibitedBehavior,
};
use math_audio_iir_fir::Biquad;
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::{PI, SQRT_2};

/// Standard log-frequency axis for analytic fixtures.
pub fn log_frequency_grid(points: usize, min_hz: f64, max_hz: f64) -> Array1<f64> {
    assert!(points >= 2, "an oracle grid needs at least two points");
    assert!(min_hz > 0.0 && max_hz > min_hz);
    Array1::logspace(10.0, min_hz.log10(), max_hz.log10(), points)
}

fn oracle(
    name: impl Into<String>,
    frequencies_hz: Array1<f64>,
    expected_transfer: Array1<Complex64>,
    components: Vec<Array1<Complex64>>,
    generating_parameters: OracleParameters,
    valid_correction_region_hz: (f64, f64),
    prohibited_behaviors: Vec<ProhibitedBehavior>,
) -> AcousticOracle {
    AcousticOracle {
        name: name.into(),
        frequencies_hz,
        expected_transfer,
        components,
        generating_parameters,
        valid_correction_region_hz,
        prohibited_behaviors,
    }
}

pub fn identity_oracle(frequencies_hz: Array1<f64>) -> AcousticOracle {
    let region = (frequencies_hz[0], frequencies_hz[frequencies_hz.len() - 1]);
    let transfer = Array1::from_elem(frequencies_hz.len(), Complex64::new(1.0, 0.0));
    oracle(
        "identity",
        frequencies_hz,
        transfer,
        Vec::new(),
        OracleParameters::Identity,
        region,
        vec![ProhibitedBehavior::NonFiniteTransfer],
    )
}

pub fn delay_oracle(frequencies_hz: Array1<f64>, delay_ms: f64) -> AcousticOracle {
    let delay_s = delay_ms / 1000.0;
    let transfer = frequencies_hz
        .mapv(|frequency| Complex64::from_polar(1.0, -2.0 * PI * frequency * delay_s));
    let region = (frequencies_hz[0], frequencies_hz[frequencies_hz.len() - 1]);
    oracle(
        format!("delay_{delay_ms:.3}ms"),
        frequencies_hz,
        transfer,
        Vec::new(),
        OracleParameters::Delay { delay_ms },
        region,
        vec![
            ProhibitedBehavior::NonFiniteTransfer,
            ProhibitedBehavior::GroupDelayResidual { max_rms_ms: 0.02 },
        ],
    )
}

pub fn polarity_oracle(frequencies_hz: Array1<f64>, inverted: bool) -> AcousticOracle {
    let gain = if inverted { -1.0 } else { 1.0 };
    let transfer = Array1::from_elem(frequencies_hz.len(), Complex64::new(gain, 0.0));
    let region = (frequencies_hz[0], frequencies_hz[frequencies_hz.len() - 1]);
    oracle(
        if inverted {
            "polarity_inverted"
        } else {
            "polarity_normal"
        },
        frequencies_hz,
        transfer,
        Vec::new(),
        OracleParameters::Polarity { inverted },
        region,
        vec![ProhibitedBehavior::NonFiniteTransfer],
    )
}

fn allpass_transfer(frequency: f64, center_hz: f64, q: f64) -> Complex64 {
    let s = Complex64::new(0.0, frequency / center_hz);
    let numerator = s * s - s / q + 1.0;
    let denominator = s * s + s / q + 1.0;
    numerator / denominator
}

pub fn allpass_oracle(frequencies_hz: Array1<f64>, center_hz: f64, q: f64) -> AcousticOracle {
    let transfer = frequencies_hz.mapv(|frequency| allpass_transfer(frequency, center_hz, q));
    let lo = frequencies_hz[0].max(center_hz / 4.0);
    let hi = frequencies_hz[frequencies_hz.len() - 1].min(center_hz * 4.0);
    oracle(
        format!("allpass_{center_hz:.0}hz_q{q:.2}"),
        frequencies_hz,
        transfer,
        Vec::new(),
        OracleParameters::AllPass { center_hz, q },
        (lo, hi),
        vec![
            ProhibitedBehavior::NonFiniteTransfer,
            ProhibitedBehavior::CorrectionOutsideRegion { max_abs_db: 0.05 },
        ],
    )
}

/// Analytic fourth-order Linkwitz-Riley low/high branches and their complex sum.
pub fn linkwitz_riley4_oracle(frequencies_hz: Array1<f64>, crossover_hz: f64) -> AcousticOracle {
    let mut low = Array1::zeros(frequencies_hz.len());
    let mut high = Array1::zeros(frequencies_hz.len());
    for (index, &frequency) in frequencies_hz.iter().enumerate() {
        let s = Complex64::new(0.0, frequency / crossover_hz);
        let denominator = s * s + SQRT_2 * s + 1.0;
        low[index] = 1.0 / (denominator * denominator);
        high[index] = s.powu(4) / (denominator * denominator);
    }
    let combined = &low + &high;
    let lo = frequencies_hz[0].max(crossover_hz / 4.0);
    let hi = frequencies_hz[frequencies_hz.len() - 1].min(crossover_hz * 4.0);
    oracle(
        format!("lr4_{crossover_hz:.0}hz"),
        frequencies_hz,
        combined,
        vec![low, high],
        OracleParameters::LinkwitzRiley4 { crossover_hz },
        (lo, hi),
        vec![
            ProhibitedBehavior::NonFiniteTransfer,
            ProhibitedBehavior::GroupDelayResidual { max_rms_ms: 0.05 },
        ],
    )
}

fn room_mode_transfer(frequency: f64, center_hz: f64, q: f64, peak_gain_db: f64) -> Complex64 {
    let ratio = frequency / center_hz;
    let peak_linear = 10.0_f64.powf(peak_gain_db / 20.0);
    let denominator = Complex64::new(1.0 - ratio * ratio, ratio / q);
    let numerator = Complex64::new(0.0, (peak_linear - 1.0) * ratio / q);
    Complex64::new(1.0, 0.0) + numerator / denominator
}

pub fn room_mode_oracle(
    frequencies_hz: Array1<f64>,
    center_hz: f64,
    q: f64,
    peak_gain_db: f64,
) -> AcousticOracle {
    let transfer =
        frequencies_hz.mapv(|frequency| room_mode_transfer(frequency, center_hz, q, peak_gain_db));
    let half_width_octaves = (1.0 / q.max(0.5)).max(0.05);
    oracle(
        format!("room_mode_{center_hz:.0}hz_q{q:.1}"),
        frequencies_hz,
        transfer,
        Vec::new(),
        OracleParameters::RoomMode {
            center_hz,
            q,
            peak_gain_db,
        },
        (
            center_hz / 2.0_f64.powf(half_width_octaves * 2.0),
            center_hz * 2.0_f64.powf(half_width_octaves * 2.0),
        ),
        vec![ProhibitedBehavior::NonFiniteTransfer],
    )
}

pub fn parallel_woofers_oracle(
    frequencies_hz: Array1<f64>,
    sources: Vec<ParallelSourceParameters>,
) -> AcousticOracle {
    let components = sources
        .iter()
        .map(|source| {
            let amplitude =
                10.0_f64.powf(source.gain_db / 20.0) * if source.inverted { -1.0 } else { 1.0 };
            let delay_s = source.delay_ms / 1000.0;
            frequencies_hz
                .mapv(|frequency| Complex64::from_polar(amplitude, -2.0 * PI * frequency * delay_s))
        })
        .collect::<Vec<_>>();
    let combined = components
        .iter()
        .fold(Array1::zeros(frequencies_hz.len()), |sum, item| sum + item);
    let region = (frequencies_hz[0], frequencies_hz[frequencies_hz.len() - 1]);
    oracle(
        format!("parallel_{}_woofers", sources.len()),
        frequencies_hz,
        combined,
        components,
        OracleParameters::ParallelWoofers { sources },
        region,
        vec![
            ProhibitedBehavior::NonFiniteTransfer,
            ProhibitedBehavior::Latency { max_ms: 25.0 },
        ],
    )
}

pub fn excess_phase_oracle(
    frequencies_hz: Array1<f64>,
    sections: Vec<(f64, f64)>,
) -> AcousticOracle {
    let transfer = frequencies_hz.mapv(|frequency| {
        sections
            .iter()
            .fold(Complex64::new(1.0, 0.0), |sum, &(center, q)| {
                sum * allpass_transfer(frequency, center, q)
            })
    });
    let lo = sections
        .iter()
        .map(|(center, _)| center / 4.0)
        .fold(f64::INFINITY, f64::min)
        .max(frequencies_hz[0]);
    let hi = sections
        .iter()
        .map(|(center, _)| center * 4.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .min(frequencies_hz[frequencies_hz.len() - 1]);
    oracle(
        "excess_phase",
        frequencies_hz,
        transfer,
        Vec::new(),
        OracleParameters::ExcessPhase { sections },
        (lo, hi),
        vec![
            ProhibitedBehavior::NonFiniteTransfer,
            ProhibitedBehavior::CorrectionOutsideRegion { max_abs_db: 0.05 },
            ProhibitedBehavior::PreRinging {
                max_energy_db: -20.0,
            },
        ],
    )
}

pub fn comb_null_oracle(
    frequencies_hz: Array1<f64>,
    reflection_gain: f64,
    reflection_delay_ms: f64,
) -> AcousticOracle {
    let delay_s = reflection_delay_ms / 1000.0;
    let reflected = frequencies_hz
        .mapv(|frequency| Complex64::from_polar(reflection_gain, -2.0 * PI * frequency * delay_s));
    let direct = Array1::from_elem(frequencies_hz.len(), Complex64::new(1.0, 0.0));
    let combined = &direct + &reflected;
    let first_null_hz = 1.0 / (2.0 * delay_s);
    let region = (frequencies_hz[0], frequencies_hz[frequencies_hz.len() - 1]);
    oracle(
        format!("comb_null_{first_null_hz:.0}hz"),
        frequencies_hz,
        combined,
        vec![direct, reflected],
        OracleParameters::CombNull {
            reflection_gain,
            reflection_delay_ms,
        },
        region,
        vec![
            ProhibitedBehavior::NonFiniteTransfer,
            ProhibitedBehavior::BoostIntoNull {
                center_hz: first_null_hz,
                half_width_octaves: 1.0 / 12.0,
                max_boost_db: 3.0,
            },
        ],
    )
}

/// Schroeder frequency from room volume and RT60.
pub fn schroeder_frequency_hz(volume_m3: f64, rt60_seconds: f64) -> f64 {
    2000.0 * (rt60_seconds / volume_m3).sqrt()
}

pub fn room_transition_oracle(
    frequencies_hz: Array1<f64>,
    volume_m3: f64,
    rt60_seconds: f64,
    transition_octaves: f64,
) -> AcousticOracle {
    let schroeder_hz = schroeder_frequency_hz(volume_m3, rt60_seconds);
    let modal = frequencies_hz
        .mapv(|frequency| room_mode_transfer(frequency, schroeder_hz * 0.55, 8.0, 6.0));
    let diffuse = frequencies_hz.mapv(|frequency| {
        let tilt_db = -0.8 * (frequency / schroeder_hz).max(1.0).log2();
        Complex64::new(10.0_f64.powf(tilt_db / 20.0), 0.0)
    });
    let transfer = Array1::from_iter(
        frequencies_hz
            .iter()
            .zip(modal.iter().zip(diffuse.iter()))
            .map(|(&frequency, (&modal_value, &diffuse_value))| {
                let distance_octaves = (frequency / schroeder_hz).log2();
                let modal_weight =
                    1.0 / (1.0 + (4.0 * distance_octaves / transition_octaves.max(0.05)).exp());
                modal_value * modal_weight + diffuse_value * (1.0 - modal_weight)
            }),
    );
    oracle(
        format!("room_transition_{schroeder_hz:.0}hz"),
        frequencies_hz,
        transfer,
        vec![modal, diffuse],
        OracleParameters::RoomTransition {
            volume_m3,
            rt60_seconds,
            schroeder_hz,
            transition_octaves,
        },
        (schroeder_hz / 4.0, schroeder_hz * 4.0),
        vec![
            ProhibitedBehavior::NonFiniteTransfer,
            ProhibitedBehavior::CorrectionOutsideRegion { max_abs_db: 1.0 },
        ],
    )
}

/// Independent coefficient-domain oracle for a generated biquad cascade.
pub fn biquad_cascade_oracle(
    name: impl Into<String>,
    frequencies_hz: Array1<f64>,
    filters: &[Biquad],
    sample_rate: f64,
) -> AcousticOracle {
    let coefficients = filters.iter().map(Biquad::constants).collect::<Vec<_>>();
    let transfer = frequencies_hz.mapv(|frequency| {
        let z_inverse = Complex64::from_polar(1.0, -2.0 * PI * frequency / sample_rate);
        let z_inverse_squared = z_inverse * z_inverse;
        coefficients
            .iter()
            .fold(Complex64::new(1.0, 0.0), |total, &(a1, a2, b0, b1, b2)| {
                let numerator = b0 + b1 * z_inverse + b2 * z_inverse_squared;
                let denominator = 1.0 + a1 * z_inverse + a2 * z_inverse_squared;
                total * numerator / denominator
            })
    });
    let region = (frequencies_hz[0], frequencies_hz[frequencies_hz.len() - 1]);
    oracle(
        name,
        frequencies_hz,
        transfer,
        Vec::new(),
        OracleParameters::BiquadCascade {
            section_count: filters.len(),
        },
        region,
        vec![ProhibitedBehavior::NonFiniteTransfer],
    )
}

/// Fast analytic corpus used by the PR gate.
pub fn analytic_oracle_suite() -> Vec<AcousticOracle> {
    let full = || log_frequency_grid(257, 20.0, 20_000.0);
    let bass = || log_frequency_grid(193, 20.0, 300.0);
    vec![
        identity_oracle(full()),
        delay_oracle(full(), 2.5),
        polarity_oracle(full(), true),
        allpass_oracle(bass(), 80.0, 1.2),
        linkwitz_riley4_oracle(full(), 100.0),
        room_mode_oracle(bass(), 63.0, 8.0, 8.0),
        parallel_woofers_oracle(
            bass(),
            vec![
                ParallelSourceParameters {
                    gain_db: 0.0,
                    delay_ms: 0.0,
                    inverted: false,
                },
                ParallelSourceParameters {
                    gain_db: -1.5,
                    delay_ms: 2.0,
                    inverted: false,
                },
            ],
        ),
        excess_phase_oracle(full(), vec![(70.0, 1.0), (180.0, 1.4)]),
        comb_null_oracle(full(), 0.98, 5.0),
        room_transition_oracle(full(), 30.0, 0.4, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_woofer_delay_is_expressed_in_milliseconds() {
        let oracle = parallel_woofers_oracle(
            Array1::from(vec![100.0, 200.0]),
            vec![ParallelSourceParameters {
                gain_db: 0.0,
                delay_ms: 2.5,
                inverted: false,
            }],
        );
        let expected = Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_2);
        assert!((oracle.components[0][0] - expected).norm() < 1e-12);
    }

    #[test]
    fn room_transition_modal_component_uses_fractional_schroeder_frequency() {
        let frequencies = Array1::from(vec![100.0, 200.0]);
        let oracle = room_transition_oracle(frequencies, 100.0, 1.0, 1.0);
        let expected = room_mode_transfer(200.0, 110.0, 8.0, 6.0);
        assert!((oracle.components[0][1] - expected).norm() < 1e-12);
    }

    #[test]
    fn polarity_oracle_preserves_the_requested_sign() {
        let frequencies = Array1::from(vec![100.0, 200.0]);
        let normal = polarity_oracle(frequencies.clone(), false);
        let inverted = polarity_oracle(frequencies, true);

        assert_eq!(normal.expected_transfer[0], Complex64::new(1.0, 0.0));
        assert_eq!(inverted.expected_transfer[0], Complex64::new(-1.0, 0.0));
    }

    #[test]
    fn room_transition_is_evenly_blended_at_the_schroeder_frequency() {
        let schroeder_hz = schroeder_frequency_hz(100.0, 1.0);
        let oracle = room_transition_oracle(
            Array1::from(vec![schroeder_hz, schroeder_hz * 2.0, schroeder_hz * 4.0]),
            100.0,
            1.0,
            1.0,
        );
        let expected = (oracle.components[0][0] + oracle.components[1][0]) * 0.5;
        let expected_diffuse_at_two_octaves = Complex64::new(10.0_f64.powf(-1.6 / 20.0), 0.0);

        assert!((oracle.expected_transfer[0] - expected).norm() < 1e-12);
        assert!((oracle.components[1][2] - expected_diffuse_at_two_octaves).norm() < 1e-12);
    }
}
