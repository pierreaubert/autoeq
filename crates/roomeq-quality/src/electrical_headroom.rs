//! Sampled electrical transfer-matrix headroom, independent of room acoustics.
//!
//! Callers must supply realized electrical paths (including route and DSP gain),
//! not microphone responses. This samples steady-state sinusoidal gain; it is
//! not a continuous-frequency supremum or a transient/true-peak certificate.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub struct ElectricalPath<'a> {
    pub input: &'a str,
    pub output: &'a str,
    /// Complex electrical response on the exact supplied frequency grid.
    pub transfer: &'a [Complex64],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampledElectricalOutputPeak {
    pub output: String,
    pub inputs: Vec<String>,
    pub input_peak_limits: BTreeMap<String, f64>,
    pub sample_rate_hz: f64,
    pub evaluated_band_hz: [f64; 2],
    pub grid_points: usize,
    pub peak_frequency_hz: f64,
    pub peak_amplitude: f64,
    /// None means exact zero transfer on every evaluated frequency, not unknown.
    pub peak_dbfs: Option<f64>,
    pub required_attenuation_db: f64,
}

/// Bound simultaneous independently phased sinusoidal inputs at each frequency.
/// Input limits are peak linear amplitudes relative to digital full scale.
/// Paths from the same input add complexly before independent-input magnitudes
/// are added. A room cancellation must never be supplied as electrical gain.
pub fn evaluate_sampled_electrical_headroom(
    frequencies_hz: &[f64],
    sample_rate_hz: f64,
    paths: &[ElectricalPath<'_>],
    input_peak_limits: &BTreeMap<String, f64>,
) -> Result<Vec<SampledElectricalOutputPeak>, String> {
    if !sample_rate_hz.is_finite()
        || sample_rate_hz <= 0.0
        || frequencies_hz.len() < 2
        || frequencies_hz
            .iter()
            .any(|f| !f.is_finite() || *f < 0.0 || *f > sample_rate_hz / 2.0)
        || frequencies_hz.windows(2).any(|w| w[0] >= w[1])
    {
        return Err("invalid electrical assessment rate/grid".into());
    }
    if paths.is_empty()
        || input_peak_limits.is_empty()
        || input_peak_limits
            .iter()
            .any(|(name, v)| name.is_empty() || !v.is_finite() || *v < 0.0)
    {
        return Err("missing or invalid electrical paths/input limits".into());
    }
    let mut matrix: BTreeMap<(&str, &str), Vec<Complex64>> = BTreeMap::new();
    let mut outputs = BTreeSet::new();
    for path in paths {
        if path.input.is_empty()
            || path.output.is_empty()
            || !input_peak_limits.contains_key(path.input)
            || path.transfer.len() != frequencies_hz.len()
            || path
                .transfer
                .iter()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err("invalid or unbounded electrical path".into());
        }
        outputs.insert(path.output);
        let total = matrix
            .entry((path.output, path.input))
            .or_insert_with(|| vec![Complex64::new(0.0, 0.0); frequencies_hz.len()]);
        for (sum, branch) in total.iter_mut().zip(path.transfer) {
            *sum += branch;
        }
    }
    let mut result = Vec::new();
    for output in outputs {
        let input_paths: Vec<_> = matrix
            .iter()
            .filter(|((name, _), _)| *name == output)
            .collect();
        let mut peak = 0.0_f64;
        let mut peak_index = 0;
        for index in 0..frequencies_hz.len() {
            let amplitude: f64 = input_paths
                .iter()
                .map(|((_, input), transfer)| transfer[index].norm() * input_peak_limits[*input])
                .sum();
            if !amplitude.is_finite() {
                return Err("electrical transfer accumulation overflowed".into());
            }
            if amplitude > peak {
                peak = amplitude;
                peak_index = index;
            }
        }
        let peak_dbfs = (peak > 0.0).then(|| 20.0 * peak.log10());
        result.push(SampledElectricalOutputPeak {
            output: output.into(),
            inputs: input_paths
                .iter()
                .map(|((_, input), _)| (*input).to_owned())
                .collect(),
            input_peak_limits: input_paths
                .iter()
                .map(|((_, input), _)| ((*input).to_owned(), input_peak_limits[*input]))
                .collect(),
            sample_rate_hz,
            evaluated_band_hz: [frequencies_hz[0], *frequencies_hz.last().unwrap()],
            grid_points: frequencies_hz.len(),
            peak_frequency_hz: frequencies_hz[peak_index],
            peak_amplitude: peak,
            peak_dbfs,
            required_attenuation_db: peak_dbfs.unwrap_or(0.0).max(0.0),
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_inputs_cannot_claim_coherent_cancellation() {
        let positive = [Complex64::new(1.0, 0.0); 2];
        let negative = [Complex64::new(-1.0, 0.0); 2];
        let limits = BTreeMap::from([("L".into(), 1.0), ("R".into(), 1.0)]);
        let mut paths = [
            ElectricalPath {
                input: "L",
                output: "sub",
                transfer: &positive,
            },
            ElectricalPath {
                input: "R",
                output: "sub",
                transfer: &negative,
            },
        ];
        let evaluate = |paths: &[ElectricalPath<'_>]| {
            evaluate_sampled_electrical_headroom(&[0.0, 24000.0], 48000.0, paths, &limits).unwrap()
        };
        let independent = evaluate(&paths);
        assert!((independent[0].required_attenuation_db - 6.020599913).abs() < 1e-8);
        paths[1].input = "L";
        let coherent = evaluate(&paths);
        assert_eq!(coherent[0].peak_amplitude, 0.0);
        assert_eq!(coherent[0].peak_dbfs, None);
    }

    #[test]
    fn distinct_physical_outputs_never_cancel_each_others_headroom() {
        let positive = [Complex64::new(10.0, 0.0); 2];
        let negative = [Complex64::new(-10.0, 0.0); 2];
        let paths = [
            ElectricalPath {
                input: "L",
                output: "sub_a",
                transfer: &positive,
            },
            ElectricalPath {
                input: "L",
                output: "sub_b",
                transfer: &negative,
            },
        ];
        let result = evaluate_sampled_electrical_headroom(
            &[20.0, 200.0],
            48000.0,
            &paths,
            &BTreeMap::from([("L".into(), 1.0)]),
        )
        .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].output, "sub_a");
        assert_eq!(result[1].output, "sub_b");
        for output in result {
            assert_eq!(output.required_attenuation_db, 20.0);
        }
    }

    #[test]
    fn six_three_db_peqs_require_eighteen_db_at_their_common_center() {
        let filter = math_audio_iir_fir::Biquad::new(
            math_audio_iir_fir::BiquadFilterType::Peak,
            80.0,
            48000.0,
            2.0,
            3.0,
        );
        let frequencies = [20.0, 80.0, 20000.0];
        let (a1, a2, b0, b1, b2) = filter.constants();
        let transfer: Vec<_> = frequencies
            .iter()
            .map(|f| {
                let z = Complex64::from_polar(1.0, -2.0 * std::f64::consts::PI * f / 48000.0);
                let section = (b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z);
                section.powu(6)
            })
            .collect();
        let paths = [ElectricalPath {
            input: "L",
            output: "left",
            transfer: &transfer,
        }];
        let limits = BTreeMap::from([("L".into(), 1.0)]);
        let result =
            evaluate_sampled_electrical_headroom(&frequencies, 48000.0, &paths, &limits).unwrap();
        assert_eq!(filter.db_gain, 3.0);
        assert!((result[0].required_attenuation_db - 18.0).abs() < 1e-8);
        assert_eq!(result[0].peak_frequency_hz, 80.0);
    }

    #[test]
    fn missing_limits_misaligned_arrays_and_nonfinite_transfer_fail() {
        let transfer = [Complex64::new(1.0, 0.0); 2];
        let paths = [ElectricalPath {
            input: "L",
            output: "left",
            transfer: &transfer,
        }];
        let limits = BTreeMap::from([("L".into(), 0.5)]);
        assert!(
            evaluate_sampled_electrical_headroom(
                &[0.0, 24000.0],
                48000.0,
                &paths,
                &BTreeMap::new()
            )
            .is_err()
        );
        assert!(
            evaluate_sampled_electrical_headroom(&[20.0, 80.0, 100.0], 48000.0, &paths, &limits)
                .is_err()
        );
        assert!(
            evaluate_sampled_electrical_headroom(&[20.0, 30000.0], 48000.0, &paths, &limits)
                .is_err()
        );
        let invalid = [Complex64::new(f64::NAN, 0.0); 2];
        let paths = [ElectricalPath {
            input: "L",
            output: "left",
            transfer: &invalid,
        }];
        assert!(
            evaluate_sampled_electrical_headroom(&[20.0, 80.0], 48000.0, &paths, &limits).is_err()
        );
    }

    #[test]
    fn fir_cascade_route_gain_and_input_limit_are_applied_at_each_rate() {
        for rate in [44100.0, 48000.0, 96000.0] {
            let frequencies = [0.0, rate / 4.0, rate / 2.0];
            // Independently evaluated DFT of the cascade [1,1] * [1,1],
            // followed by a 0.5 linear route gain. Its DC response is 2.
            let taps = [1.0, 2.0, 1.0];
            let transfer: Vec<Complex64> = frequencies
                .iter()
                .map(|f| {
                    taps.iter()
                        .enumerate()
                        .map(|(n, tap)| {
                            Complex64::from_polar(
                                *tap * 0.5,
                                -2.0 * std::f64::consts::PI * f * n as f64 / rate,
                            )
                        })
                        .sum()
                })
                .collect();
            let paths = [ElectricalPath {
                input: "L",
                output: "driver",
                transfer: &transfer,
            }];
            let limits = BTreeMap::from([("L".into(), 0.25)]);
            let output =
                evaluate_sampled_electrical_headroom(&frequencies, rate, &paths, &limits).unwrap();
            assert!((output[0].peak_amplitude - 0.5).abs() < 1e-12);
            assert!((output[0].peak_dbfs.unwrap() + 6.020599913).abs() < 1e-8);
            assert_eq!(output[0].required_attenuation_db, 0.0);
            assert_eq!(output[0].sample_rate_hz, rate);
            assert_eq!(output[0].input_peak_limits, limits);
            assert!(serde_json::to_string(&output).is_ok());
        }
    }
}
