//! Finite-observation octave decay diagnostics for identified baseline/candidate captures.
//!
//! Reuses the ETC's finite, zero-extended analytic filters. Reverse integration
//! stops before the operator-declared noise window. Noise is estimated separately
//! for each capture; it is not subtracted or treated as calibrated ambient SPL.
//! Fits describe this playback observation, not passive-room damping or audibility.

use super::{
    ViewProvenance,
    captured_etc::{envelope, kernel},
    fit_t60,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Explicit noise-region declaration and engineering fit-admission budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchedDecaySettings {
    /// Signal-free interval in both raw captures, relative to their common sample zero.
    pub noise_window_ms: [f64; 2],
    /// Required estimated signal-to-noise energy margin throughout the fit, in dB.
    pub minimum_fit_margin_db: f64,
    /// Minimum coefficient of determination for the retained -5 to -25 dB fit.
    pub minimum_r_squared: f64,
}

/// A finite-window -5 to -25 dB slope extrapolated to sixty decibels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayFit {
    /// Sixty-decibel extrapolation in seconds, not a passive-room RT measurement.
    pub extrapolated_60_db_seconds: f64,
    /// Coefficient of determination of the retained sample fit.
    pub r_squared: f64,
    /// First and last fitted sample times relative to capture sample zero.
    pub fit_window_ms: [f64; 2],
    /// Actual decay range of the retained samples, in decibels.
    pub observed_span_db: f64,
    /// Number of retained fit samples.
    pub samples: usize,
    /// Absent only when estimated noise is exactly zero; not proof of noiseless acquisition.
    pub minimum_estimated_margin_db: Option<f64>,
}

/// Common-reference and individually normalized energy tails for one octave.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedDecayBand {
    /// Nominal octave center in hertz; finite filter skirts are not brick walls.
    pub center_hz: f64,
    /// Common capture time axis in milliseconds.
    pub times_ms: Vec<f64>,
    /// Baseline tail relative to its initial integrated energy, display-floored.
    pub pre_db: Vec<f64>,
    /// Candidate tail relative to the same baseline energy, display-floored.
    pub post_db: Vec<f64>,
    /// Baseline tail normalized by its own initial energy.
    pub pre_normalized_db: Vec<f64>,
    /// A silent candidate has no normalization reference.
    pub post_normalized_db: Option<Vec<f64>>,
    /// Integral of analytic-envelope squared, in raw-sample-squared seconds.
    pub baseline_energy_reference: f64,
    /// Baseline analytic-envelope squared mean in the supported noise interior.
    pub pre_noise_power: f64,
    /// Candidate analytic-envelope squared mean in the supported noise interior.
    pub post_noise_power: f64,
    /// Analysis-filter half-support in milliseconds, not playback latency.
    pub filter_half_support_ms: f64,
    /// Baseline fit only when the declared noise and fit budgets hold.
    pub pre_fit: Option<DecayFit>,
    /// Candidate fit only when the declared noise and fit budgets hold.
    pub post_fit: Option<DecayFit>,
    /// Baseline fit refusal reason; traces can remain available.
    pub pre_fit_unavailable: Option<String>,
    /// Candidate fit refusal reason; traces can remain available.
    pub post_fit_unavailable: Option<String>,
}

/// Matched decay diagnostics; these do not participate in acoustic acceptance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedDecayView {
    /// Capture, graph, calibration declaration, and processing provenance.
    pub provenance: ViewProvenance,
    /// Caller-supplied settings identity, not an acquisition signature.
    pub settings_hash: String,
    /// Explicit operator declarations and fit budgets used for these traces.
    pub settings: MatchedDecaySettings,
    /// Versioned analysis method identifier.
    pub method: String,
    /// Rendering floor in decibels; never an estimated noise floor.
    pub display_floor_db: f64,
    /// Supported matched octave traces and fit dispositions.
    pub bands: Vec<MatchedDecayBand>,
    /// Unavailable nominal centers, in hertz, mapped to their reasons.
    pub unavailable_bands: BTreeMap<String, String>,
    /// Limits of inference from the finite-window observation.
    pub scope: String,
}

fn energy_tail(values: &[f64], rate: f64) -> Result<Vec<f64>, String> {
    let mut total = 0.0;
    let mut energy: Vec<_> = values
        .iter()
        .rev()
        .map(|value| {
            total += value * value / rate;
            total
        })
        .collect();
    if !total.is_finite() {
        return Err("octave energy integration overflow".into());
    }
    energy.reverse();
    Ok(energy)
}

fn db_trace(energy: &[f64], reference: f64) -> Vec<f64> {
    // A rendering floor, never a replacement for the unfloored fit data.
    energy
        .iter()
        .map(|value| (10.0 * (value.log10() - reference.log10())).max(-160.0))
        .collect()
}

fn fit(
    energy: &[f64],
    noise: f64,
    rate: f64,
    settings: &MatchedDecaySettings,
) -> Result<DecayFit, String> {
    if energy.is_empty() || energy[0] <= 0.0 {
        return Err("no nonzero normalization reference".into());
    }
    let normalized: Vec<_> = energy
        .iter()
        .map(|value| 10.0 * (value.log10() - energy[0].log10()))
        .collect();
    if !normalized.iter().any(|db| *db <= -25.0) {
        return Err("observation does not span the -5 to -25 dB interval".into());
    }
    let indices: Vec<_> = normalized
        .iter()
        .enumerate()
        .filter_map(|(i, db)| (-25.0..=-5.0).contains(db).then_some(i))
        .collect();
    if indices.len() < 3 {
        return Err("fewer than three samples in the -5 to -25 dB interval".into());
    }
    let mut minimum_margin = None::<f64>;
    for &index in &indices {
        let noise_energy = noise * (energy.len() - index) as f64 / rate;
        if noise_energy > 0.0 {
            let signal = energy[index] - noise_energy;
            if signal <= 0.0 {
                return Err("fit reaches the estimated integrated noise floor".into());
            }
            let margin = 10.0 * (signal.log10() - noise_energy.log10());
            if !margin.is_finite() || margin < settings.minimum_fit_margin_db {
                return Err("fit fails the declared noise-margin budget".into());
            }
            minimum_margin = Some(minimum_margin.map_or(margin, |old| old.min(margin)));
        }
    }
    let times: Vec<_> = (0..energy.len()).map(|i| i as f64 / rate).collect();
    let (seconds, r_squared) = fit_t60(&times, &normalized, -5.0, -25.0);
    if !seconds.is_finite()
        || seconds <= 0.0
        || !r_squared.is_finite()
        || r_squared < settings.minimum_r_squared
    {
        return Err("decay slope fails the declared fit-quality budget".into());
    }
    let first = indices[0];
    let last = indices[indices.len() - 1];
    Ok(DecayFit {
        extrapolated_60_db_seconds: seconds,
        r_squared,
        fit_window_ms: [times[first] * 1000.0, times[last] * 1000.0],
        observed_span_db: normalized[first] - normalized[last],
        samples: indices.len(),
        minimum_estimated_margin_db: minimum_margin,
    })
}

/// Compute finite-window octave tails and explicitly noise-gated slope fits.
///
/// Nominal supported octaves are selected from 63–4000 Hz. The integration
/// endpoint precedes the declared noise window by one filter half-support;
/// noise estimation uses only its interior with full measured filter support.
/// Both common-baseline energy and separate normalizations are retained.
///
/// # Errors
/// Rejects invalid settings, unsupported sizes/rates, nonfinite samples, and
/// numerical overflow. Individual unsupported bands and fits remain unavailable.
pub fn matched_capture_decay(
    pre: &[f64],
    post: &[f64],
    support: [f64; 2],
    mut provenance: ViewProvenance,
    settings_hash: String,
    settings: MatchedDecaySettings,
) -> Result<MatchedDecayView, String> {
    let rate = provenance.sample_rate_hz;
    let [noise_start, noise_end] = settings.noise_window_ms;
    if !rate.is_finite()
        || !(12_000.0..=384_000.0).contains(&rate)
        || pre.len() != post.len()
        || pre.is_empty()
        || pre.len() > 65_536
        || pre.iter().chain(post).any(|value| !value.is_finite())
        || support.iter().any(|value| !value.is_finite())
        || support[0] <= 0.0
        || support[1] <= support[0]
        || support[1] > rate / 2.0
        || !noise_start.is_finite()
        || !noise_end.is_finite()
        || noise_start <= 0.0
        || noise_end <= noise_start
        || noise_end > pre.len() as f64 * 1000.0 / rate
        || !settings.minimum_fit_margin_db.is_finite()
        || settings.minimum_fit_margin_db <= 0.0
        || !settings.minimum_r_squared.is_finite()
        || !(0.0..=1.0).contains(&settings.minimum_r_squared)
    {
        return Err("decay needs finite matched captures, supported bands, an explicit signal-free noise window, and finite fit budgets".into());
    }
    let start = (noise_start * rate / 1000.0).ceil() as usize;
    let end = (noise_end * rate / 1000.0).floor() as usize;
    provenance.processing_chain = "Hann analytic octave linear convolution; finite-window reverse energy integration; separately estimated declared noise-window power; no tail completion, subtraction, recentering or SPL conversion".into();
    let mut result = MatchedDecayView {
        provenance, settings_hash, settings, method: "finite_window_octave_schroeder_v1".into(),
        display_floor_db: -160.0, bands: Vec::new(), unavailable_bands: BTreeMap::new(),
        scope: "finite-window analytic-envelope energy; raw units, not calibrated acoustic energy; no noise subtraction or unobserved tail completion; separate normalizations do not erase the common-reference level change; T20 slope extrapolation is not passive-room RT; symmetric filter spreading and finite-window truncation affect estimates; signal-free noise region and stationary noise are operator assumptions, not authenticated facts; no safety or audibility verdict".into(),
    };
    for center in [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0] {
        let unavailable = if center / std::f64::consts::SQRT_2 < support[0]
            || center * std::f64::consts::SQRT_2 > support[1]
        {
            Some("nominal octave outside declared usable band")
        } else {
            None
        };
        let (taps, half) = kernel(rate, center);
        if let Some(reason) = unavailable.or_else(|| {
            (start <= half + 3 || end <= start + 2 * half + 3)
                .then_some("insufficient measured time support for noise-window filter margins")
        }) {
            result
                .unavailable_bands
                .insert(center.to_string(), reason.into());
            continue;
        }
        let count = start - half;
        let before = envelope(pre, &taps, half, pre.len() - half);
        let after = envelope(post, &taps, half, post.len() - half);
        if before.iter().chain(&after).any(|value| !value.is_finite()) {
            return Err("decay octave analysis overflow".into());
        }
        let noise_power = |values: &[f64]| {
            values[start + half..end - half]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                / (end - start - 2 * half) as f64
        };
        let pre_noise = noise_power(&before);
        let post_noise = noise_power(&after);
        if !pre_noise.is_finite() || !post_noise.is_finite() {
            return Err("decay noise estimate overflow".into());
        }
        let pre_energy = energy_tail(&before[..count], rate)?;
        let post_energy = energy_tail(&after[..count], rate)?;
        if pre_energy[0] <= 0.0 {
            result
                .unavailable_bands
                .insert(center.to_string(), "silent baseline band".into());
            continue;
        }
        let pre_fit = fit(&pre_energy, pre_noise, rate, &result.settings);
        let post_fit = fit(&post_energy, post_noise, rate, &result.settings);
        result.bands.push(MatchedDecayBand {
            center_hz: center,
            times_ms: (0..count).map(|i| i as f64 * 1000.0 / rate).collect(),
            pre_db: db_trace(&pre_energy, pre_energy[0]),
            post_db: db_trace(&post_energy, pre_energy[0]),
            pre_normalized_db: db_trace(&pre_energy, pre_energy[0]),
            post_normalized_db: (post_energy[0] > 0.0)
                .then(|| db_trace(&post_energy, post_energy[0])),
            baseline_energy_reference: pre_energy[0],
            pre_noise_power: pre_noise,
            post_noise_power: post_noise,
            filter_half_support_ms: half as f64 * 1000.0 / rate,
            pre_fit_unavailable: pre_fit.as_ref().err().cloned(),
            post_fit_unavailable: post_fit.as_ref().err().cloned(),
            pre_fit: pre_fit.ok(),
            post_fit: post_fit.ok(),
        });
    }
    if result.bands.is_empty() {
        return Err("no octave has sufficient band, time, and nonzero baseline support".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> MatchedDecaySettings {
        MatchedDecaySettings {
            noise_window_ms: [120.0, 200.0],
            minimum_fit_margin_db: 10.0,
            minimum_r_squared: 0.99,
        }
    }

    #[test]
    fn captured_decay_known_slope_and_noise_refusal() {
        let rate = 1000.0;
        let energy: Vec<_> = (0..500)
            .map(|i| 10.0_f64.powf(-6.0 * i as f64 / rate / 0.5))
            .collect();
        let fitted = fit(&energy, 0.0, rate, &settings()).unwrap();
        assert!((fitted.extrapolated_60_db_seconds - 0.5).abs() < 1e-12);
        assert!(fitted.r_squared > 0.999999);
        assert!(fitted.minimum_estimated_margin_db.is_none());
        assert!(
            fit(&energy, 100.0, rate, &settings())
                .unwrap_err()
                .contains("noise")
        );
        assert!(
            fit(&energy[..20], 0.0, rate, &settings())
                .unwrap_err()
                .contains("span")
        );
        let curved: Vec<_> = (0..100).map(|i| 1.0 / (1.0 + i as f64).powi(2)).collect();
        assert!(
            fit(&curved, 0.0, rate, &settings())
                .unwrap_err()
                .contains("fit-quality")
        );
        assert!(fit(&[0.0; 100], 0.0, rate, &settings()).is_err());
    }

    #[test]
    fn captured_decay_keeps_level_separate_from_shape_and_checks_support() {
        let rate = 12_000.0;
        let pre: Vec<_> = (0..2400)
            .map(|i| {
                let t = i as f64 / rate;
                if t < 0.09 {
                    (-3.0 * 10.0_f64.ln() * t / 0.04).exp()
                        * (std::f64::consts::TAU * 1000.0 * t).cos()
                } else {
                    0.0
                }
            })
            .collect();
        let post: Vec<_> = pre.iter().map(|value| value * 0.5).collect();
        let provenance = ViewProvenance {
            measurement_ids: vec!["synthetic-baseline".into(), "synthetic-candidate".into()],
            graph_identity: "graph".into(),
            sample_rate_hz: rate,
            calibration: "relative-raw-units".into(),
            processing_chain: "raw".into(),
        };
        let view = matched_capture_decay(
            &pre,
            &post,
            [50.0, 5900.0],
            provenance.clone(),
            "settings".into(),
            settings(),
        )
        .unwrap();
        assert!(view.unavailable_bands.contains_key("63"));
        for band in &view.bands {
            assert!((band.post_db[0] + 20.0 * 2.0_f64.log10()).abs() < 1e-10);
            assert!(
                band.pre_normalized_db
                    .iter()
                    .zip(band.post_normalized_db.as_ref().unwrap())
                    .all(|(a, b)| (a - b).abs() < 1e-9)
            );
            assert!(band.times_ms.last().unwrap() < &120.0);
        }
        let silent = matched_capture_decay(
            &pre,
            &vec![0.0; pre.len()],
            [50.0, 5900.0],
            provenance.clone(),
            "settings".into(),
            settings(),
        )
        .unwrap();
        assert!(
            silent
                .bands
                .iter()
                .all(|band| band.post_normalized_db.is_none() && band.post_fit.is_none())
        );
        let mut noisy = pre.clone();
        for (index, value) in noisy.iter_mut().enumerate().skip(1440) {
            *value = (std::f64::consts::TAU * 1000.0 * index as f64 / rate).cos();
        }
        let noise_limited = matched_capture_decay(
            &noisy,
            &noisy,
            [50.0, 5900.0],
            provenance.clone(),
            "settings".into(),
            settings(),
        )
        .unwrap();
        let band = noise_limited
            .bands
            .iter()
            .find(|band| band.center_hz == 1000.0)
            .unwrap();
        assert!(band.pre_fit.is_none());
        assert!(band.pre_fit_unavailable.as_ref().unwrap().contains("noise"));
        for invalid in [
            MatchedDecaySettings {
                noise_window_ms: [180.0, 220.0],
                ..settings()
            },
            MatchedDecaySettings {
                minimum_fit_margin_db: f64::NAN,
                ..settings()
            },
            MatchedDecaySettings {
                minimum_r_squared: 1.1,
                ..settings()
            },
        ] {
            assert!(
                matched_capture_decay(
                    &pre,
                    &post,
                    [50.0, 5900.0],
                    provenance.clone(),
                    "settings".into(),
                    invalid
                )
                .is_err()
            );
        }
    }
}
