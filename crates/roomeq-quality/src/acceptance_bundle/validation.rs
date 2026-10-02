//! Structural and cross-view validation before evaluating bundle headroom.
//!
//! These checks establish internal consistency, not measurement authenticity,
//! acoustic acceptance, perceptual validity, or complete routed-output coverage.

use super::{AcceptanceBundle, ViewProvenance, ViewSettings};

// Numerical agreement only, expressed in samples or relative amplitude below.
// This is not an acoustic, timing-uncertainty, or audibility acceptance budget.
const NUMERIC_TOLERANCE: f64 = 1e-7;

fn ordered(values: &[f64], minimum_len: usize) -> bool {
    values.len() >= minimum_len
        && values.iter().all(|value| value.is_finite())
        && values.windows(2).all(|pair| pair[0] < pair[1])
}

fn traces(axis: &[f64], values: &[&[f64]], minimum_len: usize) -> bool {
    ordered(axis, minimum_len)
        && values.iter().all(|values| {
            values.len() == axis.len() && values.iter().all(|value| value.is_finite())
        })
}

fn band_valid(band: [f64; 2]) -> bool {
    band[0].is_finite() && band[1].is_finite() && band[0] > 0.0 && band[1] > band[0]
}

fn frequency_axis(axis: &[f64], band: [f64; 2], minimum_len: usize) -> bool {
    ordered(axis, minimum_len)
        && axis
            .iter()
            .all(|value| *value >= band[0] && *value <= band[1])
}

fn sampled_time_axis(axis: &[f64], sample_rate_hz: f64) -> bool {
    ordered(axis, 2)
        && sample_rate_hz.is_finite()
        && sample_rate_hz > 0.0
        && axis.windows(2).all(|pair| {
            ((pair[1] - pair[0]) * sample_rate_hz / 1000.0 - 1.0).abs() <= NUMERIC_TOLERANCE
        })
}

fn step_matches(ir: &[f64], step: &[f64]) -> bool {
    let mut total = 0.0;
    ir.len() == step.len()
        && ir.iter().zip(step).all(|(sample, actual)| {
            total += sample;
            total.is_finite()
                && actual.is_finite()
                && (total - actual).abs() <= NUMERIC_TOLERANCE * total.abs().max(1.0)
        })
}

fn provenance_failures(
    name: &str,
    provenance: &ViewProvenance,
    settings: &ViewSettings,
    graph: Option<&str>,
    failures: &mut Vec<String>,
) {
    let ids: std::collections::HashSet<_> = provenance.measurement_ids.iter().collect();
    if ids.is_empty()
        || ids.len() != provenance.measurement_ids.len()
        || ids.iter().any(|id| id.trim().is_empty())
    {
        failures.push(format!(
            "{name} lacks unique nonempty measurement identities"
        ));
    }
    if provenance.graph_identity.trim().is_empty()
        || graph.is_some_and(|expected| expected != provenance.graph_identity)
    {
        failures.push(format!("{name} graph provenance is missing or mismatched"));
    }
    if !provenance.sample_rate_hz.is_finite()
        || provenance.sample_rate_hz <= 0.0
        || provenance.sample_rate_hz != settings.sample_rate_hz
    {
        failures.push(format!(
            "{name} sample-rate provenance is missing or mismatched"
        ));
    }
    if provenance.processing_chain.trim().is_empty() || provenance.calibration.trim().is_empty() {
        failures.push(format!("{name} lacks processing/calibration provenance"));
    }
}

pub(super) fn validate_bundle_contents(bundle: &AcceptanceBundle) -> Vec<String> {
    let mut failures = Vec::new();
    let settings = &bundle.settings;
    if !settings.sample_rate_hz.is_finite()
        || settings.sample_rate_hz <= 0.0
        || !band_valid(settings.freq_limits_hz)
        || settings.freq_limits_hz[1] > settings.sample_rate_hz / 2.0
        || !settings.reference_level_db.is_finite()
        || !matches!(settings.general_smoothing_bands_per_octave, 6 | 12)
        || settings.window.trim().is_empty()
        || settings.normalization.trim().is_empty()
        || !ordered(&settings.etc_window_ms, 2)
        || !frequency_axis(
            &settings.etc_bands_hz,
            [f64::MIN_POSITIVE, settings.sample_rate_hz / 2.0],
            1,
        )
    {
        failures.push("invalid acceptance view settings".into());
    }
    let provenance = [
        (
            "magnitude",
            bundle.magnitude.as_ref().map(|view| &view.provenance),
        ),
        (
            "ir_step",
            bundle.ir_step.as_ref().map(|view| &view.provenance),
        ),
        ("etc", bundle.etc.as_ref().map(|view| &view.provenance)),
        ("decay", bundle.decay.as_ref().map(|view| &view.provenance)),
        (
            "ambient_noise",
            bundle.ambient_noise.as_ref().map(|view| &view.provenance),
        ),
        (
            "headroom",
            bundle.headroom.as_ref().map(|view| &view.provenance),
        ),
        (
            "disposition",
            bundle.disposition.as_ref().map(|view| &view.provenance),
        ),
    ];
    let graph = bundle
        .disposition
        .as_ref()
        .map(|view| view.graph_identity.as_str())
        .or_else(|| {
            provenance
                .iter()
                .find_map(|(_, value)| value.map(|value| value.graph_identity.as_str()))
        });
    for (name, value) in provenance {
        if let Some(value) = value {
            provenance_failures(name, value, settings, graph, &mut failures);
        }
    }
    if let Some(view) = &bundle.magnitude
        && (!traces(&view.freqs, &[&view.pre_db, &view.post_db], 2)
            || !frequency_axis(&view.freqs, settings.freq_limits_hz, 2))
    {
        failures.push(
            "magnitude traces are empty, nonfinite, misaligned, or outside their declared band"
                .into(),
        );
    }
    if let Some(view) = &bundle.ir_step
        && (!traces(
            &view.times_ms,
            &[&view.pre_ir, &view.post_ir, &view.pre_step, &view.post_step],
            2,
        ) || !sampled_time_axis(&view.times_ms, settings.sample_rate_hz)
            || view.common_reference.trim().is_empty()
            || !step_matches(&view.pre_ir, &view.pre_step)
            || !step_matches(&view.post_ir, &view.post_step))
    {
        failures.push("ir_step traces lack a valid common reference, sample grid, or consistent cumulative step".into());
    }
    if let Some(view) = &bundle.etc {
        let centers: Vec<f64> = view.bands.iter().map(|band| band.center_hz).collect();
        if centers != settings.etc_bands_hz
            || !view.lr_similarity_db.is_finite()
            || view.lr_similarity_db < 0.0
        {
            failures.push(
                "etc lacks the declared octave-band coverage or finite similarity metric".into(),
            );
        }
        for band in &view.bands {
            let window = settings.etc_window_ms;
            let endpoint_tolerance_ms = 1000.0 / settings.sample_rate_hz;
            if !traces(&band.times_ms, &[&band.pre_db, &band.post_db], 2)
                || !sampled_time_axis(&band.times_ms, settings.sample_rate_hz)
                || band
                    .times_ms
                    .first()
                    .is_none_or(|start| (*start - window[0]).abs() > endpoint_tolerance_ms)
                || band
                    .times_ms
                    .last()
                    .is_none_or(|end| (*end - window[1]).abs() > endpoint_tolerance_ms)
            {
                failures.push(format!(
                    "etc {} Hz trace does not cover its declared time window/sample grid",
                    band.center_hz
                ));
            }
        }
    }
    if let Some(view) = &bundle.decay {
        let fit_valid = |(seconds, quality): (f64, f64)| {
            seconds.is_finite()
                && seconds > 0.0
                && quality.is_finite()
                && (0.0..=1.0).contains(&quality)
        };
        if !band_valid(view.valid_range_hz)
            || view.valid_range_hz[0] < settings.freq_limits_hz[0]
            || view.valid_range_hz[1] > settings.freq_limits_hz[1]
            || !traces(
                &view.center_freqs,
                &[&view.absolute_tail_db, &view.normalized_tail_db],
                1,
            )
            || !frequency_axis(&view.center_freqs, view.valid_range_hz, 1)
            || !fit_valid(view.t20_s)
            || !fit_valid(view.t30_s)
            || !view.playback_tail_change_db.is_finite()
            || !view.passive_room_rt_s.is_finite()
            || view.passive_room_rt_s <= 0.0
        {
            failures.push(
                "decay view lacks finite, aligned, physically defined traces/fit fields".into(),
            );
        }
    }
    if let Some(view) = &bundle.ambient_noise
        && (!traces(&view.freqs, &[&view.noise_spl_db], 2)
            || !frequency_axis(&view.freqs, settings.freq_limits_hz, 2)
            || view.calibration.trim().is_empty()
            || view.calibration.trim().eq_ignore_ascii_case("uncalibrated")
            || view.calibration != view.provenance.calibration)
    {
        failures.push("ambient_noise trace or calibration provenance is invalid".into());
    }
    if let Some(view) = &bundle.headroom {
        let names: std::collections::HashSet<_> = view
            .outputs
            .iter()
            .map(|output| &output.output_name)
            .collect();
        let required: std::collections::HashSet<_> = bundle.required_output_ids.iter().collect();
        if required.is_empty()
            || required.len() != bundle.required_output_ids.len()
            || required.iter().any(|name| name.trim().is_empty())
            || required != names
        {
            failures.push(
                "headroom does not exactly cover a declared unique physical-output plan".into(),
            );
        }
        if names.len() != view.outputs.len() || names.iter().any(|name| name.trim().is_empty()) {
            failures.push("headroom output identities are empty or duplicated".into());
        }
    }
    if let Some(view) = &bundle.disposition
        && (view.channels.is_empty()
            || view
                .channels
                .windows(2)
                .any(|pair| pair[0].channel >= pair[1].channel)
            || view.channels.iter().any(|channel| {
                channel.channel.trim().is_empty()
                    || !channel.gain_db.is_finite()
                    || !channel.delay_ms.is_finite()
                    || channel.delay_ms < 0.0
                    || channel.drivers.iter().any(|driver| {
                        driver.name.trim().is_empty()
                            || !driver.gain_db.is_finite()
                            || !driver.delay_ms.is_finite()
                            || driver.delay_ms < 0.0
                    })
            }))
    {
        failures.push("disposition lacks unique channels or finite causal stage facts".into());
    }
    failures
}
