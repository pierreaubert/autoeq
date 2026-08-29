use super::{compute_flat_loss, predict_bass_management_sum};
use crate::Curve;

pub fn bass_management_objective(curve: Option<&Curve>, xover_freq: f64) -> Option<f64> {
    let curve = curve?;
    // Use a symmetric band around the crossover in log-frequency space.
    // When a cap (20 Hz low or 2 kHz high) is hit, adjust the other side
    // to maintain equal octave span on both sides.
    let mut min_freq = xover_freq / 2.0;
    let mut max_freq = xover_freq * 2.0;
    if min_freq < 20.0 || max_freq > 2000.0 {
        let ratio = if min_freq < 20.0 {
            xover_freq / 20.0
        } else {
            2000.0 / xover_freq
        };
        min_freq = (xover_freq / ratio).max(20.0);
        max_freq = (xover_freq * ratio).min(2000.0);
    }
    max_freq = max_freq.max(min_freq + 1.0);
    Some(compute_flat_loss(curve, min_freq, max_freq))
}

/// Score the routed response against the configured target shape around the
/// crossover. A target-independent flatness score biases redirected bass low
/// whenever the requested house curve rises toward low frequencies.
pub fn bass_management_objective_with_target(
    curve: Option<&Curve>,
    target: Option<&Curve>,
    xover_freq: f64,
) -> Option<f64> {
    let Some(target) = target else {
        return bass_management_objective(curve, xover_freq);
    };
    let curve = curve?;
    let target = autoeq_core::curve_transforms::interpolate_log_space(&curve.freq, target);
    let mut error = curve.clone();
    for (level, target_level) in error.spl.iter_mut().zip(target.spl.iter()) {
        *level -= *target_level;
    }
    error.phase = None;
    bass_management_objective(Some(&error), xover_freq)
}

pub fn bass_management_crossover_type_candidates(requested: &str) -> Vec<String> {
    let requested = requested.trim();
    if requested.eq_ignore_ascii_case("auto") || requested.eq_ignore_ascii_case("optimize") {
        vec![
            "LR24".to_string(),
            "LR48".to_string(),
            "BW12".to_string(),
            "BW24".to_string(),
        ]
    } else {
        vec![requested.to_string()]
    }
}

pub fn select_bass_management_crossover_type(
    requested: &str,
    main_curve: &Curve,
    sub_curve: &Curve,
    xover_freq: f64,
    sample_rate: f64,
) -> String {
    let candidates = bass_management_crossover_type_candidates(requested);
    if candidates.len() == 1 {
        return candidates[0].clone();
    }

    candidates
        .iter()
        .filter(|candidate| {
            candidate
                .parse::<autoeq_optim::loss::CrossoverType>()
                .is_ok()
        })
        .filter_map(|candidate| {
            let predicted = predict_bass_management_sum(
                main_curve,
                sub_curve,
                candidate,
                xover_freq,
                sample_rate,
                0.0,
                0.0,
                0.0,
                0.0,
                false,
            );
            bass_management_objective(predicted.as_ref(), xover_freq)
                .map(|objective| (candidate.clone(), objective))
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(candidate, _)| candidate)
        .unwrap_or_else(|| "LR24".to_string())
}
