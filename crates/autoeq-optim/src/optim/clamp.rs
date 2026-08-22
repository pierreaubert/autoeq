use super::misc::interpolate_boost_envelope;
use crate::PeqModel;

/// Clamp positive filter gains in the parameter vector using a frequency-dependent envelope.
///
/// For each filter, if its gain is positive (boost), clamp it to the envelope's
/// max boost at that filter's center frequency. Returns a new owned vector.
pub fn clamp_gains_to_envelope(
    x: &[f64],
    envelope: &[(f64, f64)],
    peq_model: PeqModel,
) -> Vec<f64> {
    use crate::param_utils;
    let mut clamped = x.to_vec();
    let num_filters = param_utils::num_filters(x, peq_model);
    for i in 0..num_filters {
        let params = param_utils::get_filter_params(x, i, peq_model);
        let freq_hz = 10f64.powf(params.freq);
        if params.gain > 0.0 {
            let max_boost = interpolate_boost_envelope(envelope, freq_hz);
            if params.gain > max_boost {
                let ppf = param_utils::params_per_filter(peq_model);
                // gain is the last parameter in each filter's group
                let gain_idx = i * ppf + (ppf - 1);
                clamped[gain_idx] = max_boost;
            }
        }
    }
    clamped
}

/// Clamp negative filter gains (cuts) to a frequency-dependent minimum.
///
/// Mirrors `clamp_gains_to_envelope` but for cuts: if a filter's gain is negative
/// and exceeds the envelope's limit (more negative), it is clamped.
/// Used for CDT protection — prevents over-cutting at frequencies where
/// the ear generates distortion tones.
pub fn clamp_cuts_to_envelope(x: &[f64], envelope: &[(f64, f64)], peq_model: PeqModel) -> Vec<f64> {
    use crate::param_utils;
    let mut clamped = x.to_vec();
    let num_filters = param_utils::num_filters(x, peq_model);
    for i in 0..num_filters {
        let params = param_utils::get_filter_params(x, i, peq_model);
        let freq_hz = 10f64.powf(params.freq);
        if params.gain < 0.0 {
            let max_cut = interpolate_boost_envelope(envelope, freq_hz); // returns negative dB
            if params.gain < max_cut {
                let ppf = param_utils::params_per_filter(peq_model);
                let gain_idx = i * ppf + (ppf - 1);
                clamped[gain_idx] = max_cut;
            }
        }
    }
    clamped
}

/// Copy parameters once and apply both optional gain envelopes in place.
///
/// The caller owns `output`, so repeated candidate evaluations reuse its
/// allocation. Returns the slice that should be evaluated.
pub fn clamp_envelopes_into<'a>(
    x: &'a [f64],
    output: &'a mut Vec<f64>,
    max_boost: Option<&[(f64, f64)]>,
    min_cut: Option<&[(f64, f64)]>,
    peq_model: PeqModel,
) -> &'a [f64] {
    if max_boost.is_none() && min_cut.is_none() {
        return x;
    }
    output.clear();
    output.extend_from_slice(x);
    let num_filters = crate::param_utils::num_filters(x, peq_model);
    let parameters_per_filter = crate::param_utils::params_per_filter(peq_model);
    for index in 0..num_filters {
        let params = crate::param_utils::get_filter_params(x, index, peq_model);
        let frequency = 10f64.powf(params.freq);
        let gain_index = index * parameters_per_filter + parameters_per_filter - 1;
        if params.gain > 0.0
            && let Some(envelope) = max_boost
        {
            output[gain_index] = params
                .gain
                .min(interpolate_boost_envelope(envelope, frequency));
        } else if params.gain < 0.0
            && let Some(envelope) = min_cut
        {
            output[gain_index] = params
                .gain
                .max(interpolate_boost_envelope(envelope, frequency));
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_gains_to_envelope_clamps_boosts_above_limit() {
        // One PK filter: freq=1000 Hz (log10=3.0), Q=1.0, gain=+10 dB
        let x = vec![3.0, 1.0, 10.0];
        // Envelope allows max 6 dB at 1000 Hz
        let envelope = vec![(100.0, 12.0), (1000.0, 6.0), (10000.0, 3.0)];
        let clamped = clamp_gains_to_envelope(&x, &envelope, PeqModel::Pk);
        assert!(
            (clamped[2] - 6.0).abs() < 1e-12,
            "boost should be clamped to envelope"
        );
    }

    #[test]
    fn clamp_gains_to_envelope_leaves_boosts_below_limit_unchanged() {
        let x = vec![3.0, 1.0, 4.0];
        let envelope = vec![(100.0, 12.0), (1000.0, 6.0), (10000.0, 3.0)];
        let clamped = clamp_gains_to_envelope(&x, &envelope, PeqModel::Pk);
        assert!(
            (clamped[2] - 4.0).abs() < 1e-12,
            "boost below envelope should stay unchanged"
        );
    }

    #[test]
    fn clamp_gains_to_envelope_leaves_cuts_unchanged() {
        let x = vec![3.0, 1.0, -8.0];
        let envelope = vec![(100.0, 12.0), (1000.0, 6.0), (10000.0, 3.0)];
        let clamped = clamp_gains_to_envelope(&x, &envelope, PeqModel::Pk);
        assert!(
            (clamped[2] - (-8.0)).abs() < 1e-12,
            "cuts should not be affected by gain clamping"
        );
    }

    #[test]
    fn clamp_cuts_to_envelope_clamps_cuts_below_limit() {
        // One PK filter: freq=1000 Hz, Q=1.0, gain=-10 dB
        let x = vec![3.0, 1.0, -10.0];
        // Envelope allows max cut of -6 dB at 1000 Hz (negative value)
        let envelope = vec![(100.0, -12.0), (1000.0, -6.0), (10000.0, -3.0)];
        let clamped = clamp_cuts_to_envelope(&x, &envelope, PeqModel::Pk);
        assert!(
            (clamped[2] - (-6.0)).abs() < 1e-12,
            "cut should be clamped to envelope"
        );
    }

    #[test]
    fn clamp_cuts_to_envelope_leaves_cuts_above_limit_unchanged() {
        let x = vec![3.0, 1.0, -4.0];
        let envelope = vec![(100.0, -12.0), (1000.0, -6.0), (10000.0, -3.0)];
        let clamped = clamp_cuts_to_envelope(&x, &envelope, PeqModel::Pk);
        assert!(
            (clamped[2] - (-4.0)).abs() < 1e-12,
            "cut above envelope limit should stay unchanged"
        );
    }

    #[test]
    fn clamp_cuts_to_envelope_leaves_boosts_unchanged() {
        let x = vec![3.0, 1.0, 8.0];
        let envelope = vec![(100.0, -12.0), (1000.0, -6.0), (10000.0, -3.0)];
        let clamped = clamp_cuts_to_envelope(&x, &envelope, PeqModel::Pk);
        assert!(
            (clamped[2] - 8.0).abs() < 1e-12,
            "boosts should not be affected by cut clamping"
        );
    }

    #[test]
    fn clamp_gains_to_envelope_empty_envelope_allows_anything() {
        let x = vec![3.0, 1.0, 20.0];
        let clamped = clamp_gains_to_envelope(&x, &[], PeqModel::Pk);
        assert!((clamped[2] - 20.0).abs() < 1e-12);
    }
}
