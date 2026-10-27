//! Q-controlled bandwidth projection shared by scoring and final emission.

use super::{ObjectiveData, QAdjustment};
use autoeq_core::param_utils::PeqLayout;
use autoeq_core::resolution::MeasurementResolution;
use std::borrow::Cow;
use std::sync::Arc;

/// Measurement support and limits for Q-controlled biquad bandwidth.
#[derive(Debug, Clone)]
pub struct PeakResolutionGuard {
    /// Original grids and smoothing widths retained through conditioning.
    pub support: Arc<MeasurementResolution>,
    /// Configured minimum Q; projection must never cross this lower bound.
    pub minimum_q: f64,
    /// Configured global Q ceiling, also used by model-specific box bounds.
    pub maximum_q: f64,
    /// Additional objective smoothing width, in octaves (`0` disables).
    pub smoothing_octaves: f64,
}

fn local_cap(
    data: &ObjectiveData,
    frequency: f64,
    kind: crate::iir::BiquadFilterType,
) -> Result<Option<(f64, f64)>, String> {
    let mut result = None;
    if let Some(guard) = &data.peak_resolution_guard {
        if !guard.minimum_q.is_finite() || guard.minimum_q <= 0.0 {
            return Err(String::from(
                "measurement resolution needs a finite positive minimum Q",
            ));
        }
        if !guard.maximum_q.is_finite()
            || guard.maximum_q <= 0.0
            || guard.maximum_q < guard.minimum_q
        {
            return Err(String::from(
                "measurement resolution needs a valid configured Q ceiling",
            ));
        }
        let cap = guard
            .support
            .max_peak_q(frequency, data.srate, guard.smoothing_octaves)
            .ok_or_else(|| format!("no measurement resolution warrant at {frequency:.3} Hz"))?;
        // Match setup_bounds' true lower endpoint, including the special
        // resonant HP/LP model box. Scoring must not project below a box that
        // final emission would reject. Free topologies use the general box.
        let minimum = if matches!(
            data.peq_model,
            crate::PeqModel::HpPk | crate::PeqModel::HpPkLp
        ) && matches!(
            kind,
            crate::iir::BiquadFilterType::HighpassVariableQ | crate::iir::BiquadFilterType::Lowpass
        ) {
            1.0_f64.max(guard.minimum_q.max(0.1)).min(guard.maximum_q)
        } else {
            guard.minimum_q.max(0.1)
        };
        result = Some((cap.min(guard.maximum_q), minimum));
    }
    if let Some(multi) = &data.multi_objective {
        for objective in &multi.objectives {
            if let Some((cap, minimum)) = local_cap(objective, frequency, kind)? {
                result = Some(match result {
                    Some((previous_cap, previous_minimum)) => {
                        (cap.min(previous_cap), minimum.max(previous_minimum))
                    }
                    None => (cap, minimum),
                });
            }
        }
    }
    Ok(result)
}

fn matched_requirements(data: &ObjectiveData, frequency: f64) -> Option<(f64, f64, f64)> {
    let mut result = None;
    if let Some(guard) = &data.peak_resolution_guard {
        if !guard.minimum_q.is_finite()
            || !guard.maximum_q.is_finite()
            || guard.minimum_q <= 0.0
            || guard.maximum_q < guard.minimum_q
            || !guard.smoothing_octaves.is_finite()
            || guard.smoothing_octaves < 0.0
        {
            return None;
        }
        let width = guard
            .support
            .width_octaves(frequency)?
            .max(guard.smoothing_octaves);
        result = Some((width, guard.minimum_q.max(0.1), guard.maximum_q));
    }
    if let Some(multi) = &data.multi_objective {
        for objective in &multi.objectives {
            if has_guard(objective) {
                let next = matched_requirements(objective, frequency)?;
                result = Some(match result {
                    Some((width, minimum, maximum)) => {
                        (width.max(next.0), minimum.max(next.1), maximum.min(next.2))
                    }
                    None => next,
                });
            }
        }
    }
    result
}

fn has_guard(data: &ObjectiveData) -> bool {
    data.peak_resolution_guard.is_some()
        || data
            .multi_objective
            .as_ref()
            .is_some_and(|multi| multi.objectives.iter().any(has_guard))
}

/// Project Q-controlled filters without allocating when parameters already comply.
pub fn guarded_parameters<'a>(
    x: &'a [f64],
    data: &ObjectiveData,
    mut adjusted: impl FnMut(QAdjustment),
) -> Result<Cow<'a, [f64]>, String> {
    if !super::is_peq_layout_loss(data.loss_type) || !has_guard(data) {
        return Ok(Cow::Borrowed(x));
    }
    let width = data.peq_model.params_per_filter();
    if width == 0 || !x.len().is_multiple_of(width) || x.iter().any(|value| !value.is_finite()) {
        return Err(String::from(
            "measurement resolution received invalid peak parameters",
        ));
    }
    let count = data.peq_model.num_filters(x);
    let q_index = data.peq_model.layout().q_idx;
    let mut projected = Cow::Borrowed(x);
    for index in 0..count {
        let filter = data.peq_model.get_filter_params(x, index);
        if filter
            .filter_type
            .is_some_and(|value| !(0.0..12.0).contains(&value))
        {
            return Err(String::from(
                "measurement resolution received an invalid filter type",
            ));
        }
        use crate::iir::BiquadFilterType;
        let kind = data
            .peq_model
            .determine_filter_type(index, count, filter.filter_type);
        if kind == BiquadFilterType::PeakMatched {
            if filter.q <= 0.0 {
                return Err(String::from(
                    "measurement resolution received nonpositive matched-peak Q",
                ));
            }
            if filter.gain == 0.0 {
                continue;
            }
            let frequency = 10.0_f64.powf(filter.freq);
            let (_, minimum, maximum) = matched_requirements(data, frequency).ok_or_else(|| {
                format!("no measurement resolution warrant for matched peak at {frequency:.3} Hz")
            })?;
            let q = super::matched_resolution::project_q(
                frequency, data.srate, filter.q, filter.gain, minimum, maximum,
                |actual_center| matched_requirements(data, actual_center).map(|value| value.0),
            ).ok_or_else(|| format!("measurement resolution cannot qualify matched-peak shape/bandwidth at {frequency:.3} Hz within configured Q bounds"))?;
            if q != filter.q {
                projected.to_mut()[index * width + q_index] = q;
                adjusted(QAdjustment {
                    filter_index: index,
                    center_hz: frequency,
                    q_before: filter.q,
                    q_after: q,
                    global_cap: maximum,
                    local_cap: q,
                });
            }
            continue;
        }
        // Standard RBJ pole damping uses alpha=sin(omega)/(2Q) for every
        // listed topology. Gain is unused by pass/notch/all-pass filters, so
        // zero gain does not make them neutral. Fixed-slope shelves ignore Q.
        if !matches!(
            kind,
            BiquadFilterType::Peak
                | BiquadFilterType::Lowpass
                | BiquadFilterType::Highpass
                | BiquadFilterType::HighpassVariableQ
                | BiquadFilterType::Bandpass
                | BiquadFilterType::Notch
                | BiquadFilterType::AllPass
        ) {
            continue;
        }
        if filter.q <= 0.0 {
            return Err(String::from(
                "measurement resolution received nonpositive filter Q",
            ));
        }
        if kind == BiquadFilterType::Peak && filter.gain == 0.0 {
            continue;
        }
        let frequency = 10.0_f64.powf(filter.freq);
        let Some((cap, minimum)) = local_cap(data, frequency, kind)? else {
            continue;
        };
        if cap < minimum || filter.q < minimum {
            return Err(format!(
                "measurement resolution at {frequency:.3} Hz permits Q <= {cap:.6}, incompatible with minimum Q {minimum:.6}"
            ));
        }
        if filter.q > cap {
            projected.to_mut()[index * width + q_index] = cap;
            adjusted(QAdjustment {
                filter_index: index,
                center_hz: frequency,
                q_before: filter.q,
                q_after: cap,
                global_cap: data
                    .peak_resolution_guard
                    .as_ref()
                    .map_or(f64::INFINITY, |guard| guard.maximum_q),
                local_cap: cap,
            });
        }
    }
    Ok(projected)
}

pub fn guard_identities(data: &ObjectiveData) -> Vec<String> {
    use sha2::{Digest, Sha256};
    let mut identities = Vec::new();
    if let Some(guard) = &data.peak_resolution_guard {
        let digest = Sha256::digest(format!("{:?}", guard.support).as_bytes());
        let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        identities.push(format!(
            "measurement_biquad_bandwidth_v3:min_q={}:max_q={}:smoothing_octaves={}:sample_rate={}:support_sha256={digest}:families=rbj_q_controlled+matched_actual_half_gain:matched_nominal_tolerance_db=0.00001",
            guard.minimum_q, guard.maximum_q, guard.smoothing_octaves, data.srate,
        ));
    }
    if let Some(multi) = &data.multi_objective {
        for objective in &multi.objectives {
            identities.extend(guard_identities(objective));
        }
    }
    identities
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PeqModel;
    use crate::optim::ObjectiveDataBuilder;
    use ndarray::Array1;

    fn objective() -> ObjectiveData {
        let grid = Array1::linspace(20.0, 160.0, 512);
        let mut data = ObjectiveDataBuilder::speaker_flat(
            grid.clone(),
            Array1::zeros(512),
            Array1::from_elem(512, -3.0),
            48000.0,
            PeqModel::Pk,
        )
        .freq_range(20.0, 160.0)
        .smoothing(false, 0)
        .build()
        .unwrap();
        data.peak_resolution_guard = Some(PeakResolutionGuard {
            support: Arc::new(
                MeasurementResolution::from_grid(&[20.0, 40.0, 80.0, 160.0]).unwrap(),
            ),
            minimum_q: 0.1,
            maximum_q: 6.0,
            smoothing_octaves: 0.0,
        });
        data
    }

    #[test]
    fn matched_peak_refuses_unrealizable_numerator_and_unsupported_width() {
        // Deep matched cuts are realized as the reciprocal of the matching
        // boost (math-iir-fir minimum-phase design), so the 20 kHz, -12 dB
        // case that the old clipped numerator could not realize now has the
        // exact boost feature. Qualification must agree for both signs.
        let cut = super::super::matched_resolution::feature(20_000.0, 48_000.0, 0.5, -12.0)
            .expect("reciprocal matched cut must be realizable");
        let boost_feature =
            super::super::matched_resolution::feature(20_000.0, 48_000.0, 0.5, 12.0).unwrap();
        assert!((cut.center_hz - boost_feature.center_hz).abs() < 1e-6);
        assert!((cut.width_octaves - boost_feature.width_octaves).abs() < 1e-9);
        // Genuinely unrealizable requests still refuse: no feature exists
        // at or above Nyquist.
        assert!(
            super::super::matched_resolution::feature(24_000.0, 48_000.0, 0.5, -12.0).is_none()
        );
        let mut data = objective();
        data.peq_model = PeqModel::Free;
        let boost = [11.0, 60.0_f64.log10(), 6.0, 3.0];
        let widened = guarded_parameters(&boost, &data, |_| {}).unwrap();
        let feature =
            super::super::matched_resolution::feature(60.0, 48_000.0, widened[2], 3.0).unwrap();
        assert!(
            feature.width_octaves >= 2.0,
            "a supported broad boost must remain available"
        );
        // The reciprocal cut widens to the same supported width at the same Q.
        let x = [11.0, 60.0_f64.log10(), 6.0, -3.0];
        let widened_cut = guarded_parameters(&x, &data, |_| {}).unwrap();
        assert!(
            (widened_cut[2] - widened[2]).abs() <= 1e-9 * widened[2].max(1.0),
            "matched cut must widen like its reciprocal boost: {} vs {}",
            widened_cut[2],
            widened[2]
        );
        let cut_feature =
            super::super::matched_resolution::feature(60.0, 48_000.0, widened_cut[2], -3.0)
                .unwrap();
        assert!(cut_feature.width_octaves >= 2.0);
        data.peak_resolution_guard.as_mut().unwrap().minimum_q = 2.0;
        assert!(
            guarded_parameters(&x, &data, |_| {})
                .unwrap_err()
                .contains("cannot qualify matched-peak")
        );
        let neutral = [11.0, 60.0_f64.log10(), 2.0, 0.0];
        assert!(matches!(
            guarded_parameters(&neutral, &data, |_| {}).unwrap(),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn constrained_highpass_model_refuses_width_below_its_true_q_box() {
        let mut data = objective();
        data.peq_model = PeqModel::HpPk;
        let x = [30.0_f64.log10(), 1.2, 0.0];
        let refusal = guarded_parameters(&x, &data, |_| {}).unwrap_err();
        assert!(
            refusal.contains("incompatible with minimum Q 1.000000"),
            "{refusal}"
        );
        // Explicitly lowering the global rail lowers this model's special
        // box too; do not invent a fixed minimum of one in that case.
        data.peak_resolution_guard.as_mut().unwrap().maximum_q = 0.6;
        let allowed = [30.0_f64.log10(), 0.6, 0.0];
        assert!(matches!(
            guarded_parameters(&allowed, &data, |_| {}).unwrap(),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn fixed_slope_shelves_do_not_acquire_q_dependent_resonances() {
        use crate::iir::{Biquad, BiquadFilterType};
        let mut data = objective();
        data.peq_model = PeqModel::Free;
        for kind in [
            BiquadFilterType::Lowshelf,
            BiquadFilterType::Highshelf,
            BiquadFilterType::LowshelfOrf,
            BiquadFilterType::HighshelfOrf,
        ] {
            let x = [
                crate::param_utils::encode_filter_type(kind),
                60.0_f64.log10(),
                6.0,
                -3.0,
            ];
            assert!(matches!(
                guarded_parameters(&x, &data, |_| {}).unwrap(),
                Cow::Borrowed(_)
            ));
            let broad = Biquad::new(kind, 60.0, 48_000.0, 0.1, -3.0);
            let high_q = Biquad::new(kind, 60.0, 48_000.0, 6.0, -3.0);
            for frequency in [20.0, 40.0, 60.0, 100.0, 160.0] {
                assert_eq!(
                    broad.complex_response(frequency),
                    high_q.complex_response(frequency)
                );
            }
        }
    }
}
