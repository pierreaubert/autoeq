use crate::Curve;
use ndarray::Array1;
use std::collections::HashMap;

fn curve_is_usable(curve: &Curve) -> bool {
    roomeq_analysis::frequency_grid::is_valid_frequency_grid(&curve.freq)
        && curve.spl.len() == curve.freq.len()
        && curve.spl.iter().all(|level| level.is_finite())
}

fn same_frequency(first: f64, second: f64) -> bool {
    let scale = first.abs().max(second.abs()).max(1.0);
    (first - second).abs() <= 1.0e-6 * scale
}

/// Interpolate magnitude curves to one deterministic grid inside their shared
/// measured span. The sparsest channel grid is canonical so alignment never
/// invents a resolution finer than the least-resolved input measurement.
pub(super) fn align_curves_to_common_grid(
    curves: &HashMap<String, Curve>,
) -> Option<HashMap<String, Curve>> {
    if curves.is_empty() || curves.values().any(|curve| !curve_is_usable(curve)) {
        return None;
    }

    let (common_min, common_max) =
        roomeq_analysis::frequency_grid::common_frequency_range(curves.values())?;
    let mut candidates: Vec<(&str, usize, Vec<f64>)> = curves
        .iter()
        .map(|(name, curve)| {
            let frequencies = curve
                .freq
                .iter()
                .copied()
                .filter(|frequency| *frequency > common_min && *frequency < common_max)
                .collect();
            (name.as_str(), curve.freq.len(), frequencies)
        })
        .collect();
    candidates.sort_by(|(left_name, left_len, _), (right_name, right_len, _)| {
        left_len
            .cmp(right_len)
            .then_with(|| left_name.cmp(right_name))
    });

    let mut frequencies = candidates
        .into_iter()
        .find_map(|(_, _, frequencies)| (frequencies.len() + 2 >= 3).then_some(frequencies))?;
    if frequencies
        .first()
        .is_none_or(|frequency| !same_frequency(*frequency, common_min))
    {
        frequencies.insert(0, common_min);
    }
    if frequencies
        .last()
        .is_none_or(|frequency| !same_frequency(*frequency, common_max))
    {
        frequencies.push(common_max);
    }
    let common_grid = Array1::from_vec(frequencies);

    curves
        .iter()
        .map(|(name, curve)| {
            let mut magnitude_curve = curve.clone();
            magnitude_curve.phase = None;
            let mut aligned = autoeq_core::curve_transforms::interpolate_log_space(
                &common_grid,
                &magnitude_curve,
            );
            aligned.phase = None;
            Some((name.clone(), aligned))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(freq: &[f64], offset: f64) -> Curve {
        Curve {
            freq: Array1::from_vec(freq.to_vec()),
            spl: Array1::from_iter(freq.iter().map(|frequency| frequency.log10() + offset)),
            phase: None,
            ..Default::default()
        }
    }

    #[test]
    fn aligns_to_sparsest_grid_inside_common_measured_overlap() {
        let curves = HashMap::from([
            (
                "dense".to_string(),
                curve(&[20.0, 30.0, 50.0, 100.0, 200.0, 500.0], 0.0),
            ),
            (
                "sparse".to_string(),
                curve(&[30.0, 60.0, 120.0, 240.0, 400.0], 1.0),
            ),
        ]);

        let aligned = align_curves_to_common_grid(&curves).expect("shared overlap grid");

        assert_eq!(
            aligned["dense"].freq.to_vec(),
            vec![30.0, 60.0, 120.0, 240.0, 400.0]
        );
        assert!(roomeq_analysis::frequency_grid::same_frequency_grid(
            &aligned["dense"].freq,
            &aligned["sparse"].freq
        ));
        assert!(aligned.values().all(|curve| curve.phase.is_none()));
    }

    #[test]
    fn rejects_curves_without_a_shared_measured_span() {
        let curves = HashMap::from([
            ("low".to_string(), curve(&[20.0, 30.0, 40.0], 0.0)),
            ("high".to_string(), curve(&[50.0, 60.0, 70.0], 0.0)),
        ]);

        assert!(align_curves_to_common_grid(&curves).is_none());
    }
}
