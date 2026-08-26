use ndarray::Array1;

pub(super) fn is_local_extremum(
    freq: &Array1<f64>,
    spl: &Array1<f64>,
    idx: usize,
    half_width_octaves: f64,
    maximum: bool,
) -> bool {
    if freq.len() != spl.len() || idx >= spl.len() || freq[idx] <= 0.0 {
        return false;
    }
    let ratio = 2.0_f64.powf(half_width_octaves.max(0.0));
    let lower = freq[idx] / ratio;
    let upper = freq[idx] * ratio;
    let center = spl[idx];
    let mut neighbours: Vec<usize> = freq
        .iter()
        .enumerate()
        .filter_map(|(index, frequency)| {
            (index != idx && *frequency >= lower && *frequency <= upper).then_some(index)
        })
        .collect();
    if neighbours.is_empty() {
        if idx > 0 {
            neighbours.push(idx - 1);
        }
        if idx + 1 < spl.len() {
            neighbours.push(idx + 1);
        }
    }

    neighbours.into_iter().all(|j| {
        if maximum {
            center > spl[j]
        } else {
            center < spl[j]
        }
    })
}

pub(super) fn interpolate_fdw_to_grid(
    src_freq: &[f32],
    src_values: &[f32],
    target_freq: &Array1<f64>,
    fallback: f64,
) -> Array1<f64> {
    if src_freq.is_empty()
        || src_values.is_empty()
        || src_freq.len() != src_values.len()
        || src_freq
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || src_values.iter().any(|value| !value.is_finite())
        || src_freq.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Array1::from_elem(target_freq.len(), fallback);
    }

    let values: Vec<f64> = target_freq
        .iter()
        .map(|&target| {
            if !target.is_finite() || target <= 0.0 {
                return fallback;
            }
            if target <= src_freq[0] as f64 {
                return src_values[0] as f64;
            }
            let last = src_freq.len() - 1;
            if target >= src_freq[last] as f64 {
                return src_values[last] as f64;
            }

            let idx = match src_freq.binary_search_by(|f| (*f as f64).partial_cmp(&target).unwrap())
            {
                Ok(i) => return src_values[i] as f64,
                Err(i) => i,
            };

            let f0 = src_freq[idx - 1] as f64;
            let f1 = src_freq[idx] as f64;
            let denom = f1.ln() - f0.ln();
            if denom.abs() <= 1e-12 {
                return src_values[idx] as f64;
            }
            let t = ((target.ln() - f0.ln()) / denom).clamp(0.0, 1.0);
            src_values[idx - 1] as f64 + t * (src_values[idx] as f64 - src_values[idx - 1] as f64)
        })
        .collect();

    Array1::from_vec(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fdw_interpolation_fails_closed_for_nan_or_unsorted_source_frequencies() {
        let target = Array1::from_vec(vec![50.0, 100.0, 200.0]);
        for source in [
            vec![20.0, f32::NAN, 500.0],
            vec![20.0, 200.0, 100.0],
            vec![20.0, 20.0, 500.0],
        ] {
            let result = interpolate_fdw_to_grid(&source, &[0.2, 0.5, 0.8], &target, 0.4);
            assert_eq!(result.as_slice().unwrap(), &[0.4, 0.4, 0.4]);
        }
    }
}
