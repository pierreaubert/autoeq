use super::misc::pressure2spl;
use super::misc::spl2pressure;
use super::misc::spl2pressure2;
use ndarray::s;
use ndarray::{Array1, Array2, Axis};

/// Compute the CEA2034 spinorama from SPL data (internal implementation)
///
/// # Arguments
/// * `spl` - 2D array of SPL measurements
/// * `idx` - Indices for grouping measurements
/// * `weights` - Weights for computing weighted averages
///
/// # Returns
/// * 2D array representing the CEA2034 spinorama
///
/// # Details
/// Computes various CEA2034 curves including On Axis, Listening Window,
/// Early Reflections, Sound Power, and Predicted In-Room response.
pub(super) fn cea2034_array(
    spl: &Array2<f64>,
    idx: &[Vec<usize>],
    weights: &Array1<f64>,
) -> Array2<f64> {
    let len_spl = spl.shape()[1];
    let p2 = spl2pressure2(spl);
    let idx_sp = idx.len() - 1;
    let idx_lw = 0usize;
    let idx_er = 1usize;
    let idx_pir = idx_sp + 1;

    let mut cea = Array2::<f64>::zeros((idx.len() + 1, len_spl));

    for (i, idx_val) in idx.iter().enumerate().take(idx_sp) {
        let curve = apply_rms(&p2, idx_val);
        cea.row_mut(i).assign(&curve);
    }

    // ER: indices 2..=6 per original logic - vectorized
    let er_rows = cea.slice(s![2..=6, ..]);
    let er_pressures = er_rows.mapv(|v| {
        let p = 10f64.powf((v - 105.0) / 20.0);
        p * p
    });
    let er_p2_sum = er_pressures.sum_axis(Axis(0));
    let er_p = er_p2_sum.mapv(|v| (v / 5.0).sqrt());
    let er_spl = pressure2spl(&er_p);
    cea.row_mut(idx_er).assign(&er_spl);

    // SP weighted
    let sp_curve = apply_weighted_rms(&p2, &idx[idx_sp], weights);
    cea.row_mut(idx_sp).assign(&sp_curve);

    // PIR - vectorized computation
    let lw_p = spl2pressure(&cea.row(idx_lw).to_owned());
    let er_p = spl2pressure(&cea.row(idx_er).to_owned());
    let sp_p = spl2pressure(&cea.row(idx_sp).to_owned());

    let lw2 = lw_p.mapv(|v| v * v);
    let er2 = er_p.mapv(|v| v * v);
    let sp2 = sp_p.mapv(|v| v * v);

    let pir = (lw2.mapv(|v| 0.12 * v) + er2.mapv(|v| 0.44 * v) + sp2.mapv(|v| 0.44 * v))
        .mapv(|v| v.sqrt());
    let pir_spl = pressure2spl(&pir);
    cea.row_mut(idx_pir).assign(&pir_spl);

    cea
}

/// Apply RMS averaging to pressure squared values
///
/// # Arguments
/// * `p2` - 2D array of squared pressure values
/// * `idx` - Indices of rows to include in RMS calculation
///
/// # Returns
/// * Array of SPL values after RMS averaging
///
/// # Formula
/// rms = sqrt(sum(p2\[idx\]) / len) then converted to SPL
pub(super) fn apply_rms(p2: &Array2<f64>, idx: &[usize]) -> Array1<f64> {
    // Vectorized sum using select and sum_axis
    let selected_rows = p2.select(Axis(0), idx);
    let sum_rows = selected_rows.sum_axis(Axis(0));
    let len_idx = idx.len() as f64;
    let r = sum_rows.mapv(|v| (v / len_idx).sqrt());
    pressure2spl(&r)
}

/// Apply weighted RMS averaging to pressure squared values
///
/// # Arguments
/// * `p2` - 2D array of squared pressure values
/// * `idx` - Indices of rows to include in weighted RMS calculation
/// * `weights` - Weights for each row
///
/// # Returns
/// * Array of SPL values after weighted RMS averaging
///
/// # Formula
/// weighted_rms = sqrt(sum(p2\[idx\] * weights\[idx\]) / sum(weights)) then converted to SPL
pub(super) fn apply_weighted_rms(
    p2: &Array2<f64>,
    idx: &[usize],
    weights: &Array1<f64>,
) -> Array1<f64> {
    // Vectorized weighted sum
    let selected_rows = p2.select(Axis(0), idx);
    let selected_weights = weights.select(Axis(0), idx);
    let sum_w = selected_weights.sum();

    // Broadcast weights to match row dimensions and compute weighted sum
    let weighted_rows = &selected_rows * &selected_weights.insert_axis(Axis(1));
    let acc = weighted_rows.sum_axis(Axis(0));
    let r = acc.mapv(|v| (v / sum_w).sqrt());
    pressure2spl(&r)
}

/// Compute the CEA2034 spinorama from SPL data
///
/// # Arguments
/// * `spl` - 2D array of SPL measurements
/// * `idx` - Indices for grouping measurements
/// * `weights` - Weights for computing weighted averages
///
/// # Returns
/// * 2D array representing the CEA2034 spinorama
pub fn cea2034(spl: &Array2<f64>, idx: &[Vec<usize>], weights: &Array1<f64>) -> Array2<f64> {
    cea2034_array(spl, idx, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn sample_spl(rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(_, c)| 80.0 + c as f64 * 0.5)
    }

    #[test]
    fn apply_rms_averages_rows() {
        let p2 = Array2::from_shape_fn((3, 5), |(r, _)| (r + 1) as f64 * 4.0);
        let rms = apply_rms(&p2, &[0, 1, 2]);
        assert_eq!(rms.len(), 5);
        // avg = (4+8+12)/3 = 8, sqrt = 2.828, pressure2spl(2.828) ~ 99.03 dB
        assert!(rms[0] > 90.0);
    }

    #[test]
    fn apply_weighted_rms_uses_weights() {
        let p2 = Array2::from_shape_fn((2, 4), |(r, _)| (r + 1) as f64 * 9.0);
        let weights = Array1::from(vec![1.0, 3.0]);
        let rms = apply_weighted_rms(&p2, &[0, 1], &weights);
        assert_eq!(rms.len(), 4);
        // weighted avg = (9*1 + 18*3)/4 = 15.75, sqrt ~3.97, spl ~111.97
        assert!(rms[0] > 100.0);
    }

    #[test]
    fn cea2034_array_produces_expected_shape() {
        let spl = sample_spl(8, 10);
        let idx: Vec<Vec<usize>> = (0..8).map(|i| vec![i]).collect();
        let weights = Array1::ones(8);
        let cea = cea2034_array(&spl, &idx, &weights);
        assert_eq!(cea.shape(), &[idx.len() + 1, 10]);
    }

    #[test]
    fn cea2034_public_wrapper_matches_array() {
        let spl = sample_spl(8, 10);
        let idx: Vec<Vec<usize>> = (0..8).map(|i| vec![i]).collect();
        let weights = Array1::ones(8);
        let a = cea2034_array(&spl, &idx, &weights);
        let b = cea2034(&spl, &idx, &weights);
        assert_eq!(a, b);
    }
}
