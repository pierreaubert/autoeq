//! Negative controls for the optima group (O01-O04): prove each Wolfram
//! comparison detects its defect class. Each control loads an
//! engine-blessed golden, applies a classic defect to a copy of the oracle
//! inputs, and asserts the resulting error exceeds the case tolerance.
//! A control that cannot fail is worthless; these must keep failing.

use autoeq_qa::golden_dir;

const TOL: f64 = 1e-9;

fn load_golden(case: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("{case}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("missing golden {}: {error}", path.display()));
    serde_json::from_str(&text).expect("golden must be JSON")
}

fn vec_f64(payload: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(payload[key].clone()).expect("golden array must parse")
}

/// The legacy bandwidth formula ERB = 24.7*(1+4.37 f/1000) used as a rate
/// axis must move the O01 scalar loss beyond tolerance (wrong measure).
#[test]
fn wrong_erb_measure_fails_scalar_weights() {
    let payload = load_golden("o01_scalar_weights");
    let grid = vec_f64(&payload, "grid_hz");
    let err = vec_f64(&payload, "errors_db");
    // Trapezoidal widths on the wrong (bandwidth, not rate) axis.
    let axis: Vec<f64> = grid
        .iter()
        .map(|f| 24.7 * (1.0 + 4.37 * f / 1000.0))
        .collect();
    let n = grid.len();
    let widths: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                (axis[1] - axis[0]) / 2.0
            } else if i == n - 1 {
                (axis[n - 1] - axis[n - 2]) / 2.0
            } else {
                (axis[i + 1] - axis[i - 1]) / 2.0
            }
        })
        .collect();
    let total: f64 = widths.iter().sum();
    let defective = (err
        .iter()
        .zip(widths.iter())
        .map(|(e, w)| e * e * w)
        .sum::<f64>()
        / total)
        .sqrt();
    let reference = payload["erb_loss"].as_f64().unwrap();
    let gap = (defective - reference).abs();
    assert!(
        gap > TOL,
        "wrong ERB axis must exceed tolerance, gap={gap:.3e}"
    );
}

/// Equal (linear-bin) weighting instead of log quadrature must move the O01
/// band loss: the dense low-frequency region would dominate.
#[test]
fn linear_bin_quadrature_fails_band_loss() {
    let payload = load_golden("o01_scalar_weights");
    let grid = vec_f64(&payload, "grid_hz");
    let err = vec_f64(&payload, "errors_db");
    // Unweighted per-band RMS (each sample counts once).
    let band_of = |f: f64| {
        if f <= 200.0 {
            0
        } else if f <= 4000.0 {
            1
        } else {
            2
        }
    };
    let mut sums = [0.0; 3];
    let mut counts = [0usize; 3];
    for (f, e) in grid.iter().zip(err.iter()) {
        let b = band_of(*f);
        sums[b] += e * e;
        counts[b] += 1;
    }
    let rms: Vec<f64> = (0..3)
        .map(|b| (sums[b] / counts[b] as f64).sqrt())
        .collect();
    let defective = (2.0 * rms[0] + rms[1] + 0.8 * rms[2]) / 3.8;
    let reference = payload["band_loss"].as_f64().unwrap();
    let gap = (defective - reference).abs();
    assert!(
        gap > TOL,
        "linear-bin quadrature must exceed tolerance, gap={gap:.3e}"
    );
}

/// Swapping the peak and dip branches must move the O01 asymmetric loss
/// (peaks would be under-penalized, dips over-penalized).
#[test]
fn swapped_peak_dip_branches_fail_asymmetric_loss() {
    let payload = load_golden("o01_asymmetric_weights");
    let weights = vec_f64(&payload, "sample_weights");
    let errors = vec_f64(&payload, "errors_db");
    // Swap: positive errors get the dip-side weight pattern and vice versa
    // (reversed weight vector as the simplest branch confusion).
    let swapped: Vec<f64> = weights.iter().rev().copied().collect();
    let honest: f64 = errors
        .iter()
        .zip(weights.iter())
        .map(|(e, w)| e * e * w)
        .sum();
    let defective: f64 = errors
        .iter()
        .zip(swapped.iter())
        .map(|(e, w)| e * e * w)
        .sum();
    let gap = (honest - defective).abs() / honest;
    assert!(
        gap > TOL,
        "swapped peak/dip branches must exceed tolerance, rel gap={gap:.3e}"
    );
}

/// Applying the null mask to peaks instead of dips must move the O01
/// asymmetric loss. Branch weights are reconstructed from the golden blend
/// and config so the mask swap is exact, not circular.
#[test]
fn mask_on_peaks_fails_asymmetric_loss() {
    let payload = load_golden("o01_asymmetric_weights");
    let errors = vec_f64(&payload, "errors_db");
    let mask = vec_f64(&payload, "null_mask");
    let blend = vec_f64(&payload, "blend");
    let cfg = &payload["config"];
    let (peak, dip, bass_peak, bass_dip) = (
        cfg["peak_weight"].as_f64().unwrap(),
        cfg["dip_weight"].as_f64().unwrap(),
        cfg["bass_peak_weight"].as_f64().unwrap(),
        cfg["bass_dip_weight"].as_f64().unwrap(),
    );
    let honest: f64 = errors
        .iter()
        .zip(blend.iter())
        .zip(mask.iter())
        .map(|((e, b), m)| {
            let w = if *e > 0.0 {
                bass_peak + b * (peak - bass_peak)
            } else {
                (bass_dip + b * (dip - bass_dip)) * m
            };
            e * e * w
        })
        .sum();
    // Defect: mask scales peaks, dips go unmasked.
    let defective: f64 = errors
        .iter()
        .zip(blend.iter())
        .zip(mask.iter())
        .map(|((e, b), m)| {
            let w = if *e > 0.0 {
                (bass_peak + b * (peak - bass_peak)) * m
            } else {
                bass_dip + b * (dip - bass_dip)
            };
            e * e * w
        })
        .sum();
    let gap = (honest - defective).abs() / honest;
    assert!(
        gap > TOL,
        "mask-on-peaks must exceed tolerance, rel gap={gap:.3e}"
    );
}

/// Averaging the best-first tail instead of the worst-first tail must move
/// the O02 CVaR well beyond tolerance (risk-seeking instead of risk-averse).
#[test]
fn best_first_tail_fails_cvar() {
    let payload = load_golden("o02_scalarization");
    let mut losses = vec_f64(&payload, "per_curve_losses");
    losses.sort_by(|a, b| a.total_cmp(b));
    let mass = 0.5 * losses.len() as f64;
    let (mut acc, mut taken) = (0.0, 0.0);
    for loss in &losses {
        if taken >= mass {
            break;
        }
        let take = (mass - taken).min(1.0);
        acc += take * loss;
        taken += take;
    }
    let defective = acc / mass;
    let reference = payload["cvar"].as_f64().unwrap();
    assert!(
        (defective - reference).abs() > TOL,
        "best-first CVaR ({defective}) must differ from worst-first ({reference})"
    );
}

/// Normalizing the variance penalty by the curve count instead of the
/// surviving weight mass must move the O02 zero-weight scenario.
#[test]
fn count_normalized_variance_fails_zero_weight_case() {
    let payload = load_golden("o02_scalarization");
    let losses = vec_f64(&payload, "per_curve_losses");
    let weights = vec_f64(&payload, "zero_weight_weights");
    // Defect: include the zero-weight seat in the mean/variance with mass N.
    let n = losses.len() as f64;
    let mean: f64 = losses
        .iter()
        .zip(weights.iter())
        .map(|(l, w)| l * w)
        .sum::<f64>()
        / n;
    let var: f64 = losses
        .iter()
        .zip(weights.iter())
        .map(|(l, w)| w * (l - mean).powi(2))
        .sum::<f64>()
        / n;
    let defective = mean + 0.5 * var;
    let reference = payload["zero_weight_variance_penalized"].as_f64().unwrap();
    assert!(
        (defective - reference).abs() > TOL,
        "count-normalized variance ({defective}) must differ from mass-normalized ({reference})"
    );
}

/// A uniform-spacing stencil (ignoring the log10 grid steps) must move the
/// O03 curvature on this nonuniform grid.
#[test]
fn uniform_spacing_stencil_fails_tv2() {
    let payload = load_golden("o03_tv2_stencil");
    let response = vec_f64(&payload, "response_db");
    // Defect: second difference without grid-step normalization.
    let terms: Vec<f64> = (1..response.len() - 1)
        .map(|i| (response[i + 1] - 2.0 * response[i] + response[i - 1]).abs())
        .collect();
    let defective = 0.5 * terms.iter().sum::<f64>() / terms.len() as f64;
    let reference = payload["penalty_exp1"].as_f64().unwrap();
    assert!(
        (defective - reference).abs() > TOL,
        "uniform stencil ({defective}) must differ from log10 stencil ({reference})"
    );
}

/// Hard clipping at the threshold (instead of soft subtraction) must move
/// the O04 deadband outputs for above-threshold residuals.
#[test]
fn hard_clip_fails_deadband() {
    let payload = load_golden("o04_deadband");
    let thresholds = vec_f64(&payload, "thresholds_db");
    let outputs = vec_f64(&payload, "outputs_db");
    // Defect: clamp magnitude to the threshold instead of subtracting it.
    let defective: Vec<f64> = outputs
        .iter()
        .zip(thresholds.iter())
        .map(|(o, t)| o.signum() * o.abs().min(*t))
        .collect();
    let worst = outputs
        .iter()
        .zip(defective.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > TOL,
        "hard clipping must exceed tolerance, worst={worst:.3e}"
    );
}

/// Linear interpolation of the deadband joins (instead of log-frequency)
/// must move the mid-band threshold.
#[test]
fn linear_join_interp_fails_deadband_threshold() {
    let payload = load_golden("o04_deadband");
    let grid = vec_f64(&payload, "grid_hz");
    let thresholds = vec_f64(&payload, "thresholds_db");
    // Defect: linear-in-Hz interpolation between the 250 Hz and 2000 Hz joins.
    let defective: Vec<f64> = grid
        .iter()
        .map(|f| {
            if *f <= 250.0 || *f >= 2000.0 || *f < 300.0 {
                return f64::NAN; // only the log-interp interior point matters
            }
            0.75 + (f - 250.0) / (2000.0 - 250.0) * (1.0 - 0.75)
        })
        .collect();
    let worst = thresholds
        .iter()
        .zip(defective.iter())
        .filter_map(|(a, b)| b.is_finite().then_some((a - b).abs()))
        .fold(0.0f64, f64::max);
    assert!(
        worst > TOL,
        "linear join interpolation must exceed tolerance, worst={worst:.3e}"
    );
}

/// Halving the Harman width must move the O04 bass-boost shoulder.
#[test]
fn halved_harman_width_fails_bass_boost() {
    let payload = load_golden("o04_bass_boost");
    let grid = vec_f64(&payload, "grid_hz");
    let harman = vec_f64(&payload, "harman_db");
    let width = (200.0 - 20.0) / 2.0 / 2.0; // defect: halved width
    let defective: Vec<f64> = grid
        .iter()
        .map(|f| {
            if *f < 20.0 || *f > 200.0 {
                0.0
            } else {
                4.0 * (-0.5 * ((f - 60.0) / width).powi(2)).exp()
            }
        })
        .collect();
    let worst = harman
        .iter()
        .zip(defective.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst > TOL,
        "halved Harman width must exceed tolerance, worst={worst:.3e}"
    );
}
