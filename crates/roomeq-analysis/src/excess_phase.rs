//! Principled excess-phase assessment and bounded correction (roadmap step 4).
//!
//! Compares measured phase against a minimum-phase reconstruction after
//! bulk-delay removal, then gates any correction on SNR coverage and
//! window (smoothing) sensitivity. Corrections are delay/all-pass-style
//! only: a unity-magnitude FIR that cancels the smoothed excess phase,
//! realized causally with an explicit, caller-bounded added latency.
//!
//! Deliberately absent: universal audibility thresholds. Every gate bound
//! (SNR floor, coverage fraction, consistency tolerance, latency cap) is a
//! caller-supplied [`ExcessPhaseConfig`] / [`CorrectionConfig`] field, so a
//! verdict can never silently rest on a hard-coded group-delay, Q, or decay
//! limit. Missing phase input yields [`Assessment::Unsupported`]; a
//! minimum-phase classification alone never marks a dip safe to boost
//! (see [`ExcessPhaseReport::uncertain_dips`]).

use autoeq_core::phase_utils::reconstruct_minimum_phase;
use ndarray::Array1;
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Caller-supplied gate parameters. No `Default` is provided on purpose:
/// every bound must be an explicit measurement decision, never an
/// inherited inaudibility claim.
#[derive(Debug, Clone)]
pub struct ExcessPhaseConfig {
    /// Cosine taper width (octaves) applied to the correction target at the
    /// band edges during assessment.
    pub taper_oct: f64,
    /// Bins below this SNR are excluded from the assessment.
    pub snr_floor_db: f64,
    /// Minimum fraction of in-band bins at or above the SNR floor.
    pub min_valid_fraction: f64,
    /// Fractional-octave smoothing width for the reported excess group delay.
    pub smooth_narrow_oct: f64,
    /// Second (wider) smoothing width for the window-sensitivity check.
    pub smooth_wide_oct: f64,
    /// Maximum tolerated narrow-vs-wide group-delay disagreement.
    pub consistency_tol_ms: f64,
    /// When true, any uncertain dip forces an `Unknown` verdict.
    pub strict_dips: bool,
    /// Depth below the wide-smoothed magnitude that counts as a dip.
    pub dip_depth_db: f64,
    /// Frequency band (Hz) the assessment covers.
    pub analysis_band_hz: (f64, f64),
}

/// Measured input. `phase_deg` is `None` for magnitude-only captures.
#[derive(Debug, Clone)]
pub struct ExcessPhaseInput {
    pub freqs_hz: Vec<f64>,
    pub magnitude_db: Vec<f64>,
    pub phase_deg: Option<Vec<f64>>,
    pub snr_db: Vec<f64>,
    pub sample_rate_hz: f64,
}

/// A magnitude dip whose boost safety is unproven.
#[derive(Debug, Clone)]
pub struct Dip {
    pub freq_hz: f64,
    pub depth_db: f64,
    pub snr_db: f64,
}

/// Full assessment record. A report instance only reaches
/// [`Assessment::Supported`] with `gates_passed == true`.
#[derive(Debug, Clone)]
pub struct ExcessPhaseReport {
    pub bulk_delay_s: f64,
    pub bulk_delay_samples: f64,
    pub valid_fraction: f64,
    /// Smoothed (narrow width) excess group delay per input bin, in ms.
    pub excess_gd_ms: Vec<f64>,
    /// Wrapped residual excess phase per input bin, in degrees.
    pub residual_excess_deg: Vec<f64>,
    pub min_phase_deg: Vec<f64>,
    pub window_sensitivity_max_ms: f64,
    pub uncertain_dips: Vec<Dip>,
    /// Per-bin validity mask (in band and at/above the SNR floor).
    pub valid: Vec<bool>,
    /// Tapered, unwrapped correction target phase per bin, in radians.
    /// Negated smoothed excess where valid, cosine-tapered to zero at the
    /// band edges and identically zero outside the valid set.
    pub correction_phase_rad: Vec<f64>,
    pub gates_passed: bool,
}

/// Assessment outcome.
#[derive(Debug, Clone)]
pub enum Assessment {
    Supported(ExcessPhaseReport),
    Unknown {
        reason: String,
        partial: ExcessPhaseReport,
    },
    Unsupported {
        reason: String,
    },
}

/// Caller-supplied correction bounds.
#[derive(Debug, Clone)]
pub struct CorrectionConfig {
    pub max_taps: usize,
    pub max_added_latency_ms: f64,
}

/// Causal unity-magnitude FIR realizing the excess-phase cancellation.
#[derive(Debug, Clone)]
pub struct CorrectionProposal {
    pub fir_taps: Vec<f64>,
    pub sample_rate_hz: f64,
    pub design_delay_samples: usize,
    /// Measured main-peak position: the exact added latency in ms.
    pub added_latency_ms: f64,
    /// Pre-main-peak energy / total energy.
    pub pre_ring_energy_ratio: f64,
    /// Peak pre-main-peak amplitude / main-peak amplitude.
    pub pre_ring_peak_ratio: f64,
    /// Time from first above-threshold pre-peak sample to the main peak.
    pub pre_ring_duration_ms: f64,
    /// Worst-case |20·log10|H|| in dB over the valid assessment bins,
    /// measured on the realized taps by direct DFT. A short FIR cannot hold
    /// unity magnitude where its length is a fraction of the signal period
    /// (low band edge); callers gate on this number with their own threshold.
    pub max_magnitude_deviation_db: f64,
    pub valid_band_hz: (f64, f64),
}

fn wrap_pi(mut x: f64) -> f64 {
    while x > PI {
        x -= 2.0 * PI;
    }
    while x < -PI {
        x += 2.0 * PI;
    }
    x
}

fn unwrap_rad(phase: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(phase.len());
    if phase.is_empty() {
        return out;
    }
    let mut offset = 0.0;
    let mut prev = phase[0];
    out.push(prev);
    for &p in phase.iter().skip(1) {
        let mut d = p - prev;
        // A jump up through +pi means the wrapped phase fell through -pi:
        // continue the curve downward by removing a full turn, and vice versa.
        while d > PI {
            d -= 2.0 * PI;
            offset -= 2.0 * PI;
        }
        while d < -PI {
            d += 2.0 * PI;
            offset += 2.0 * PI;
        }
        out.push(p + offset);
        prev = p;
    }
    out
}

/// Fractional-octave boxcar smoothing via prefix sums over log2 frequency.
fn smooth_fractional_octave(freqs: &[f64], values: &[f64], width_oct: f64) -> Vec<f64> {
    let n = freqs.len();
    let mut out = vec![0.0; n];
    if n == 0 {
        return out;
    }
    let half = width_oct / 2.0;
    let logf: Vec<f64> = freqs.iter().map(|f| f.max(1e-9).log2()).collect();
    let mut prefix = vec![0.0; n + 1];
    for (i, v) in values.iter().enumerate() {
        prefix[i + 1] = prefix[i] + v;
    }
    let mut lo = 0;
    let mut hi = 0;
    for i in 0..n {
        while lo < n && logf[i] - logf[lo] > half {
            lo += 1;
        }
        while hi + 1 < n && logf[hi + 1] - logf[i] <= half {
            hi += 1;
        }
        let (a, b) = (lo.min(i), hi.max(i));
        let count = (b - a + 1) as f64;
        out[i] = (prefix[b + 1] - prefix[a]) / count;
    }
    out
}

fn least_squares_slope(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    if x.len() != y.len() || x.len() < 3 {
        return None;
    }
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        num += (xi - mx) * (yi - my);
        den += (xi - mx) * (xi - mx);
    }
    if den.abs() < 1e-18 {
        return None;
    }
    let slope = num / den;
    Some((slope, my - slope * mx))
}

fn interp_linear(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    let i = xs.partition_point(|&xi| xi < x).clamp(1, n - 1);
    let t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]).max(f64::EPSILON);
    ys[i - 1] * (1.0 - t) + ys[i] * t
}

/// Assess excess phase for one measured response.
pub fn assess_excess_phase(input: &ExcessPhaseInput, cfg: &ExcessPhaseConfig) -> Assessment {
    let n = input.freqs_hz.len();
    let unsupported = |why: &str| Assessment::Unsupported {
        reason: why.to_string(),
    };
    if n < 8
        || input.magnitude_db.len() != n
        || input.snr_db.len() != n
        || input.sample_rate_hz <= 0.0
    {
        return unsupported("insufficient or inconsistent input vectors");
    }
    let phase_deg = match &input.phase_deg {
        Some(p) if p.len() == n => p.clone(),
        _ => {
            return unsupported(
                "blocked_external: no measured phase; magnitude-only data cannot support phase correction",
            );
        }
    };

    let (f_lo, f_hi) = cfg.analysis_band_hz;
    let in_band: Vec<bool> = input
        .freqs_hz
        .iter()
        .map(|&f| f >= f_lo && f <= f_hi)
        .collect();
    let valid: Vec<bool> = input
        .snr_db
        .iter()
        .enumerate()
        .map(|(i, &snr)| in_band[i] && snr >= cfg.snr_floor_db)
        .collect();
    let band_count = in_band.iter().filter(|&&b| b).count();
    let valid_count = valid.iter().filter(|&&b| b).count();
    let valid_fraction = if band_count > 0 {
        valid_count as f64 / band_count as f64
    } else {
        0.0
    };

    // Minimum-phase reconstruction from magnitude (degrees -> radians).
    let freq_arr = Array1::from_vec(input.freqs_hz.clone());
    let mag_arr = Array1::from_vec(input.magnitude_db.clone());
    let min_phase_deg = reconstruct_minimum_phase(&freq_arr, &mag_arr).to_vec();
    let min_phase_rad: Vec<f64> = min_phase_deg.iter().map(|d| d.to_radians()).collect();

    // Shared fallback report builder for gate failures.
    let blank_report = || ExcessPhaseReport {
        bulk_delay_s: 0.0,
        bulk_delay_samples: 0.0,
        valid_fraction,
        excess_gd_ms: vec![0.0; n],
        residual_excess_deg: vec![0.0; n],
        min_phase_deg: min_phase_deg.clone(),
        window_sensitivity_max_ms: f64::INFINITY,
        uncertain_dips: Vec::new(),
        valid: valid.clone(),
        correction_phase_rad: vec![0.0; n],
        gates_passed: false,
    };
    let unknown = |reason: String| Assessment::Unknown {
        reason,
        partial: blank_report(),
    };

    if valid_fraction < cfg.min_valid_fraction {
        // Evaluated before any fitting: rejected data must not influence
        // the bulk-delay estimate even transiently.
        return Assessment::Unknown {
            reason: format!(
                "blocked_external: snr coverage {valid_fraction:.3} below required {0:.3}; noisy data returns unknown",
                cfg.min_valid_fraction
            ),
            partial: ExcessPhaseReport {
                bulk_delay_s: 0.0,
                bulk_delay_samples: 0.0,
                valid_fraction,
                excess_gd_ms: vec![0.0; n],
                residual_excess_deg: vec![0.0; n],
                min_phase_deg: min_phase_deg.clone(),
                window_sensitivity_max_ms: f64::INFINITY,
                uncertain_dips: Vec::new(),
                valid: valid.clone(),
                correction_phase_rad: vec![0.0; n],
                gates_passed: false,
            },
        };
    }

    // Bulk-delay fit, coarse-to-fine. A single unwrapped least-squares fit
    // aliases whenever the true inter-bin phase step exceeds half a turn
    // (large delays on sparse log grids), so gross timing comes first from
    // low-frequency valid bins where steps are unambiguous, then the
    // residual is unwrapped and refined over all valid bins.
    let meas_rad: Vec<f64> = phase_deg.iter().map(|d| d.to_radians()).collect();
    let f_geo_mid = (f_lo.max(1e-9) * f_hi.max(1e-9)).sqrt();
    let fit_subset = |f_max: f64| -> (Vec<f64>, Vec<f64>) {
        let idx: Vec<usize> = input
            .freqs_hz
            .iter()
            .enumerate()
            .filter(|&(i, f)| valid[i] && *f <= f_max)
            .map(|(i, _)| i)
            .collect();
        let sub_freq: Vec<f64> = idx.iter().map(|&i| input.freqs_hz[i]).collect();
        let sub_phase: Vec<f64> = idx.iter().map(|&i| meas_rad[i]).collect();
        let unwrapped = unwrap_rad(&sub_phase);
        (sub_freq, unwrapped)
    };
    // Coarse pass on the lower half of the band (geometric mid).
    let (cx, cy) = fit_subset(f_geo_mid);
    let Some((coarse_slope, _)) = least_squares_slope(&cx, &cy) else {
        return unknown("bulk-delay coarse fit degenerate over valid bins".to_string());
    };
    let mut bulk_delay_s = (-coarse_slope / (2.0 * PI)).max(0.0);
    // Refinement passes over all valid bins with the current estimate
    // removed, iterated to convergence: each pass fits only the residual
    // slope, so delay-like tilt migrates out of the residual and into the
    // reported bulk figure instead of burdening the correction FIR.
    for _ in 0..32 {
        let prev = bulk_delay_s;
        let removed: Vec<f64> = meas_rad
            .iter()
            .enumerate()
            .map(|(i, &p)| p + 2.0 * PI * input.freqs_hz[i] * bulk_delay_s)
            .collect();
        let idx: Vec<usize> = valid
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v)
            .map(|(i, _)| i)
            .collect();
        let rx: Vec<f64> = idx.iter().map(|&i| input.freqs_hz[i]).collect();
        let ry: Vec<f64> = {
            let sub: Vec<f64> = idx.iter().map(|&i| removed[i]).collect();
            unwrap_rad(&sub)
        };
        let Some((slope, _)) = least_squares_slope(&rx, &ry) else {
            return unknown("bulk-delay refinement fit degenerate".to_string());
        };
        bulk_delay_s = (bulk_delay_s - slope / (2.0 * PI)).max(0.0);
        if (bulk_delay_s - prev).abs() < 1e-12 {
            break;
        }
    }
    let unwrapped_meas: Vec<f64> = meas_rad
        .iter()
        .enumerate()
        .map(|(i, &p)| p + 2.0 * PI * input.freqs_hz[i] * bulk_delay_s)
        .collect();
    let unwrapped_meas = unwrap_rad(&unwrapped_meas);

    let bulk_delay_samples = bulk_delay_s * input.sample_rate_hz;

    // Residual excess phase and group delay (`unwrapped_meas` already has
    // the fitted bulk delay removed).
    let residual: Vec<f64> = unwrapped_meas
        .iter()
        .enumerate()
        .map(|(i, &p)| p - min_phase_rad[i])
        .collect();
    let residual_unwrapped = unwrap_rad(&residual);
    let mut gd_ms = vec![0.0; n];
    for (i, g) in gd_ms.iter_mut().enumerate() {
        let (a, b) = if i == 0 {
            (0, 1.min(n - 1))
        } else if i + 1 >= n {
            (n.saturating_sub(2), n - 1)
        } else {
            (i - 1, i + 1)
        };
        let df = (input.freqs_hz[b] - input.freqs_hz[a]).max(f64::EPSILON);
        *g = -(residual_unwrapped[b] - residual_unwrapped[a]) / (2.0 * PI * df) * 1000.0;
    }
    let narrow = smooth_fractional_octave(&input.freqs_hz, &gd_ms, cfg.smooth_narrow_oct);
    let wide = smooth_fractional_octave(&input.freqs_hz, &gd_ms, cfg.smooth_wide_oct);
    // Candidate dip regions: local minima deeper than the wide-smoothed
    // magnitude by dip_depth_db. Their rapid min-phase swing is a magnitude
    // artifact with its own gate below, so it must not trip the
    // window-sensitivity gate here.
    let smooth_mag =
        smooth_fractional_octave(&input.freqs_hz, &input.magnitude_db, cfg.smooth_wide_oct);
    let mut dip_regions: Vec<(usize, usize)> = Vec::new();
    {
        let mut i = 1;
        while i + 1 < n {
            let is_min = input.magnitude_db[i] <= input.magnitude_db[i - 1]
                && input.magnitude_db[i] <= input.magnitude_db[i + 1];
            let depth = smooth_mag[i] - input.magnitude_db[i];
            if is_min && in_band[i] && depth >= cfg.dip_depth_db {
                let start = i;
                while i + 1 < n && smooth_mag[i + 1] - input.magnitude_db[i + 1] >= cfg.dip_depth_db
                {
                    i += 1;
                }
                dip_regions.push((start, i));
            }
            i += 1;
        }
    }
    // Dilate the exclusion by the wide-smoothing half-width plus margin: the
    // min-phase bump of a notch spreads its skirts past the bins where the
    // magnitude depth itself exceeds the threshold.
    let dilate = cfg.smooth_wide_oct / 2.0 + cfg.smooth_narrow_oct;
    let logf: Vec<f64> = input.freqs_hz.iter().map(|f| f.max(1e-9).log2()).collect();
    let in_dip = |i: usize| {
        dip_regions.iter().any(|&(a, b)| {
            if i >= a && i <= b {
                return true;
            }
            let edge = if i < a {
                logf[a] - logf[i]
            } else {
                logf[i] - logf[b]
            };
            edge <= dilate
        })
    };
    let mut sensitivity_max = 0.0f64;
    for i in 0..n {
        if valid[i] && !in_dip(i) {
            sensitivity_max = sensitivity_max.max((narrow[i] - wide[i]).abs());
        }
    }
    if sensitivity_max > cfg.consistency_tol_ms {
        return unknown(format!(
            "window-sensitive excess group delay: narrow/wide disagreement {sensitivity_max:.3} ms exceeds {0:.3} ms",
            cfg.consistency_tol_ms
        ));
    }

    // Uncertain dips: candidate regions with marginal SNR. Minimum-phase
    // shape alone never clears these for boosting.
    let mut uncertain_dips = Vec::new();
    for &(start, end) in &dip_regions {
        let deepest = (start..=end)
            .min_by(|&a, &b| {
                input.magnitude_db[a]
                    .partial_cmp(&input.magnitude_db[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(start);
        let snr = input.snr_db[deepest];
        if snr < cfg.snr_floor_db + 3.0 {
            uncertain_dips.push(Dip {
                freq_hz: input.freqs_hz[deepest],
                depth_db: smooth_mag[deepest] - input.magnitude_db[deepest],
                snr_db: snr,
            });
        }
    }
    if cfg.strict_dips && !uncertain_dips.is_empty() {
        return unknown(format!(
            "uncertain dip at {:.1} Hz ({:.1} dB, snr {:.1} dB) blocks boost-safe verdict",
            uncertain_dips[0].freq_hz, uncertain_dips[0].depth_db, uncertain_dips[0].snr_db
        ));
    }

    // Correction target: negated narrow-smoothed residual where valid,
    // cosine-tapered to zero at the band edges.
    let smooth_resid =
        smooth_fractional_octave(&input.freqs_hz, &residual_unwrapped, cfg.smooth_narrow_oct);
    let log_lo = f_lo.max(1e-9).log2();
    let log_hi = f_hi.max(1e-9).log2();
    let taper_half = cfg.taper_oct.max(1e-6) / 2.0;
    let mut correction_phase_rad = vec![0.0; n];
    for (i, c) in correction_phase_rad.iter_mut().enumerate() {
        if !valid[i] {
            continue;
        }
        let lf = input.freqs_hz[i].max(1e-9).log2();
        let edge = ((lf - log_lo) / taper_half)
            .min((log_hi - lf) / taper_half)
            .clamp(0.0, 1.0);
        let taper = 0.5 - 0.5 * (PI * edge.min(1.0)).cos();
        *c = -smooth_resid[i] * taper;
    }

    Assessment::Supported(ExcessPhaseReport {
        bulk_delay_s,
        bulk_delay_samples,
        valid_fraction,
        excess_gd_ms: narrow,
        residual_excess_deg: residual.iter().map(|&r| wrap_pi(r).to_degrees()).collect(),
        min_phase_deg,
        window_sensitivity_max_ms: sensitivity_max,
        uncertain_dips,
        valid,
        correction_phase_rad,
        gates_passed: true,
    })
}

/// Design the bounded causal correction FIR for a supported assessment.
///
/// The target is unity magnitude with the tapered correction phase, so the
/// filter can only be delay/all-pass-like: it never applies magnitude EQ.
/// Returns an error for gated-out reports instead of a best-effort filter.
pub fn propose_correction(
    report: &ExcessPhaseReport,
    freqs_hz: &[f64],
    sample_rate_hz: f64,
    cfg: &CorrectionConfig,
) -> Result<CorrectionProposal, String> {
    if !report.gates_passed {
        return Err(
            "blocked_external: assessment did not pass SNR/window gates; no correction proposed"
                .to_string(),
        );
    }
    let n = freqs_hz.len();
    if n != report.correction_phase_rad.len() || n < 8 || sample_rate_hz <= 0.0 {
        return Err("inconsistent report/frequency input".to_string());
    }
    if cfg.max_taps < 16 {
        return Err("max_taps too small for a causal realization".to_string());
    }
    if cfg.max_added_latency_ms <= 0.0 {
        return Err("max_added_latency_ms must be positive".to_string());
    }

    // Frequency-sample the unity-magnitude target onto a uniform grid.
    let m = (16 * n).next_power_of_two().clamp(4096, 65536);
    let half = m / 2;
    let mut spectrum = vec![Complex64::new(0.0, 0.0); m];
    for (k, bin) in spectrum.iter_mut().enumerate().take(half + 1) {
        let f = k as f64 * sample_rate_hz / m as f64;
        // DC and Nyquist must stay real to preserve conjugate symmetry
        // (and a delay-like correction carries no phase at DC anyway).
        let phase = if k == 0 || k == half {
            0.0
        } else {
            interp_linear(freqs_hz, &report.correction_phase_rad, f)
        };
        *bin = Complex64::from_polar(1.0, phase);
    }
    for k in 1..half {
        spectrum[m - k] = spectrum[k].conj();
    }
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(m);
    let mut buf = spectrum.clone();
    ifft.process(&mut buf);
    let inv = m as f64;
    let ir: Vec<f64> = buf.iter().map(|c| c.re / inv).collect();

    // Causal realization: circular shift so the main energy sits at the
    // design delay, then apply a delay-centered window to the caller-bounded
    // tap count.
    let max_delay =
        ((cfg.max_added_latency_ms / 1000.0 * sample_rate_hz).round() as usize).clamp(1, half);
    let taps = cfg.max_taps.min(m);
    let realize = |delay: usize| -> Vec<f64> {
        // Delay-centered cos^2 window: the taper peak sits on the filter
        // energy (the design delay), not at taps/2. A window centered
        // elsewhere weights one lobe tail more than the other and skews the
        // realized group delay away from the reported peak. Half-widths reach
        // zero exactly at the available edges.
        let left = delay.max(1) as f64;
        let right = taps.saturating_sub(delay).max(2) as f64;
        let mut out = vec![0.0; taps];
        for (i, t) in out.iter_mut().enumerate() {
            let src = (i + m - delay) % m;
            let w = if i <= delay {
                let u = (delay - i) as f64 / left;
                (0.5 * PI * u.min(1.0)).cos().powi(2)
            } else {
                let u = (i - delay) as f64 / right;
                (0.5 * PI * u.min(1.0)).cos().powi(2)
            };
            *t = ir[src] * w;
        }
        out
    };
    let argmax = |taps_out: &[f64]| {
        taps_out
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    // First pass at the full cap; if the lobe's genuine group-delay spread
    // pushes the measured peak past it, redesign once with the design delay
    // reduced by the overshoot. Moving the design (rather than shearing the
    // taps) keeps the realized phase faithful to the target.
    let mut delay = max_delay.min(m.saturating_sub(taps / 2).max(1));
    let mut taps_out = realize(delay);
    let mut peak = argmax(&taps_out);
    if peak > max_delay {
        delay = max_delay.saturating_sub(peak - max_delay).max(1);
        taps_out = realize(delay);
        peak = argmax(&taps_out);
    }
    let peak_abs = taps_out[peak].abs().max(f64::EPSILON);
    let total: f64 = taps_out.iter().map(|t| t * t).sum();
    let pre: f64 = taps_out[..peak].iter().map(|t| t * t).sum();
    let pre_peak = taps_out[..peak]
        .iter()
        .map(|t| t.abs())
        .fold(0.0_f64, f64::max);
    let first_above = taps_out[..peak]
        .iter()
        .position(|t| t.abs() >= 0.01 * peak_abs)
        .unwrap_or(peak);
    let f_lo = freqs_hz[0];
    let f_hi = *freqs_hz.last().unwrap_or(&f_lo);
    // Realized magnitude fidelity over the valid bins (direct DFT).
    let mut max_mag_dev_db = 0.0f64;
    for (i, &f) in freqs_hz.iter().enumerate() {
        if !report.valid[i] {
            continue;
        }
        let (mut acc_re, mut acc_im) = (0.0, 0.0);
        for (k, t) in taps_out.iter().enumerate() {
            let ph = 2.0 * PI * f * k as f64 / sample_rate_hz;
            acc_re += t * ph.cos();
            acc_im -= t * ph.sin();
        }
        let mag = (acc_re * acc_re + acc_im * acc_im).sqrt().max(1e-12);
        max_mag_dev_db = max_mag_dev_db.max((20.0 * mag.log10()).abs());
    }

    Ok(CorrectionProposal {
        fir_taps: taps_out,
        sample_rate_hz,
        design_delay_samples: delay,
        added_latency_ms: peak as f64 / sample_rate_hz * 1000.0,
        pre_ring_energy_ratio: if total > 0.0 { pre / total } else { 0.0 },
        pre_ring_peak_ratio: pre_peak / peak_abs,
        pre_ring_duration_ms: (peak.saturating_sub(first_above)) as f64 / sample_rate_hz * 1000.0,
        max_magnitude_deviation_db: max_mag_dev_db,
        valid_band_hz: (f_lo, f_hi),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn log_freqs(n: usize, f_lo: f64, f_hi: f64) -> Vec<f64> {
        (0..n)
            .map(|i| f_lo * (f_hi / f_lo).powf(i as f64 / (n - 1) as f64))
            .collect()
    }

    fn test_config() -> ExcessPhaseConfig {
        ExcessPhaseConfig {
            taper_oct: 0.5,
            snr_floor_db: 10.0,
            min_valid_fraction: 0.8,
            smooth_narrow_oct: 1.0 / 6.0,
            smooth_wide_oct: 1.0 / 2.0,
            consistency_tol_ms: 0.5,
            strict_dips: false,
            dip_depth_db: 6.0,
            analysis_band_hz: (40.0, 18_000.0),
        }
    }

    /// Flat magnitude + pure delay: min-phase ~ 0, excess ~ 0 after bulk removal.
    fn pure_delay_input(tau_s: f64) -> ExcessPhaseInput {
        let freqs = log_freqs(256, 40.0, 18_000.0);
        let phase_deg: Vec<f64> = freqs
            .iter()
            .map(|&f| (-2.0 * PI * f * tau_s).to_degrees())
            .collect();
        ExcessPhaseInput {
            freqs_hz: freqs,
            magnitude_db: vec![0.0; 256],
            phase_deg: Some(phase_deg),
            snr_db: vec![40.0; 256],
            sample_rate_hz: FS,
        }
    }

    #[test]
    fn pure_delay_recovers_bulk_and_reports_supported() {
        let input = pure_delay_input(0.0023);
        match assess_excess_phase(&input, &test_config()) {
            Assessment::Supported(report) => {
                assert!((report.bulk_delay_s - 0.0023).abs() < 0.0023 * 0.05 + 1e-5);
                assert!((report.bulk_delay_samples - 0.0023 * FS).abs() < 0.05 * 0.0023 * FS + 1.0);
                assert!(report.gates_passed);
                let max_gd = report
                    .excess_gd_ms
                    .iter()
                    .zip(report.valid.iter())
                    .filter(|&(_, v)| *v)
                    .map(|(&g, _)| g.abs())
                    .fold(0.0_f64, f64::max);
                assert!(max_gd < 0.2, "residual excess gd {max_gd} ms");
            }
            other => panic!("expected Supported, got {other:?}"),
        }
    }

    #[test]
    fn first_order_allpass_shows_expected_excess_gd() {
        // H(f) = A(f) with A a first-order all-pass, corner f0:
        // arg A = -2 atan(f/f0), so excess gd(f) = 1 / (pi f0 (1 + (f/f0)^2)).
        let f0 = 200.0;
        let freqs = log_freqs(256, 40.0, 18_000.0);
        let phase_deg: Vec<f64> = freqs
            .iter()
            .map(|&f| {
                let w = f / f0;
                (-2.0 * (w).atan()).to_degrees()
            })
            .collect();
        let input = ExcessPhaseInput {
            freqs_hz: freqs.clone(),
            magnitude_db: vec![0.0; 256],
            phase_deg: Some(phase_deg),
            snr_db: vec![40.0; 256],
            sample_rate_hz: FS,
        };
        match assess_excess_phase(&input, &test_config()) {
            Assessment::Supported(report) => {
                // Expected excess gd at 40 Hz in ms.
                let expected = 1000.0 / (PI * f0 * (1.0 + (40.0 / f0).powi(2)));
                let idx = freqs
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (*a - 40.0).abs().partial_cmp(&(*b - 40.0).abs()).unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap();
                let got = report.excess_gd_ms[idx];
                assert!(
                    (got - expected).abs() < 0.35,
                    "got {got} ms, expected {expected} ms"
                );
            }
            other => panic!("expected Supported, got {other:?}"),
        }
    }

    #[test]
    fn low_snr_returns_unknown_and_blocks_correction() {
        let mut input = pure_delay_input(0.001);
        input.snr_db = vec![-10.0; 256];
        let cfg = test_config();
        match assess_excess_phase(&input, &cfg) {
            Assessment::Unknown { reason, partial } => {
                assert!(reason.contains("snr"), "{reason}");
                assert!(!partial.gates_passed);
                let err = propose_correction(
                    &partial,
                    &input.freqs_hz,
                    FS,
                    &CorrectionConfig {
                        max_taps: 256,
                        max_added_latency_ms: 5.0,
                    },
                )
                .expect_err("gated-out report must not yield a filter");
                assert!(err.contains("blocked_external"), "{err}");
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn missing_phase_is_unsupported() {
        let mut input = pure_delay_input(0.001);
        input.phase_deg = None;
        match assess_excess_phase(&input, &test_config()) {
            Assessment::Unsupported { reason } => assert!(reason.contains("phase"), "{reason}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn gates_are_caller_controlled_not_hard_coded() {
        // Same data: passes at 0.5 coverage demand, fails at 0.999.
        let mut input = pure_delay_input(0.001);
        for i in (0..256).step_by(4) {
            input.snr_db[i] = -10.0;
        }
        let mut loose = test_config();
        loose.min_valid_fraction = 0.5;
        assert!(matches!(
            assess_excess_phase(&input, &loose),
            Assessment::Supported(_)
        ));
        let mut tight = test_config();
        tight.min_valid_fraction = 0.999;
        assert!(matches!(
            assess_excess_phase(&input, &tight),
            Assessment::Unknown { .. }
        ));
    }

    #[test]
    fn sharp_step_is_window_sensitive() {
        // A near-discontinuous excess-phase step: narrow smoothing follows
        // it while wide smoothing does not.
        let mut input = pure_delay_input(0.001);
        let phase = input.phase_deg.as_mut().unwrap();
        for (i, f) in input.freqs_hz.iter().enumerate() {
            if *f > 1000.0 {
                phase[i] += 120.0;
            }
        }
        let mut cfg = test_config();
        cfg.consistency_tol_ms = 0.05;
        match assess_excess_phase(&input, &cfg) {
            Assessment::Unknown { reason, .. } => assert!(reason.contains("window"), "{reason}"),
            other => panic!("expected window-sensitive Unknown, got {other:?}"),
        }
    }

    #[test]
    fn strict_dips_block_uncertain_boost() {
        let mut input = pure_delay_input(0.001);
        // Deep narrow dip with poor local SNR.
        for i in 120..136 {
            input.magnitude_db[i] = -18.0;
            input.snr_db[i] = 2.0;
        }
        let mut cfg = test_config();
        cfg.strict_dips = true;
        match assess_excess_phase(&input, &cfg) {
            Assessment::Unknown { reason, .. } => assert!(reason.contains("dip"), "{reason}"),
            other => panic!("expected dip-blocked Unknown, got {other:?}"),
        }
        let loose = test_config();
        match assess_excess_phase(&input, &loose) {
            Assessment::Supported(report) => assert!(!report.uncertain_dips.is_empty()),
            other => panic!("expected Supported with dip note, got {other:?}"),
        }
    }

    #[test]
    fn correction_is_unity_magnitude_causal_and_bounded() {
        let f0 = 200.0;
        let freqs = log_freqs(256, 40.0, 18_000.0);
        let phase_deg: Vec<f64> = freqs
            .iter()
            .map(|&f| (-2.0 * (f / f0).atan()).to_degrees())
            .collect();
        let input = ExcessPhaseInput {
            freqs_hz: freqs.clone(),
            magnitude_db: vec![0.0; 256],
            phase_deg: Some(phase_deg),
            snr_db: vec![40.0; 256],
            sample_rate_hz: FS,
        };
        let Assessment::Supported(report) = assess_excess_phase(&input, &test_config()) else {
            panic!("expected Supported");
        };
        let ccfg = CorrectionConfig {
            max_taps: 512,
            max_added_latency_ms: 4.0,
        };
        let proposal = propose_correction(&report, &freqs, FS, &ccfg).expect("correction");
        assert_eq!(proposal.fir_taps.len(), 512);
        assert!(proposal.added_latency_ms <= 4.0 + 1e-9);
        assert!(proposal.added_latency_ms > 0.0);
        assert!((0.0..=1.0).contains(&proposal.pre_ring_energy_ratio));
        assert!((0.0..=1.0).contains(&proposal.pre_ring_peak_ratio));

        // Realized response: FFT the taps, compensate the design-delay linear
        // phase, and compare against the target.
        let m = 8192;
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(m);
        let mut buf = vec![Complex64::new(0.0, 0.0); m];
        for (i, t) in proposal.fir_taps.iter().enumerate() {
            buf[i] = Complex64::new(*t, 0.0);
        }
        fft.process(&mut buf);
        let delay = proposal.design_delay_samples;
        let nearest = |f: f64| {
            freqs
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| (*a - f).abs().partial_cmp(&(*b - f).abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        let response = |f: f64| {
            let k = (f * m as f64 / FS).round() as usize;
            buf[k] * Complex64::from_polar(1.0, 2.0 * PI * f * delay as f64 / FS)
        };
        // Unity magnitude where a 512-tap filter has time support (>= 500 Hz
        // at 48 kHz); the low edge is covered by the reported deviation below.
        let mut worst_db = 0.0f64;
        for &f in &[500.0, 2000.0, 8000.0] {
            let h = response(f);
            worst_db = worst_db.max(h.norm().log10().abs() * 20.0);
            // Phase should cancel the measured excess at probe freqs.
            let idx = nearest(f);
            let mut diff = (h.arg() - report.correction_phase_rad[idx]).abs();
            while diff > PI {
                diff = (diff - 2.0 * PI).abs();
            }
            assert!(diff < 0.35, "f={f}: residual {diff} rad");
        }
        assert!(worst_db < 1.0, "magnitude deviation {worst_db} dB");
        // The reported deviation must be consistent with an independent
        // measurement over the same valid bins.
        let mut check_db = 0.0f64;
        for (i, &f) in freqs.iter().enumerate() {
            if !report.valid[i] {
                continue;
            }
            let k = (f * m as f64 / FS).round() as usize;
            check_db = check_db.max(buf[k].norm().log10().abs() * 20.0);
        }
        assert!(
            (proposal.max_magnitude_deviation_db - check_db).abs() < 0.6,
            "reported {} vs measured {check_db}",
            proposal.max_magnitude_deviation_db
        );
        // The low edge is window-limited, not method-limited: the all-pass
        // lobe spreads ~73 samples while the 4 ms cap leaves 191 samples of
        // left support. The deviation metric documents this honestly (< 3 dB
        // here); with a latency budget that fits the lobe, the same design
        // holds unity magnitude across the whole band.
        assert!(
            proposal.max_magnitude_deviation_db < 3.0,
            "short-budget deviation {}",
            proposal.max_magnitude_deviation_db
        );
        let roomy = propose_correction(
            &report,
            &freqs,
            FS,
            &CorrectionConfig {
                max_taps: 2048,
                max_added_latency_ms: 16.0,
            },
        )
        .expect("roomy correction");
        assert!(roomy.added_latency_ms <= 16.0 + 1e-9);
        assert!(
            roomy.max_magnitude_deviation_db < 1.0,
            "roomy-budget deviation {}",
            roomy.max_magnitude_deviation_db
        );
    }
}
