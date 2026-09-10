use ndarray::Array1;

pub(super) fn same_frequency_grid(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1.0))
}

const DISPLAY_MIN_FREQ: f64 = math_audio_iir_fir::AUDIBLE_MIN_FREQ;

const DISPLAY_MAX_FREQ: f64 = math_audio_iir_fir::AUDIBLE_MAX_FREQ;

/// Extend a curve's frequency range to cover 20 Hz – 20 kHz for display.
///
/// Points outside the measurement range are extrapolated using the SPL trend
/// over the edge octave at each boundary (least-squares fit in log
/// frequency, capped at [`MAX_EDGE_TREND_DB_PER_OCTAVE`]). A single adjacent
/// bin pair must not define the trend: dense captures place neighbours
/// fractions of a percent apart, so one noisy bin implies hundreds of
/// dB-per-octave and invents fantasy levels (a subwoofer measured to 200 Hz
/// reached 1004 dB at 20 kHz). The original measurement data points are
/// preserved.
pub fn extend_curve_to_full_range(curve: &crate::Curve) -> crate::Curve {
    if curve.freq.is_empty() || curve.spl.len() != curve.freq.len() {
        return curve.clone();
    }

    let meas_min = curve.freq[0];
    let meas_max = curve.freq[curve.freq.len() - 1];

    // If curve already approximately covers 20 Hz – 20 kHz, return as-is
    if meas_min <= DISPLAY_MIN_FREQ * 1.05 && meas_max >= DISPLAY_MAX_FREQ * 0.95 {
        return curve.clone();
    }

    let first_spl = curve.spl[0];
    let last_spl = curve.spl[curve.spl.len() - 1];
    let low_trend = edge_octave_trend(
        curve.freq.as_slice().unwrap_or(&[]),
        curve.spl.as_slice().unwrap_or(&[]),
        false,
    );
    let high_trend = edge_octave_trend(
        curve.freq.as_slice().unwrap_or(&[]),
        curve.spl.as_slice().unwrap_or(&[]),
        true,
    );
    let points_per_decade = 50;

    // Measured phase must survive display extension: downstream deployed-route
    // replay coherently sums these curves, and dropping phase here turns that
    // replay into a panic. Extended points hold the nearest measured edge
    // value (constant extrapolation; slope extrapolation on wrapped phase
    // would invent discontinuities).
    let phase_in = curve
        .phase
        .as_ref()
        .filter(|phase| phase.len() == curve.freq.len());
    let mut phase_vec = phase_in.map(|_| Vec::new());

    let mut freq_vec = Vec::new();
    let mut spl_vec = Vec::new();

    // Prepend log-spaced points from 20 Hz to first measurement frequency
    if meas_min > DISPLAY_MIN_FREQ * 1.05 {
        let log_start = DISPLAY_MIN_FREQ.log10();
        let log_end = meas_min.log10();
        let decades = log_end - log_start;
        let n_points = ((decades * points_per_decade as f64).ceil() as usize).max(1);
        if let (Some(phase), Some(out)) = (phase_in, phase_vec.as_mut()) {
            out.extend(std::iter::repeat_n(phase[0], n_points));
        }
        for i in 0..n_points {
            let t = i as f64 / n_points as f64;
            let f = 10f64.powf(log_start + t * (log_end - log_start));
            freq_vec.push(f);
            spl_vec.push(
                low_trend
                    .map(|(edge_freq, edge_spl, slope)| {
                        extrapolate_spl_with_trend(f, edge_freq, edge_spl, slope)
                    })
                    .unwrap_or(first_spl),
            );
        }
    }

    // Copy original data
    freq_vec.extend(curve.freq.iter());
    spl_vec.extend(curve.spl.iter());
    if let (Some(phase), Some(out)) = (phase_in, phase_vec.as_mut()) {
        out.extend(phase.iter());
    }

    // Append log-spaced points from last measurement frequency to 20 kHz
    if meas_max < DISPLAY_MAX_FREQ * 0.95 {
        let log_start = meas_max.log10();
        let log_end = DISPLAY_MAX_FREQ.log10();
        let decades = log_end - log_start;
        let n_points = ((decades * points_per_decade as f64).ceil() as usize).max(1);
        if let (Some(phase), Some(out)) = (phase_in, phase_vec.as_mut()) {
            out.extend(std::iter::repeat_n(phase[phase.len() - 1], n_points));
        }
        for i in 1..=n_points {
            let t = i as f64 / n_points as f64;
            let f = 10f64
                .powf(log_start + t * (log_end - log_start))
                .min(DISPLAY_MAX_FREQ);
            freq_vec.push(f);
            spl_vec.push(
                high_trend
                    .map(|(edge_freq, edge_spl, slope)| {
                        extrapolate_spl_with_trend(f, edge_freq, edge_spl, slope)
                    })
                    .unwrap_or(last_spl),
            );
        }
    }

    crate::Curve {
        freq: Array1::from(freq_vec),
        spl: Array1::from(spl_vec),
        phase: phase_vec.map(Array1::from),
        ..Default::default()
    }
}

/// Steepest driver trend continued outside the measurement, in dB per octave.
///
/// Acoustic roll-offs reach ~24 dB/octave; anything steeper at the edge is
/// single-bin noise or a filter skirt already represented in DSP, never a
/// driver trend worth continuing for decades.
const MAX_EDGE_TREND_DB_PER_OCTAVE: f64 = 24.0;

/// Minimum log span of the edge-octave fit, in decades (~half an octave).
///
/// Fits spanning less than this fall back to holding the edge value instead
/// of trusting an underdetermined slope.
const MIN_EDGE_TREND_SPAN_DECADES: f64 = 0.15;

/// Least-squares SPL trend over the edge octave, least sensitive to the
/// noisy last bin that dense captures produce.
///
/// Returns `(edge_frequency, edge_spl, slope_db_per_log10)` where the edge
/// is the measured boundary the trend anchors to. Returns `None` when the
/// edge octave holds fewer than two usable points or spans less than
/// [`MIN_EDGE_TREND_SPAN_DECADES`], in which case callers hold the edge
/// value.
fn edge_octave_trend(frequencies: &[f64], levels: &[f64], high: bool) -> Option<(f64, f64, f64)> {
    if frequencies.len() != levels.len() || frequencies.len() < 2 {
        return None;
    }
    let (edge_freq, edge_spl, low, high_freq) = if high {
        let last = frequencies.len() - 1;
        (
            frequencies[last],
            levels[last],
            frequencies[last] / 2.0,
            frequencies[last],
        )
    } else {
        (
            frequencies[0],
            levels[0],
            frequencies[0],
            frequencies[0] * 2.0,
        )
    };
    if !edge_freq.is_finite() || edge_freq <= 0.0 || !edge_spl.is_finite() {
        return None;
    }
    let mut count = 0_u32;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    for (&frequency, &level) in frequencies.iter().zip(levels.iter()) {
        if !frequency.is_finite() || !level.is_finite() || frequency <= 0.0 {
            continue;
        }
        if frequency < low || frequency > high_freq {
            continue;
        }
        let x = frequency.log10();
        count += 1;
        sum_x += x;
        sum_y += level;
        sum_xx += x * x;
        sum_xy += x * level;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
    }
    if count < 2 || max_x - min_x < MIN_EDGE_TREND_SPAN_DECADES {
        return None;
    }
    let denom = f64::from(count) * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return None;
    }
    let slope = (f64::from(count) * sum_xy - sum_x * sum_y) / denom;
    if !slope.is_finite() {
        return None;
    }
    // One octave spans log10(2) decades; convert the dB/octave cap to the
    // dB-per-log10 slope unit used here.
    let cap_db_per_decade = MAX_EDGE_TREND_DB_PER_OCTAVE / 2.0_f64.log10();
    Some((
        edge_freq,
        edge_spl,
        slope.clamp(-cap_db_per_decade, cap_db_per_decade),
    ))
}

fn extrapolate_spl_with_trend(
    freq: f64,
    edge_freq: f64,
    edge_spl: f64,
    slope_db_per_log10: f64,
) -> f64 {
    if freq <= 0.0 || edge_freq <= 0.0 {
        return edge_spl;
    }
    edge_spl + slope_db_per_log10 * (freq.log10() - edge_freq.log10())
}

/// Get a descriptive name for a driver based on its index and total count
pub(super) fn get_driver_name(index: usize, n_drivers: usize) -> String {
    match (n_drivers, index) {
        (2, 0) => "woofer",
        (2, 1) => "tweeter",
        (3, 0) => "woofer",
        (3, 1) => "midrange",
        (3, 2) => "tweeter",
        (4, 0) => "woofer",
        (4, 1) => "lower_midrange",
        (4, 2) => "upper_midrange",
        (4, 3) => "tweeter",
        _ => return format!("driver_{}", index),
    }
    .to_string()
}
