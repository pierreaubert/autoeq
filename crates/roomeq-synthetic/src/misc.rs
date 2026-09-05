use crate::Curve;
use math_audio_iir_fir::Biquad;

/// Apply known biquad filters to a curve (simulates room modes).
///
/// Computes the combined dB response of the given filters at each frequency
/// point and adds it to the SPL. When the curve carries phase, the filters'
/// full complex response is applied as well: each filter's unwrapped phase
/// (degrees, unwrapped along the ascending grid so all-pass-like 360°
/// excursions stay continuous) is added to the existing phase. A curve
/// without phase keeps `phase: None` — a magnitude-only perturbation, not a
/// physical oracle for phase-sensitive QA.
pub fn apply_known_eq(curve: &Curve, filters: &[Biquad], _sample_rate: f64) -> Curve {
    let mut spl = curve.spl.clone();

    for filter in filters {
        let response = filter.np_log_result(&curve.freq);
        spl += &response;
    }

    let phase = curve.phase.as_ref().map(|existing| {
        let mut filter_phase = ndarray::Array1::<f64>::zeros(curve.freq.len());
        for filter in filters {
            let mut previous_raw = 0.0;
            let mut cumulative = 0.0;
            let mut first = true;
            for (index, &frequency) in curve.freq.iter().enumerate() {
                let raw = filter.complex_response(frequency).arg().to_degrees();
                if !raw.is_finite() {
                    continue;
                }
                if first {
                    cumulative = raw;
                    first = false;
                } else {
                    let mut delta = raw - previous_raw;
                    while delta > 180.0 {
                        delta -= 360.0;
                    }
                    while delta <= -180.0 {
                        delta += 360.0;
                    }
                    cumulative += delta;
                }
                previous_raw = raw;
                filter_phase[index] += cumulative;
            }
        }
        existing + &filter_phase
    });

    Curve {
        freq: curve.freq.clone(),
        spl,
        phase,
        ..Default::default()
    }
}

/// Xorshift64 PRNG — simple, fast, deterministic.
pub(super) fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}
