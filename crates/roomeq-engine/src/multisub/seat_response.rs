//! Coherent measured-seat responses after the selected MSO controls.

use crate::Curve;
use crate::error::{AutoeqError, Result};

/// Preserve each seat for subsequent shared EQ. Measurements are indexed
/// [subwoofer][seat]; no source is broadcast to an unmeasured seat.
/// Only the common measured support is used for this shared-EQ input.
pub fn render_mso_seat_responses(
    measurements: &[Vec<Curve>],
    gains_db: &[f64],
    delays_ms: &[f64],
) -> Result<Vec<Curve>> {
    let invalid = |message: &str| AutoeqError::InvalidMeasurement {
        message: format!("MSO shared-EQ seat response: {message}"),
    };
    let seats = measurements.first().map_or(0, Vec::len);
    if seats == 0
        || gains_db.len() != measurements.len()
        || delays_ms.len() != measurements.len()
        || measurements.iter().any(|sub| sub.len() != seats)
    {
        return Err(invalid("control or seat count mismatch"));
    }
    if gains_db
        .iter()
        .chain(delays_ms)
        .any(|value| !value.is_finite())
    {
        return Err(invalid("nonfinite gain or delay"));
    }
    let curves: Vec<_> = measurements.iter().flatten().collect();
    for curve in &curves {
        if curve.freq.len() < 2
            || curve.spl.len() != curve.freq.len()
            || curve.freq.iter().any(|f| !f.is_finite() || *f <= 0.0)
            || curve
                .freq
                .windows(2)
                .into_iter()
                .any(|pair| pair[1] <= pair[0])
            || curve.spl.iter().any(|spl| !spl.is_finite())
            || !crate::topology::curve_has_usable_phase(curve)
        {
            return Err(invalid("invalid measurement or missing measured phase"));
        }
    }
    let grid = crate::topology::shared_measurement_grid(&curves)
        .filter(|grid| grid.len() >= 2)
        .ok_or_else(|| invalid("insufficient common measured support"))?;
    (0..seats)
        .map(|seat| {
            let controlled: Vec<_> = measurements
                .iter()
                .enumerate()
                .map(|(sub, curves)| {
                    let mut curve =
                        autoeq_core::curve_transforms::interpolate_log_space(&grid, &curves[seat]);
                    curve.spl.mapv_inplace(|spl| spl + gains_db[sub]);
                    let phase = curve.phase.as_mut().expect("measured phase checked above");
                    for (phase, frequency) in phase.iter_mut().zip(grid.iter()) {
                        *phase -= 360.0 * frequency * delays_ms[sub] / 1000.0;
                    }
                    curve
                })
                .collect();
            let response =
                crate::topology::complex_sum_mains(&controlled.iter().collect::<Vec<_>>());
            if response
                .spl
                .iter()
                .chain(response.phase.as_ref().unwrap())
                .any(|v| !v.is_finite())
            {
                return Err(invalid("nonfinite controlled response"));
            }
            Ok(response)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn curve(level: f64) -> Curve {
        Curve {
            freq: array![50.0, 100.0, 200.0],
            spl: array![level, level, level],
            phase: Some(array![0.0, 0.0, 0.0]),
            ..Default::default()
        }
    }

    #[test]
    fn mso_seat_renderer_retains_seats_and_selected_controls() {
        let measurements = vec![vec![curve(0.0), curve(20.0)], vec![curve(0.0), curve(0.0)]];
        let rendered =
            render_mso_seat_responses(&measurements, &[0.0, -6.020599913279624], &[0.0, 5.0])
                .unwrap();
        assert_eq!(rendered.len(), 2);
        // At 100 Hz, 5 ms is half a cycle: second source subtracts 0.5.
        assert!((rendered[0].spl[1] - 20.0 * 0.5_f64.log10()).abs() < 1e-10);
        assert!((rendered[1].spl[1] - 20.0 * 9.5_f64.log10()).abs() < 1e-10);
        assert_eq!(rendered[0].freq, rendered[1].freq);
    }

    #[test]
    fn mso_seat_renderer_rejects_missing_seats_phase_and_controls() {
        assert!(
            render_mso_seat_responses(
                &[vec![curve(0.0), curve(1.0)], vec![curve(0.0)]],
                &[0.0; 2],
                &[0.0; 2]
            )
            .is_err()
        );
        let mut missing = curve(0.0);
        missing.phase = None;
        assert!(render_mso_seat_responses(&[vec![missing]], &[0.0], &[0.0]).is_err());
        assert!(render_mso_seat_responses(&[vec![curve(0.0)]], &[f64::NAN], &[0.0]).is_err());
    }

    #[test]
    fn mso_seat_renderer_does_not_extrapolate_support() {
        let mut limited = curve(0.0);
        limited.freq = array![75.0, 100.0, 150.0];
        let rendered =
            render_mso_seat_responses(&[vec![curve(0.0)], vec![limited]], &[0.0; 2], &[0.0; 2])
                .unwrap();
        assert_eq!(rendered[0].freq, array![75.0, 100.0, 150.0]);
    }
}
