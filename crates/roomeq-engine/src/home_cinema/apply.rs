use super::misc::band_metrics;
pub use super::types::*;
use crate::room_result::ChannelOptimizationResult;
use autoeq_core::{Curve, interpolate_log_space};

pub use roomeq_model::target_tilt::apply_role_target_curve_shape;

pub fn predicted_seat_report(
    seat_index: usize,
    seat_curve: &Curve,
    result: &ChannelOptimizationResult,
    band_hz: (f64, f64),
    primary_seat: usize,
    weight: f64,
    max_deviation_db: f64,
) -> Option<MultiSeatPredictionReport> {
    let predicted =
        apply_result_delta_to_seat(seat_curve, &result.initial_curve, &result.final_curve);
    let (rms, max_abs, min_dev, _) = band_metrics(&predicted, band_hz)?;
    Some(MultiSeatPredictionReport {
        seat_index,
        weight,
        is_primary: seat_index == primary_seat,
        rms_target_error_db: rms,
        max_abs_deviation_db: max_abs,
        pass: max_abs <= max_deviation_db,
        null_risk: min_dev < -max_deviation_db,
    })
}

pub fn apply_result_delta_to_seat(
    seat_curve: &Curve,
    initial: &Curve,
    final_curve: &Curve,
) -> Curve {
    let initial_on_seat = interpolate_log_space(&seat_curve.freq, initial);
    let final_on_seat = interpolate_log_space(&seat_curve.freq, final_curve);
    Curve {
        freq: seat_curve.freq.clone(),
        spl: &seat_curve.spl + &(&final_on_seat.spl - &initial_on_seat.spl),
        phase: seat_curve.phase.clone(),
        ..Default::default()
    }
}
