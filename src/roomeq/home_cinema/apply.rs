use super::super::types::TargetResponseConfig;
use super::misc::band_metrics;
use super::role::role_for_channel;
pub use super::types::*;
use crate::Curve;

pub(super) fn predicted_seat_report(
    seat_index: usize,
    seat_curve: &Curve,
    result: &super::super::optimize::ChannelOptimizationResult,
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

pub(super) fn apply_result_delta_to_seat(
    seat_curve: &Curve,
    initial: &Curve,
    final_curve: &Curve,
) -> Curve {
    let initial_on_seat = crate::read::interpolate_log_space(&seat_curve.freq, initial);
    let final_on_seat = crate::read::interpolate_log_space(&seat_curve.freq, final_curve);
    Curve {
        freq: seat_curve.freq.clone(),
        spl: &seat_curve.spl + &(&final_on_seat.spl - &initial_on_seat.spl),
        phase: seat_curve.phase.clone(),
        ..Default::default()
    }
}

pub fn apply_role_target_curve_shape(
    channel_name: &str,
    target_curve: &mut Curve,
    target: &TargetResponseConfig,
) {
    let Some(role_targets) = target.role_targets.as_ref().filter(|cfg| cfg.enabled) else {
        return;
    };
    let role = role_for_channel(channel_name);

    if role == HomeCinemaRole::Center && role_targets.center_dialog_boost_db.abs() > 0.001 {
        apply_log_band_emphasis(
            target_curve,
            role_targets.center_dialog_low_hz,
            role_targets.center_dialog_high_hz,
            role_targets.center_dialog_boost_db,
        );
    }

    if role_targets.cinema_x_curve_enabled
        && role_targets.cinema_x_curve_db_per_octave.abs() > 0.001
    {
        apply_high_frequency_slope(
            target_curve,
            role_targets.cinema_x_curve_start_hz,
            role_targets.cinema_x_curve_db_per_octave,
        );
    }

    if let Some(distance_m) = role_targets.listening_distance_m {
        let ref_m = role_targets.cinema_reference_distance_m;
        if distance_m > ref_m
            && ref_m > 0.0
            && role_targets.distance_treble_rolloff_db_per_doubling.abs() > 0.001
        {
            let distance_doublings = (distance_m / ref_m).log2();
            apply_high_frequency_slope(
                target_curve,
                role_targets.cinema_x_curve_start_hz,
                -role_targets.distance_treble_rolloff_db_per_doubling.abs() * distance_doublings,
            );
        }
    }
}

fn apply_log_band_emphasis(target_curve: &mut Curve, low_hz: f64, high_hz: f64, gain_db: f64) {
    if !(low_hz > 0.0 && high_hz > low_hz) {
        return;
    }
    let center_hz = (low_hz * high_hz).sqrt();
    let half_width_oct = (high_hz / low_hz).log2() / 2.0;
    if half_width_oct <= 0.0 {
        return;
    }

    for (freq, spl) in target_curve.freq.iter().zip(target_curve.spl.iter_mut()) {
        let distance_oct = (*freq / center_hz).max(1e-9).log2().abs();
        if distance_oct <= half_width_oct {
            let weight = 0.5 * (1.0 + (std::f64::consts::PI * distance_oct / half_width_oct).cos());
            *spl += gain_db * weight;
        }
    }
}

fn apply_high_frequency_slope(target_curve: &mut Curve, start_hz: f64, slope_db_per_octave: f64) {
    if start_hz <= 0.0 {
        return;
    }
    for (freq, spl) in target_curve.freq.iter().zip(target_curve.spl.iter_mut()) {
        if *freq > start_hz {
            *spl += slope_db_per_octave * (*freq / start_hz).log2();
        }
    }
}
