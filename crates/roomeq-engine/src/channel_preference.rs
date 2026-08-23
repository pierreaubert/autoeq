use math_audio_iir_fir::{Biquad, BiquadFilterType};
use roomeq_model::{HomeCinemaRole, RoomConfig, TargetShape};

/// Build the separately bypassable listener/content preference layer.
pub(crate) fn build_preference_filters(
    channel_name: &str,
    room_config: &RoomConfig,
    sample_rate: f64,
) -> Vec<Biquad> {
    let Some(target) = room_config.optimizer.target_response.as_ref() else {
        return Vec::new();
    };
    let mut filters = crate::cea2034::generate_preference_filters(&target.preference, sample_rate);
    if target.shape == TargetShape::Harman {
        push_tilt(&mut filters, -0.8, sample_rate);
    }
    let Some(role_targets) = target.role_targets.as_ref().filter(|config| config.enabled) else {
        return filters;
    };

    let role = roomeq_model::home_cinema::role_for_channel(channel_name);
    let (slope, treble, bass) = match role {
        HomeCinemaRole::FrontLeft | HomeCinemaRole::FrontRight => {
            (role_targets.front_slope_offset_db_per_octave, 0.0, 0.0)
        }
        HomeCinemaRole::Center => (
            role_targets.center_slope_offset_db_per_octave,
            role_targets.center_treble_shelf_db,
            0.0,
        ),
        HomeCinemaRole::SideSurroundLeft
        | HomeCinemaRole::SideSurroundRight
        | HomeCinemaRole::RearSurroundLeft
        | HomeCinemaRole::RearSurroundRight
        | HomeCinemaRole::WideLeft
        | HomeCinemaRole::WideRight => (
            role_targets.surround_slope_offset_db_per_octave,
            role_targets.surround_treble_shelf_db,
            0.0,
        ),
        HomeCinemaRole::TopFrontLeft
        | HomeCinemaRole::TopFrontRight
        | HomeCinemaRole::TopMiddleLeft
        | HomeCinemaRole::TopMiddleRight
        | HomeCinemaRole::TopRearLeft
        | HomeCinemaRole::TopRearRight => (
            role_targets.height_slope_offset_db_per_octave,
            role_targets.height_treble_shelf_db,
            0.0,
        ),
        HomeCinemaRole::Subwoofer => (
            role_targets.subwoofer_slope_offset_db_per_octave,
            0.0,
            role_targets.subwoofer_bass_shelf_db,
        ),
        HomeCinemaRole::Lfe => (
            role_targets.lfe_slope_offset_db_per_octave,
            0.0,
            role_targets.lfe_bass_shelf_db,
        ),
        HomeCinemaRole::Unknown => (0.0, 0.0, 0.0),
    };
    push_tilt(&mut filters, slope, sample_rate);
    push_shelf(
        &mut filters,
        BiquadFilterType::Highshelf,
        4_000.0,
        treble,
        sample_rate,
    );
    push_shelf(
        &mut filters,
        BiquadFilterType::Lowshelf,
        80.0,
        bass,
        sample_rate,
    );

    if role == HomeCinemaRole::Center && role_targets.center_dialog_boost_db.abs() > 1e-3 {
        let center =
            (role_targets.center_dialog_low_hz * role_targets.center_dialog_high_hz).sqrt();
        let bandwidth = role_targets.center_dialog_high_hz - role_targets.center_dialog_low_hz;
        filters.push(Biquad::new(
            BiquadFilterType::Peak,
            center,
            sample_rate,
            (center / bandwidth.max(1.0)).clamp(0.2, 4.0),
            role_targets.center_dialog_boost_db,
        ));
    }
    if !role.is_sub_or_lfe() && role_targets.cinema_x_curve_enabled {
        let start = role_targets.cinema_x_curve_start_hz.max(20.0);
        push_shelf(
            &mut filters,
            BiquadFilterType::Highshelf,
            start,
            role_targets.cinema_x_curve_db_per_octave * (20_000.0 / start).log2(),
            sample_rate,
        );
    }
    if !role.is_sub_or_lfe()
        && let Some(distance) = role_targets.listening_distance_m
        && distance > role_targets.cinema_reference_distance_m
        && role_targets.cinema_reference_distance_m > 0.0
    {
        push_shelf(
            &mut filters,
            BiquadFilterType::Highshelf,
            role_targets.cinema_x_curve_start_hz,
            -role_targets.distance_treble_rolloff_db_per_doubling.abs()
                * (distance / role_targets.cinema_reference_distance_m).log2(),
            sample_rate,
        );
    }
    filters
}

fn push_tilt(filters: &mut Vec<Biquad>, slope_db_per_octave: f64, sample_rate: f64) {
    // A complementary low/high-shelf pair keeps the 1 kHz reference near
    // unity while approximating the configured ten-octave house-curve tilt.
    let edge_gain = slope_db_per_octave * 5.0;
    push_shelf(
        filters,
        BiquadFilterType::Lowshelf,
        1_000.0,
        -edge_gain,
        sample_rate,
    );
    push_shelf(
        filters,
        BiquadFilterType::Highshelf,
        1_000.0,
        edge_gain,
        sample_rate,
    );
}

fn push_shelf(
    filters: &mut Vec<Biquad>,
    filter_type: BiquadFilterType,
    frequency: f64,
    gain_db: f64,
    sample_rate: f64,
) {
    if gain_db.abs() > 1e-3 {
        filters.push(Biquad::new(
            filter_type,
            frequency,
            sample_rate,
            0.707,
            gain_db,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roomeq_model::{RoleTargetConfig, TargetResponseConfig, UserPreference};

    #[test]
    fn harman_house_curve_is_realized_only_in_preference_layer() {
        let mut config = RoomConfig::default();
        config.optimizer.target_response = Some(TargetResponseConfig {
            shape: TargetShape::Harman,
            ..TargetResponseConfig::default()
        });
        let harman = build_preference_filters("L", &config, 48_000.0);
        assert_eq!(harman.len(), 2);

        config.optimizer.target_response.as_mut().unwrap().shape = TargetShape::Flat;
        assert!(build_preference_filters("L", &config, 48_000.0).is_empty());
    }

    #[test]
    fn role_and_user_preferences_are_output_filters() {
        let mut config = RoomConfig::default();
        config.optimizer.target_response = Some(TargetResponseConfig {
            preference: UserPreference {
                bass_shelf_db: 2.0,
                ..UserPreference::default()
            },
            role_targets: Some(RoleTargetConfig {
                center_dialog_boost_db: 3.0,
                ..RoleTargetConfig::default()
            }),
            ..TargetResponseConfig::default()
        });
        let center = build_preference_filters("C", &config, 48_000.0);
        let left = build_preference_filters("L", &config, 48_000.0);
        assert_eq!(center.len(), left.len() + 1);
        assert!(!left.is_empty());
    }

    #[test]
    fn every_role_and_cinema_distance_preference_is_realized() {
        let mut config = RoomConfig::default();
        config.optimizer.target_response = Some(TargetResponseConfig {
            role_targets: Some(RoleTargetConfig {
                surround_slope_offset_db_per_octave: -0.2,
                height_slope_offset_db_per_octave: -0.3,
                subwoofer_slope_offset_db_per_octave: 0.2,
                lfe_slope_offset_db_per_octave: 0.3,
                surround_treble_shelf_db: -1.0,
                height_treble_shelf_db: -1.5,
                subwoofer_bass_shelf_db: 1.0,
                lfe_bass_shelf_db: 1.5,
                cinema_x_curve_enabled: true,
                cinema_x_curve_db_per_octave: -0.5,
                cinema_x_curve_start_hz: 2_000.0,
                listening_distance_m: Some(6.0),
                cinema_reference_distance_m: 3.0,
                distance_treble_rolloff_db_per_doubling: 1.0,
                ..RoleTargetConfig::default()
            }),
            ..TargetResponseConfig::default()
        });

        // Surround/height channels get tilt, their role shelf, X-curve, and
        // distance rolloff. Sub/LFE channels get tilt and their bass shelf but
        // intentionally exclude the two cinema treble contours.
        assert_eq!(build_preference_filters("SL", &config, 48_000.0).len(), 5);
        assert_eq!(build_preference_filters("TFL", &config, 48_000.0).len(), 5);
        assert_eq!(build_preference_filters("Sub", &config, 48_000.0).len(), 3);
        assert_eq!(build_preference_filters("LFE", &config, 48_000.0).len(), 3);

        // Unknown non-bass roles have no role shelf but still receive the two
        // explicitly requested cinema treble contours.
        assert_eq!(build_preference_filters("Aux", &config, 48_000.0).len(), 2);
    }
}
