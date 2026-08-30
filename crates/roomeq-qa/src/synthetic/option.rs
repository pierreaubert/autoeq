use roomeq_model::{
    DecomposedCorrectionSerdeConfig, ExcursionProtectionConfig, FirConfig, MultiMeasurementConfig,
    MultiMeasurementStrategy, PreRingingSerdeConfig, ProcessingMode, RoomConfig,
    SchroederSplitConfig, SpatialRobustnessSerdeConfig, TargetResponseConfig,
};

pub(super) fn isolate_schroeder_split_from_multi_measurement(
    config: &mut RoomConfig,
    option_names: &[&str],
) {
    if option_names.contains(&"schroeder") {
        config.optimizer.multi_measurement = None;
    }
}

pub(super) fn option_psychoacoustic(config: &mut RoomConfig) {
    config.optimizer.psychoacoustic = true;
}

pub(super) fn option_asymmetric(config: &mut RoomConfig) {
    config.optimizer.asymmetric_loss = true;
}

pub(super) fn option_broadband(config: &mut RoomConfig) {
    let tr = config
        .optimizer
        .target_response
        .get_or_insert_with(TargetResponseConfig::default);
    tr.broadband_precorrection = true;
}

pub(super) fn option_excursion(config: &mut RoomConfig) {
    config.optimizer.excursion_protection = Some(ExcursionProtectionConfig {
        enabled: true,
        ..Default::default()
    });
}

pub(super) fn option_schroeder(config: &mut RoomConfig) {
    config.optimizer.schroeder_split = Some(SchroederSplitConfig {
        enabled: true,
        ..Default::default()
    });
}

pub(super) fn option_spatial_robustness(config: &mut RoomConfig) {
    config.optimizer.multi_measurement = Some(MultiMeasurementConfig {
        strategy: MultiMeasurementStrategy::SpatialRobustness,
        spatial_robustness: Some(SpatialRobustnessSerdeConfig {
            variance_threshold_db: 3.0,
            transition_width_db: 2.0,
            min_correction_depth: 0.1,
            mask_smoothing_octaves: 1.0 / 6.0,
        }),
        ..Default::default()
    });
}

pub(super) fn option_pre_ringing(config: &mut RoomConfig) {
    // Enable FIR with pre-ringing control
    config.optimizer.processing_mode = ProcessingMode::PhaseLinear;
    if config.optimizer.fir.is_none() {
        config.optimizer.fir = Some(FirConfig {
            taps: 2048,
            phase: "kirkeby".to_string(),
            correct_excess_phase: false,
            phase_smoothing: 0.167,
            pre_ringing: Some(PreRingingSerdeConfig {
                threshold_db: -30.0,
                max_time_s: 0.005,
            }),
            max_boost_db: None,
        });
    } else if let Some(ref mut fir) = config.optimizer.fir {
        fir.pre_ringing = Some(PreRingingSerdeConfig {
            threshold_db: -30.0,
            max_time_s: 0.005,
        });
    }
}

pub(super) fn option_decomposed_correction(config: &mut RoomConfig) {
    config.optimizer.decomposed_correction = Some(DecomposedCorrectionSerdeConfig {
        enabled: true,
        schroeder_freq: 200.0,
        room_dimensions: None,
        min_mode_q: 3.0,
        min_mode_prominence_db: 3.0,
        mode_correction_weight: 1.0,
        early_reflection_weight: 0.3,
        steady_state_weight: 0.5,
        ..Default::default()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schroeder_option_clears_combined_multi_measurement_mode() {
        let mut config = RoomConfig::default();
        config.optimizer.multi_measurement = Some(MultiMeasurementConfig::default());

        isolate_schroeder_split_from_multi_measurement(
            &mut config,
            &["schroeder", "spatial_robustness"],
        );

        assert!(config.optimizer.multi_measurement.is_none());
    }
}
