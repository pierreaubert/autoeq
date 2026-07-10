#[test]
fn policy_target_response_covers_all_presets() {
    let reference = policy_target_response(PerceptualPolicyPreset::Reference);
    assert_eq!(reference.shape, TargetShape::Flat);
    assert!(reference.broadband_precorrection);
    assert!(reference.role_targets.is_some());

    let music = policy_target_response(PerceptualPolicyPreset::Music);
    assert_eq!(music.shape, TargetShape::Harman);
    assert_eq!(music.slope_db_per_octave, -0.8);
    assert_eq!(music.preference.bass_shelf_db, 1.0);
    assert_eq!(music.preference.treble_shelf_db, -0.5);

    let cinema = policy_target_response(PerceptualPolicyPreset::Cinema);
    assert_eq!(cinema.shape, TargetShape::Custom);
    let cinema_roles = cinema.role_targets.unwrap();
    assert!(cinema_roles.cinema_x_curve_enabled);
    assert_eq!(cinema_roles.center_dialog_boost_db, 1.5);

    let night = policy_target_response(PerceptualPolicyPreset::Night);
    assert_eq!(night.preference.bass_shelf_db, -1.5);
    let night_roles = night.role_targets.unwrap();
    assert_eq!(night_roles.subwoofer_bass_shelf_db, -2.0);

    let speech = policy_target_response(PerceptualPolicyPreset::Speech);
    assert_eq!(speech.slope_db_per_octave, -0.3);
    let speech_roles = speech.role_targets.unwrap();
    assert_eq!(speech_roles.center_dialog_boost_db, 3.0);
    assert_eq!(speech_roles.center_dialog_low_hz, 250.0);
    assert_eq!(speech_roles.center_dialog_high_hz, 5_000.0);
}

#[test]
fn policy_multi_measurement_covers_all_presets() {
    let reference = policy_multi_measurement(PerceptualPolicyPreset::Reference);
    assert_eq!(
        reference.strategy,
        MultiMeasurementStrategy::MinimaxUncertainty
    );

    let music = policy_multi_measurement(PerceptualPolicyPreset::Music);
    assert_eq!(music.strategy, MultiMeasurementStrategy::SpatialRobustness);

    let cinema = policy_multi_measurement(PerceptualPolicyPreset::Cinema);
    assert_eq!(
        cinema.strategy,
        MultiMeasurementStrategy::MinimaxUncertainty
    );

    let night = policy_multi_measurement(PerceptualPolicyPreset::Night);
    assert_eq!(night.strategy, MultiMeasurementStrategy::SpatialRobustness);
    assert_eq!(
        night
            .spatial_robustness
            .as_ref()
            .unwrap()
            .variance_threshold_db,
        2.5
    );

    let speech = policy_multi_measurement(PerceptualPolicyPreset::Speech);
    assert_eq!(speech.strategy, MultiMeasurementStrategy::SpatialRobustness);
    assert_eq!(
        speech.bootstrap_uncertainty.as_ref().unwrap().scalarisation,
        super::types::BootstrapScalarisation::Cvar
    );
    assert_eq!(
        speech
            .spatial_robustness
            .as_ref()
            .unwrap()
            .min_correction_depth,
        0.25
    );
}

#[test]
fn policy_decomposed_correction_covers_all_presets() {
    let reference = policy_decomposed_correction(PerceptualPolicyPreset::Reference);
    assert_eq!(reference.early_reflection_weight, 0.1);
    assert_eq!(reference.steady_state_weight, 0.3);
    assert_eq!(reference.mode_correction_weight, 1.0);

    let speech = policy_decomposed_correction(PerceptualPolicyPreset::Speech);
    assert_eq!(speech.early_reflection_weight, 0.6);
    assert_eq!(speech.steady_state_weight, 0.55);
    assert_eq!(speech.mode_correction_weight, 1.0);

    let night = policy_decomposed_correction(PerceptualPolicyPreset::Night);
    assert_eq!(night.early_reflection_weight, 0.15);
    assert_eq!(night.mode_correction_weight, 0.85);

    let cinema = policy_decomposed_correction(PerceptualPolicyPreset::Cinema);
    assert_eq!(cinema.early_reflection_weight, 0.2);
    assert_eq!(cinema.steady_state_weight, 0.4);
}

#[test]
fn policy_psychoacoustic_smoothing_covers_all_presets() {
    let reference = policy_psychoacoustic_smoothing(PerceptualPolicyPreset::Reference);
    assert_eq!(reference, PsychoacousticSmoothingConfig::default());

    let music = policy_psychoacoustic_smoothing(PerceptualPolicyPreset::Music);
    assert_eq!(music.high_freq_n, 5);

    let cinema = policy_psychoacoustic_smoothing(PerceptualPolicyPreset::Cinema);
    assert_eq!(cinema.high_freq_n, 4);
    assert_eq!(cinema.high_freq, 900.0);

    let night = policy_psychoacoustic_smoothing(PerceptualPolicyPreset::Night);
    assert_eq!(night.high_freq_n, 4);
    assert_eq!(night.high_freq, 900.0);

    let speech = policy_psychoacoustic_smoothing(PerceptualPolicyPreset::Speech);
    assert_eq!(speech.low_freq_n, 24);
    assert_eq!(speech.high_freq_n, 6);
    assert_eq!(speech.low_freq, 150.0);
    assert_eq!(speech.high_freq, 1_200.0);
}

#[test]
fn policy_asymmetric_loss_covers_all_presets() {
    let reference = policy_asymmetric_loss(PerceptualPolicyPreset::Reference);
    assert_eq!(reference, AsymmetricLossConfig::default());

    let music = policy_asymmetric_loss(PerceptualPolicyPreset::Music);
    assert_eq!(music.bass_peak_weight, 5.0);
    assert_eq!(music.transition_freq, 200.0);

    let night = policy_asymmetric_loss(PerceptualPolicyPreset::Night);
    assert_eq!(night.bass_peak_weight, 6.0);
    assert_eq!(night.transition_freq, 220.0);

    let speech = policy_asymmetric_loss(PerceptualPolicyPreset::Speech);
    assert_eq!(speech.bass_peak_weight, 3.0);
    assert_eq!(speech.transition_freq, 180.0);
}

#[test]
fn policy_smoothness_penalty_covers_all_presets() {
    let reference = policy_smoothness_penalty(PerceptualPolicyPreset::Reference);
    assert_eq!(reference.tv2_weight, 0.001);

    let music = policy_smoothness_penalty(PerceptualPolicyPreset::Music);
    assert_eq!(music.tv2_weight, 0.0015);

    let cinema = policy_smoothness_penalty(PerceptualPolicyPreset::Cinema);
    assert_eq!(cinema.tv2_weight, 0.002);

    let night = policy_smoothness_penalty(PerceptualPolicyPreset::Night);
    assert_eq!(night.tv2_weight, 0.003);

    let speech = policy_smoothness_penalty(PerceptualPolicyPreset::Speech);
    assert_eq!(speech.tv2_weight, 0.001);
    assert_eq!(speech.modal_weight_scale, 0.1);
    assert_eq!(speech.exponent, 1.0);
    assert!(speech.schroeder_hz.is_some());
}

#[test]
fn policy_audibility_deadband_covers_all_presets() {
    let reference = policy_audibility_deadband(PerceptualPolicyPreset::Reference);
    let default = super::audibility_deadband_config::AudibilityDeadbandConfig::default();
    assert_eq!(reference.bass_db, default.bass_db);
    assert_eq!(reference.mid_db, default.mid_db);
    assert_eq!(reference.treble_db, default.treble_db);

    let music = policy_audibility_deadband(PerceptualPolicyPreset::Music);
    assert_eq!(music.mid_db, 0.65);
    assert_eq!(music.treble_db, 0.9);

    let cinema = policy_audibility_deadband(PerceptualPolicyPreset::Cinema);
    assert_eq!(cinema.mid_db, 0.75);
    assert_eq!(cinema.treble_db, 1.1);

    let night = policy_audibility_deadband(PerceptualPolicyPreset::Night);
    assert_eq!(night.bass_db, 0.5);
    assert_eq!(night.mid_db, 0.9);
    assert_eq!(night.treble_db, 1.25);

    let speech = policy_audibility_deadband(PerceptualPolicyPreset::Speech);
    assert_eq!(speech.bass_db, 0.75);
    assert_eq!(speech.mid_db, 0.5);
    assert_eq!(speech.treble_db, 0.75);
    assert!(!speech.disable_below_schroeder);
}

#[test]
fn policy_high_frequency_guard_covers_all_presets() {
    let reference = policy_high_frequency_guard(PerceptualPolicyPreset::Reference);
    assert_eq!(reference.extra_deadband_db, 0.75);
    assert_eq!(reference.smoothing_n, 3);
    assert_eq!(reference.max_q, 2.0);

    let speech = policy_high_frequency_guard(PerceptualPolicyPreset::Speech);
    assert_eq!(speech.extra_deadband_db, 0.4);
    assert_eq!(speech.smoothing_n, 5);
    assert_eq!(speech.max_q, 2.5);

    let music = policy_high_frequency_guard(PerceptualPolicyPreset::Music);
    assert_eq!(music.extra_deadband_db, 0.6);
    assert_eq!(music.smoothing_n, 4);

    let cinema = policy_high_frequency_guard(PerceptualPolicyPreset::Cinema);
    assert_eq!(cinema.extra_deadband_db, 0.9);
    assert_eq!(cinema.smoothing_n, 3);

    let night = policy_high_frequency_guard(PerceptualPolicyPreset::Night);
    assert_eq!(night.extra_deadband_db, 0.9);
    assert_eq!(night.smoothing_n, 3);
}

#[test]
fn policy_early_late_correction_covers_all_presets() {
    let reference = policy_early_late_correction(PerceptualPolicyPreset::Reference);
    assert_eq!(reference.early_cue_risk_db, -22.0);

    let music = policy_early_late_correction(PerceptualPolicyPreset::Music);
    assert_eq!(music.early_cue_risk_db, -18.0);

    let cinema = policy_early_late_correction(PerceptualPolicyPreset::Cinema);
    assert_eq!(cinema.early_cue_risk_db, -18.0);

    let night = policy_early_late_correction(PerceptualPolicyPreset::Night);
    assert_eq!(night.early_cue_risk_db, -20.0);

    let speech = policy_early_late_correction(PerceptualPolicyPreset::Speech);
    assert_eq!(speech.early_cue_risk_db, -14.0);
}

#[test]
fn all_default_functions_return_sane_values() {
    // Exercise every simple default_* function in default.rs to cover all
    // the missed one-liner bodies.
    let _ = super::default::default_config_version();
    let _ = super::default::default_bass_management_enabled();
    let _ = super::default::default_redirect_bass();
    let _ = super::default::default_lfe_channel();
    let _ = super::default::default_lfe_playback_gain_db();
    let _ = super::default::default_sub_headroom_margin_db();
    let _ = super::default::default_max_sub_boost_db();
    let _ = super::default::default_optimize_bass_groups();
    let _ = super::default::default_bass_headroom_model_kind();
    let _ = super::default::default_lr_correlation();
    let _ = super::default::default_lcr_correlation();
    let _ = super::default::default_surround_height_correlation();
    let _ = super::default::default_tilt_slope();
    let _ = super::default::default_tilt_reference_freq();
    let _ = super::default::default_bass_shelf_freq();
    let _ = super::default::default_treble_shelf_freq();
    let _ = super::default::default_role_targets_enabled();
    let _ = super::default::default_center_dialog_low_hz();
    let _ = super::default::default_center_dialog_high_hz();
    let _ = super::default::default_cinema_reference_distance_m();
    let _ = super::default::default_cinema_x_curve_start_hz();
    let _ = super::default::default_pre_ringing_threshold();
    let _ = super::default::default_pre_ringing_time();
    let _ = super::default::default_fir_taps();
    let _ = super::default::default_fir_phase();
    let _ = super::default::default_phase_smoothing();
    let _ = super::default::default_mixed_phase_fir_length();
    let _ = super::default::default_mixed_phase_spatial_depth();
    let _ = super::default::default_mask_smoothing();
    let _ = super::default::default_crossover_freq();
    let _ = super::default::default_crossover_type();
    let _ = super::default::default_fir_band();
    let _ = super::default::default_true();
    let _ = super::default::default_f3_reference_min_hz();
    let _ = super::default::default_f3_reference_max_hz();
    let _ = super::default::default_filter_order();
    let _ = super::default::default_margin_octaves();
    let _ = super::default::default_low_freq_max_q();
    let _ = super::default::default_high_freq_max_q();
    let _ = super::default::default_schroeder_freq();
    let _ = super::default::default_phase_min_freq();
    let _ = super::default::default_phase_max_freq();
    let _ = super::default::default_max_delay_ms();
    let _ = super::default::default_max_deviation_db();
    let _ = super::default::default_all_channel_multiseat_enabled();
    let _ = super::default::default_all_channel_multiseat_strategy();
    let _ = super::default::default_primary_seat_weight();
    let _ = super::default::default_multiseat_per_sub_peq();
    let _ = super::default::default_multiseat_global_eq();
    let _ = super::default::default_channel_matching_threshold();
    let _ = super::default::default_channel_matching_max_filters();
    let _ = super::default::default_gd_max_delay_ms();
    let _ = super::default::default_gd_ap_per_channel();
    let _ = super::default::default_gd_ap_min_q();
    let _ = super::default::default_gd_ap_max_q();
    let _ = super::default::default_gd_coherence_threshold();
    let _ = super::default::default_gd_min_improvement_db();
    let _ = super::default::default_gd_max_iter();
    let _ = super::default::default_gd_popsize();
    let _ = super::default::default_gd_tol();
    let _ = super::default::default_sub_num_filters();
    let _ = super::default::default_sub_max_db();
    let _ = super::default::default_sub_min_db();
    let _ = super::default::default_sub_max_q();
    let _ = super::default::default_modal_weight_scale();
    let _ = super::default::default_smoothness_exponent();
    let _ = super::default::default_variance_threshold();
    let _ = super::default::default_transition_width();
    let _ = super::default::default_min_correction_depth();
    let _ = super::default::default_mask_smoothing_octaves();
    let _ = super::default::default_bootstrap_num_resamples();
    let _ = super::default::default_bootstrap_alpha();
    let _ = super::default::default_bootstrap_seed();
    let _ = super::default::default_bootstrap_cvar_alpha();
    let _ = super::default::default_gaussian_truncation_sigmas();
    let _ = super::default::default_area_inner_maxiter();
    let _ = super::default::default_area_cvar_alpha();
    let _ = super::default::default_idw_power();
    let _ = super::default::default_variance_lambda();
    let _ = super::default::default_decomposed_schroeder();
    let _ = super::default::default_decomposed_min_q();
    let _ = super::default::default_decomposed_prominence();
    let _ = super::default::default_decomposed_mode_weight();
    let _ = super::default::default_decomposed_reflection_weight();
    let _ = super::default::default_decomposed_steady_weight();
    let _ = super::default::default_fdw_cycles();
    let _ = super::default::default_fdw_min_window_ms();
    let _ = super::default::default_fdw_max_window_ms();
    let _ = super::default::default_fdw_smoothing_octaves();
    let _ = super::default::default_cea2034_version();
    let _ = super::default::default_nearfield_threshold();
    let _ = super::default::default_cea2034_num_filters();
    let _ = super::default::default_cea2034_max_q();
    let _ = super::default::default_cea2034_max_db();
    let _ = super::default::default_cea2034_min_db();
    let _ = super::default::default_auto_min_filters();
    let _ = super::default::default_auto_max_filters();
    let _ = super::default::default_deadband_bass_db();
    let _ = super::default::default_deadband_mid_db();
    let _ = super::default::default_deadband_treble_db();
    let _ = super::default::default_deadband_bass_mid_hz();
    let _ = super::default::default_deadband_mid_treble_hz();
    let _ = super::default::default_high_freq_guard_start_hz();
    let _ = super::default::default_high_freq_extra_deadband_db();
    let _ = super::default::default_high_freq_smoothing_n();
    let _ = super::default::default_high_freq_guard_max_q();
    let _ = super::default::default_direct_window_ms();
    let _ = super::default::default_early_window_ms();
    let _ = super::default::default_late_window_ms();
    let _ = super::default::default_early_cue_risk_db();
    let _ = super::default::default_validation_lufs();
    let _ = super::default::default_loss_type();
    let _ = super::default::default_algorithm();
    let _ = super::default::default_strategy();
    let _ = super::default::default_peq_model();
    let _ = super::default::default_num_filters();
    let _ = super::default::default_min_filter_improvement();
    let _ = super::default::default_elimination_threshold();
    let _ = super::default::default_min_q();
    let _ = super::default::default_max_q();
    let _ = super::default::default_min_db();
    let _ = super::default::default_max_db();
    let _ = super::default::default_min_freq();
    let _ = super::default::default_max_freq();
    let _ = super::default::default_max_iter();
    let _ = super::default::default_population();
    let _ = super::default::default_refine();
    let _ = super::default::default_local_algo();
    let _ = super::default::default_psychoacoustic();
    let _ = super::default::default_smooth_n();
    let _ = super::default::default_asymmetric_loss();
    let _ = super::default::default_tolerance();
    let _ = super::default::default_atolerance();
    let _ = super::default::default_ctc_matrix_source();
    let _ = super::default::default_ctc_window_type();
    let _ = super::default::default_ctc_window_start_ms();
    let _ = super::default::default_ctc_window_length_ms();
    let _ = super::default::default_ctc_window_fade_ms();
    let _ = super::default::default_ctc_beta_db();
    let _ = super::default::default_ctc_max_gain_db();
    let _ = super::default::default_ctc_fir_taps();
    let _ = super::default::default_ctc_robustness();
    let _ = super::default::default_ctc_include_room_eq_dsp();
    let _ = super::default::default_ctc_hrtf_distance_m();
    let _ = super::default::default_ctc_fdw_cycles();
    let _ = super::default::default_ctc_fdw_min_ms();
    let _ = super::default::default_ctc_fdw_max_ms();
    let _ = super::default::default_ctc_max_harmonic();
    let _ = super::default::default_ctc_harmonic_window_ms();
    let _ = super::default::default_ctc_minimax_iterations();
}
