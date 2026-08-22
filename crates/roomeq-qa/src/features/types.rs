use roomeq_model::{
    ExcursionProtectionConfig, RoomConfig, SchroederSplitConfig, TargetResponseConfig,
};

pub(super) struct FeatureStep {
    pub(super) name: &'static str,
    /// Step changes the loss function, making step-over-step score comparisons
    /// invalid at this boundary (optimizer targets a different objective).
    pub(super) changes_loss: bool,
    pub(super) apply: fn(&mut RoomConfig),
}

pub(super) fn feature_steps() -> Vec<FeatureStep> {
    let registry = crate::registry::load_registry().expect("RoomEQ QA registry must be valid");
    let suite = registry
        .suite_for_runner("features")
        .expect("RoomEQ QA registry must define features suite");
    suite
        .cases
        .iter()
        .map(|name| match name.as_str() {
            "baseline" => FeatureStep {
                name: "Baseline",
                changes_loss: false,
                apply: |_| {},
            },
            "psychoacoustic" => FeatureStep {
                name: "+ psychoacoustic",
                changes_loss: true,
                apply: |config| config.optimizer.psychoacoustic = true,
            },
            "asymmetric_loss" => FeatureStep {
                name: "+ asymmetric_loss",
                changes_loss: true,
                apply: |config| config.optimizer.asymmetric_loss = true,
            },
            "broadband" => FeatureStep {
                name: "+ broadband",
                changes_loss: true,
                apply: |config| {
                    config
                        .optimizer
                        .target_response
                        .get_or_insert_with(TargetResponseConfig::default)
                        .broadband_precorrection = true;
                },
            },
            "excursion_protection" => FeatureStep {
                name: "+ excursion_protection",
                changes_loss: true,
                apply: |config| {
                    config.optimizer.excursion_protection = Some(ExcursionProtectionConfig {
                        enabled: true,
                        ..ExcursionProtectionConfig::default()
                    });
                },
            },
            "schroeder_split" => FeatureStep {
                name: "+ schroeder_split",
                changes_loss: true,
                apply: |config| {
                    config.optimizer.schroeder_split = Some(SchroederSplitConfig {
                        enabled: true,
                        schroeder_freq: 300.0,
                        ..SchroederSplitConfig::default()
                    });
                },
            },
            other => panic!("unknown features registry case '{other}'"),
        })
        .collect()
}

pub(super) struct StepResult {
    pub(super) name: &'static str,
    pub(super) pre_score: f64,
    pub(super) post_score: f64,
    /// Worst (max) slope across channels in dB/octave
    pub(super) worst_slope: f64,
    /// True if this step changed the loss function relative to the previous step.
    pub(super) changes_loss: bool,
    /// Average EPA preference across channels (higher = better).
    /// `None` if EPA metrics were not available.
    pub(super) epa_preference: Option<f64>,
    /// True when the runtime safety gate explicitly removed one or more
    /// correction stages from the final DSP realization.
    pub(super) correction_reverted: bool,
}
