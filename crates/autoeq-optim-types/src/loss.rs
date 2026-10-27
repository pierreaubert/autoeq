//! Objective selection shared by optimizer and configuration clients.

/// The type of loss function to use during optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossType {
    /// Flat loss function (minimize deviation from target curve)
    SpeakerFlat,
    /// Flat loss with asymmetric weighting (peaks penalized 2x more than dips)
    /// Use this for room correction where nulls cannot be fixed with EQ
    SpeakerFlatAsymmetric,
    /// Harman/Olive preference-score objective.
    ///
    /// The helper functions return a preference score, and the optimizer wrapper
    /// converts that score into a minimization objective.
    SpeakerScore,
    /// Flat loss function (minimize deviation from target curve)
    HeadphoneFlat,
    /// Harman headphone preference-score objective.
    ///
    /// The helper functions return a preference score, and the optimizer wrapper
    /// converts that score into a minimization objective.
    HeadphoneScore,
    /// Multi-driver crossover optimization (flatten combined response)
    DriversFlat,
    /// Multi-subwoofer optimization (flatten summed response)
    MultiSubFlat,
    /// EPA (Evaluation, Potency, Activity) perceptual loss.
    /// Combines spectral flatness with sharpness, roughness, and loudness
    /// balance penalties derived from Zwicker psychoacoustic metrics.
    Epa,
}
