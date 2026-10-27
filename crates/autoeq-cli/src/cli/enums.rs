//! Terminal enum metadata mapped into parser-independent domain values.

use autoeq_core::PeqModel;
use autoeq_optim_types::LossType;
use clap::ValueEnum;
use clap::builder::{EnumValueParser, TypedValueParser};

/// The type of loss function to use during optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliLossType {
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

pub(super) fn loss_type_parser() -> impl TypedValueParser<Value = LossType> {
    EnumValueParser::<CliLossType>::new().map(|value| match value {
        CliLossType::SpeakerFlat => LossType::SpeakerFlat,
        CliLossType::SpeakerFlatAsymmetric => LossType::SpeakerFlatAsymmetric,
        CliLossType::SpeakerScore => LossType::SpeakerScore,
        CliLossType::HeadphoneFlat => LossType::HeadphoneFlat,
        CliLossType::HeadphoneScore => LossType::HeadphoneScore,
        CliLossType::DriversFlat => LossType::DriversFlat,
        CliLossType::MultiSubFlat => LossType::MultiSubFlat,
        CliLossType::Epa => LossType::Epa,
    })
}

/// PEQ model types that define the structure and constraints of the equalizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CliPeqModel {
    #[value(name = "pk")]
    Pk,
    #[value(name = "hp-pk")]
    HpPk,
    #[value(name = "hp-pk-lp")]
    HpPkLp,
    #[value(name = "ls-pk")]
    LsPk,
    #[value(name = "ls-pk-hs")]
    LsPkHs,
    #[value(name = "pk-ls-hs")]
    PkLsHs,
    #[value(name = "free-pk-free")]
    FreePkFree,
    #[value(name = "free")]
    Free,
}

pub(super) fn peq_model_parser() -> impl TypedValueParser<Value = PeqModel> {
    EnumValueParser::<CliPeqModel>::new().map(|value| match value {
        CliPeqModel::Pk => PeqModel::Pk,
        CliPeqModel::HpPk => PeqModel::HpPk,
        CliPeqModel::HpPkLp => PeqModel::HpPkLp,
        CliPeqModel::LsPk => PeqModel::LsPk,
        CliPeqModel::LsPkHs => PeqModel::LsPkHs,
        CliPeqModel::PkLsHs => PeqModel::PkLsHs,
        CliPeqModel::FreePkFree => PeqModel::FreePkFree,
        CliPeqModel::Free => PeqModel::Free,
    })
}
