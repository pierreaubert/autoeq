//! CLI value names mapped into export-library domain types.

use clap::builder::{EnumValueParser, TypedValueParser};
use roomeq_export::ExportFormat;

/// Supported export formats for DSP chain output
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CliExportFormat {
    /// CamillaDSP YAML configuration
    #[value(name = "camilladsp")]
    CamillaDsp,
    /// Equalizer APO / Peace GUI text format (also works with PipeWire parametric-equalizer module)
    #[value(name = "apo")]
    EqualizerApo,
    /// EasyEffects JSON preset
    #[value(name = "easyeffects")]
    EasyEffects,
    /// Wavelet GraphicEQ text format
    #[value(name = "wavelet")]
    Wavelet,
    /// PipeWire filter-chain SPA-JSON configuration
    #[value(name = "pipewire")]
    PipeWire,
    /// Roon DSP Engine preset (JSON)
    #[value(name = "roon")]
    RoonDsp,
    /// REW Generic EQ reference text for manual entry; REW cannot reload this text.
    #[value(name = "rew")]
    Rew,
    /// Raw normalized biquad coefficients with channel/order metadata
    #[value(name = "coefficients", alias = "biquad-coefficients")]
    BiquadCoefficients,
}

pub(crate) fn parser() -> impl TypedValueParser<Value = ExportFormat> {
    EnumValueParser::<CliExportFormat>::new().map(|value| match value {
        CliExportFormat::CamillaDsp => ExportFormat::CamillaDsp,
        CliExportFormat::EqualizerApo => ExportFormat::EqualizerApo,
        CliExportFormat::EasyEffects => ExportFormat::EasyEffects,
        CliExportFormat::Wavelet => ExportFormat::Wavelet,
        CliExportFormat::PipeWire => ExportFormat::PipeWire,
        CliExportFormat::RoonDsp => ExportFormat::RoonDsp,
        CliExportFormat::Rew => ExportFormat::Rew,
        CliExportFormat::BiquadCoefficients => ExportFormat::BiquadCoefficients,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_spellings_and_coefficient_alias_keep_domain_values() {
        for (name, expected) in [
            ("camilladsp", ExportFormat::CamillaDsp),
            ("apo", ExportFormat::EqualizerApo),
            ("easyeffects", ExportFormat::EasyEffects),
            ("wavelet", ExportFormat::Wavelet),
            ("pipewire", ExportFormat::PipeWire),
            ("roon", ExportFormat::RoonDsp),
            ("rew", ExportFormat::Rew),
            ("coefficients", ExportFormat::BiquadCoefficients),
            ("biquad-coefficients", ExportFormat::BiquadCoefficients),
        ] {
            let command = clap::Command::new("roomeq").arg(
                clap::Arg::new("format")
                    .long("export-format")
                    .value_parser(parser()),
            );
            let matches = command
                .try_get_matches_from(["roomeq", "--export-format", name])
                .unwrap();
            assert_eq!(matches.get_one::<ExportFormat>("format"), Some(&expected));
        }
    }
}
