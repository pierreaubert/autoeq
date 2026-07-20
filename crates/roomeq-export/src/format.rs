use std::path::{Path, PathBuf};

/// Supported external RoomEQ export formats.
///
/// This is an engine-neutral contract: renderers may reject a format when the
/// realized DSP graph uses routing or stages that target cannot represent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ExternalExportFormat {
    #[value(name = "camilladsp")]
    CamillaDsp,
    #[value(name = "apo")]
    EqualizerApo,
    #[value(name = "easyeffects")]
    EasyEffects,
    #[value(name = "wavelet")]
    Wavelet,
    #[value(name = "pipewire")]
    PipeWire,
    #[value(name = "roon")]
    RoonDsp,
    #[value(name = "rew")]
    Rew,
    #[value(name = "coefficients", alias = "biquad-coefficients")]
    BiquadCoefficients,
}

impl ExternalExportFormat {
    pub fn default_extension(self) -> &'static str {
        match self {
            Self::CamillaDsp => "yaml",
            Self::EqualizerApo => "txt",
            Self::EasyEffects => "json",
            Self::Wavelet => "txt",
            Self::PipeWire => "conf",
            Self::RoonDsp => "json",
            Self::Rew => "txt",
            Self::BiquadCoefficients => "json",
        }
    }

    pub fn default_file_name(self) -> &'static str {
        match self {
            Self::CamillaDsp => "room_eq_cdsp.yaml",
            Self::EqualizerApo => "room_eq.txt",
            Self::EasyEffects => "room_eq.json",
            Self::Wavelet => "room_eq.txt",
            Self::PipeWire => "room_eq.conf",
            Self::RoonDsp => "room_eq.json",
            Self::Rew => "room_eq_rew.txt",
            Self::BiquadCoefficients => "room_eq_biquads.json",
        }
    }

    pub fn default_export_path(self, output_path: &Path) -> PathBuf {
        if matches!(self, Self::CamillaDsp)
            && let Some(stem) = output_path.file_stem().and_then(|stem| stem.to_str())
        {
            let mut path = output_path.to_path_buf();
            path.set_file_name(format!("{stem}_cdsp.{}", self.default_extension()));
            return path;
        }
        output_path.with_extension(self.default_extension())
    }
}

#[cfg(test)]
mod tests {
    use super::ExternalExportFormat;
    use std::path::Path;

    #[test]
    fn camilladsp_path_keeps_the_dsp_suffix() {
        assert_eq!(
            ExternalExportFormat::CamillaDsp.default_export_path(Path::new("result.json")),
            Path::new("result_cdsp.yaml")
        );
    }

    #[test]
    fn other_paths_replace_the_output_extension() {
        assert_eq!(
            ExternalExportFormat::PipeWire.default_export_path(Path::new("result.json")),
            Path::new("result.conf")
        );
    }
}
