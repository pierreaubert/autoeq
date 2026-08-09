use math_audio_iir_fir::BiquadFilterType;

/// Map filter type string to CamillaDSP filter type
pub(super) fn camilladsp_filter_type(ft: &str) -> &str {
    match ft {
        "peak" => "Peaking",
        "lowshelf" => "Lowshelf",
        "highshelf" => "Highshelf",
        "lowpass" => "Lowpass",
        "highpass" | "highpassvariableq" => "Highpass",
        "notch" => "Notch",
        "bandpass" => "Bandpass",
        "allpass" => "Allpass",
        other => other,
    }
}

/// Map filter type string to APO filter abbreviation
pub(super) fn apo_filter_type(ft: &str) -> &str {
    match ft {
        "peak" => "PK",
        "lowshelf" => "LSC",
        "highshelf" => "HSC",
        "lowpass" => "LPQ",
        "highpass" | "highpassvariableq" => "HPQ",
        "notch" => "NO",
        "bandpass" => "BP",
        "allpass" => "AP",
        other => other,
    }
}

/// Map filter type string to EasyEffects type
pub(super) fn easyeffects_filter_type(ft: &str) -> &str {
    match ft {
        "peak" => "Bell",
        "lowshelf" => "Lo Shelf",
        "highshelf" => "Hi Shelf",
        "lowpass" => "Lo-pass",
        "highpass" | "highpassvariableq" => "Hi-pass",
        "notch" => "Notch",
        "bandpass" => "Bandpass",
        "allpass" => "Allpass",
        other => other,
    }
}

/// Fixed graphic EQ band center frequencies for Wavelet
pub(super) const WAVELET_BANDS: [f64; 11] = [
    20.0, 32.0, 64.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 20_000.0,
];

/// Parse filter type string to BiquadFilterType enum
pub(super) fn parse_biquad_filter_type(ft: &str) -> anyhow::Result<BiquadFilterType> {
    let filter_type = match ft {
        "peak" => BiquadFilterType::Peak,
        "lowshelf" => BiquadFilterType::Lowshelf,
        "highshelf" => BiquadFilterType::Highshelf,
        "lowpass" => BiquadFilterType::Lowpass,
        "highpass" => BiquadFilterType::Highpass,
        "highpassvariableq" => BiquadFilterType::HighpassVariableQ,
        "notch" => BiquadFilterType::Notch,
        "bandpass" => BiquadFilterType::Bandpass,
        "allpass" => BiquadFilterType::AllPass,
        "lowshelforf" => BiquadFilterType::LowshelfOrf,
        "highshelforf" => BiquadFilterType::HighshelfOrf,
        "peakmatched" => BiquadFilterType::PeakMatched,
        other => anyhow::bail!("Unsupported biquad filter type '{other}'"),
    };

    Ok(filter_type)
}

/// Map filter type string to Roon parametric EQ type
pub(super) fn roon_filter_type(ft: &str) -> anyhow::Result<&str> {
    Ok(match ft {
        "peak" => "Peak/Dip",
        "lowshelf" => "Low Shelf",
        "highshelf" => "High Shelf",
        "lowpass" => "Low Pass",
        "highpass" | "highpassvariableq" => "High Pass",
        "bandpass" => "Band Pass",
        "notch" => "Band Stop",
        "allpass" => anyhow::bail!("Roon cannot represent all-pass filters"),
        other => anyhow::bail!("Roon does not support filter type '{other}'"),
    })
}
