use super::consts::HIGHSHELF_FREQ;
use super::consts::LOWSHELF_FREQ;
use super::consts::MIN_CORRECTION_DB;
use super::types::SpectralAlignmentResult;
use math_audio_iir_fir::{Biquad, BiquadFilterType, DEFAULT_Q_HIGH_LOW_SHELF};
use roomeq_model::PluginConfigWrapper;

/// Create the biquad shelf filters represented by a spectral alignment result.
pub fn create_alignment_filters(result: &SpectralAlignmentResult, sample_rate: f64) -> Vec<Biquad> {
    let mut shelf_filters = Vec::new();

    if result.lowshelf_gain_db.abs() >= MIN_CORRECTION_DB {
        shelf_filters.push(Biquad::new(
            BiquadFilterType::Lowshelf,
            LOWSHELF_FREQ,
            sample_rate,
            DEFAULT_Q_HIGH_LOW_SHELF,
            result.lowshelf_gain_db,
        ));
    }

    if result.highshelf_gain_db.abs() >= MIN_CORRECTION_DB {
        shelf_filters.push(Biquad::new(
            BiquadFilterType::Highshelf,
            HIGHSHELF_FREQ,
            sample_rate,
            DEFAULT_Q_HIGH_LOW_SHELF,
            result.highshelf_gain_db,
        ));
    }

    shelf_filters
}

/// Convert a spectral alignment into the DSP plugins consumed by the room
/// execution graph.
pub fn create_alignment_plugins(
    result: &SpectralAlignmentResult,
    sample_rate: f64,
) -> (Option<PluginConfigWrapper>, Option<PluginConfigWrapper>) {
    let shelf_filters = create_alignment_filters(result, sample_rate);
    let eq_plugin = (!shelf_filters.is_empty())
        .then(|| crate::output::create_labeled_eq_plugin(&shelf_filters, "broadband"));
    let gain_plugin = (result.flat_gain_db.abs() >= MIN_CORRECTION_DB)
        .then(|| crate::output::create_gain_plugin(result.flat_gain_db));
    (eq_plugin, gain_plugin)
}

#[cfg(test)]
mod plugin_tests {
    use super::*;

    #[test]
    fn alignment_plugins_preserve_broadband_output_contract() {
        let result = SpectralAlignmentResult {
            lowshelf_gain_db: -2.0,
            highshelf_gain_db: 1.5,
            flat_gain_db: -1.0,
            residual_rms_db: 0.5,
        };
        let (eq, gain) = create_alignment_plugins(&result, 48_000.0);
        assert_eq!(eq.expect("shelf plugin").plugin_type, "eq");
        assert_eq!(gain.expect("gain plugin").plugin_type, "gain");
    }
}
