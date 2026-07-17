use super::consts::HIGHSHELF_FREQ;
use super::consts::LOWSHELF_FREQ;
use super::consts::MIN_CORRECTION_DB;
use super::types::SpectralAlignmentResult;
use math_audio_iir_fir::{Biquad, BiquadFilterType, DEFAULT_Q_HIGH_LOW_SHELF};

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
