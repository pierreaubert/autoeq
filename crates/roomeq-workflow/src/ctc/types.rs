use super::dsp_response_cache::apply_room_eq_dsp_to_spectrum;
use super::load::load_hrtf_spectrum;
use super::load::load_measured_spectrum;
use super::load::load_raw_sweep_spectrum;
use super::misc::CTC_ARTIFACT_VERSION;
use super::misc::checked_sample_rate;
use super::misc::invalid_ctc_configuration;
use roomeq_engine::ctc::solve_prepared_ctc;
pub(super) use roomeq_engine::ctc::{
    PreparedCtcFilter as CtcFirFilterArtifact, PreparedCtcMatrix as MatrixSpectrum,
    build_matrix_spectrum,
};
use roomeq_engine::error::Result;
use roomeq_model::{ChannelDspChain, CtcConfig, SystemConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub use roomeq_model::{
    CtcBinauralDiagnostics, CtcDeliveredResponseMetrics, CtcHrtfCandidateComparison, CtcReport,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct CtcArtifact {
    pub(super) version: String,
    pub(super) source: String,
    pub(super) sample_rate: u32,
    pub(super) speakers: Vec<String>,
    pub(super) ears: Vec<String>,
    pub(super) fir_taps: usize,
    pub(super) latency_samples: usize,
    pub(super) latency_ms: f64,
    pub(super) max_filter_gain_db: f64,
    pub(super) max_condition_number: f64,
    pub(super) mean_reconstruction_error: f64,
    pub(super) worst_position_error: f64,
    pub(super) mean_crosstalk_residual_db: f64,
    pub(super) max_electrical_sum_gain_db: f64,
    pub(super) driver_headroom_limited: bool,
    pub(super) room_eq_correction_applied: bool,
    pub(super) room_eq_correction_channels: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) delivered_response: Option<CtcDeliveredResponseMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) binaural_diagnostics: Option<CtcBinauralDiagnostics>,
    pub(super) filters: Vec<CtcFirFilterArtifact>,
}

pub fn maybe_generate_recommended_xtc(
    config: &CtcConfig,
    sys: &SystemConfig,
    sample_rate: f64,
    output_dir: &Path,
    channels: Option<&HashMap<String, ChannelDspChain>>,
) -> Result<Option<CtcReport>> {
    if !config.enabled {
        return Ok(None);
    }
    if config.fir_taps < 16 || !config.fir_taps.is_power_of_two() {
        return Err(invalid_ctc_configuration(
            "ctc.fir_taps must be a power of two >= 16",
        ));
    }

    let sample_rate_u32 = checked_sample_rate(sample_rate)?;
    let fft_size = config.fir_taps;
    let mut spectrum = match config.matrix_source.as_str() {
        "measured" => {
            let measurements = config.measurements.as_ref().ok_or_else(|| {
                invalid_ctc_configuration("ctc.matrix_source='measured' requires ctc.measurements")
            })?;
            load_measured_spectrum(measurements, &config.window, sample_rate_u32, fft_size)?
        }
        "raw_sweep" => {
            let measurements = config.measurements.as_ref().ok_or_else(|| {
                invalid_ctc_configuration("ctc.matrix_source='raw_sweep' requires ctc.measurements")
            })?;
            load_raw_sweep_spectrum(measurements, config, sample_rate_u32, fft_size)?
        }
        "hrtf_database" | "hrtf" => {
            let hrtf = config.hrtf.as_ref().ok_or_else(|| {
                invalid_ctc_configuration("ctc.matrix_source='hrtf_database' requires ctc.hrtf")
            })?;
            load_hrtf_spectrum(hrtf, sample_rate_u32, fft_size)?
        }
        other => {
            return Err(invalid_ctc_configuration(format!(
                "unsupported ctc.matrix_source '{}'; expected 'measured', 'raw_sweep', or 'hrtf_database'",
                other
            )));
        }
    };

    for speaker in &spectrum.speakers {
        if !sys.speakers.contains_key(speaker) {
            return Err(invalid_ctc_configuration(format!(
                "ctc speaker '{}' is not present in system.speakers",
                speaker
            )));
        }
    }
    if spectrum.speakers.len() < 2 {
        return Err(invalid_ctc_configuration(
            "ctc requires at least two speaker roles",
        ));
    }
    let room_eq_correction_channels = if config.include_room_eq_dsp {
        if let Some(channels) = channels {
            apply_room_eq_dsp_to_spectrum(&mut spectrum, sys, channels, sample_rate)?;
            spectrum
                .speakers
                .iter()
                .filter_map(|speaker| {
                    let channel_name = sys.speakers.get(speaker)?;
                    channels
                        .get(channel_name)
                        .is_some_and(|chain| !chain.plugins.is_empty())
                        .then(|| channel_name.clone())
                })
                .collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };
    let room_eq_correction_applied = !room_eq_correction_channels.is_empty();

    let solution = solve_prepared_ctc(&spectrum, config, sample_rate)?;
    let latency_samples = solution.latency_samples;
    let latency_ms = solution.latency_ms;
    let max_filter_gain_db = solution.max_filter_gain_db;
    let max_condition_json = solution.max_condition_number;
    let mean_reconstruction_error = solution.mean_reconstruction_error;
    let worst_position_error = solution.worst_position_error;
    let mean_crosstalk_residual_db = solution.mean_crosstalk_residual_db;
    let max_electrical_sum_gain_db = solution.max_electrical_sum_gain_db;
    let driver_headroom_limited = solution.driver_headroom_limited;
    let delivered_response = solution.delivered_response;
    let binaural_diagnostics = solution.binaural_diagnostics;
    let filters = solution.filters;

    std::fs::create_dir_all(output_dir)?;
    let artifact_path = output_dir.join("recommended_xtc_matrix.json");
    let artifact = CtcArtifact {
        version: CTC_ARTIFACT_VERSION.to_string(),
        source: spectrum.source.clone(),
        sample_rate: sample_rate_u32,
        speakers: spectrum.speakers.clone(),
        ears: spectrum.ears.clone(),
        fir_taps: config.fir_taps,
        latency_samples,
        latency_ms,
        max_filter_gain_db,
        max_condition_number: max_condition_json,
        mean_reconstruction_error,
        worst_position_error,
        mean_crosstalk_residual_db,
        max_electrical_sum_gain_db,
        driver_headroom_limited,
        room_eq_correction_applied,
        room_eq_correction_channels: room_eq_correction_channels.clone(),
        delivered_response: Some(delivered_response.clone()),
        binaural_diagnostics: Some(binaural_diagnostics.clone()),
        filters,
    };
    let json = serde_json::to_vec_pretty(&artifact)?;
    std::fs::write(&artifact_path, json)?;

    Ok(Some(CtcReport {
        enabled: true,
        source: spectrum.source,
        artifact: artifact_path.to_string_lossy().to_string(),
        speakers: spectrum.speakers,
        ears: spectrum.ears,
        head_positions: spectrum.positions.len(),
        fir_taps: config.fir_taps,
        latency_samples,
        latency_ms,
        max_filter_gain_db,
        max_condition_number: max_condition_json,
        mean_reconstruction_error,
        worst_position_error,
        mean_crosstalk_residual_db,
        max_electrical_sum_gain_db,
        driver_headroom_limited,
        room_eq_correction_applied,
        room_eq_correction_channels,
        delivered_response: Some(delivered_response),
        binaural_diagnostics: Some(binaural_diagnostics),
    }))
}
