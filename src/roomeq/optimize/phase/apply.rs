use super::super::types::ChannelOptimizationResult;
use super::super::*;
use super::misc::compute_phase_alignment_delay_schedule;
use super::misc::convolve;
use super::sync::sync_reported_phase_adjustment;

/// Apply standalone phase correction to a channel (rePhase-style).
///
/// Generates a phase-only FIR from the measurement's excess phase and appends it
/// to the channel's DSP chain. If the channel already has a magnitude FIR, the
/// two are convolved together so `fir_coeffs` remains a single filter for IR
/// computation.
pub(in super::super) fn apply_phase_correction(
    name: &str,
    ch: &mut ChannelOptimizationResult,
    chain: &mut crate::roomeq::types::ChannelDspChain,
    config: &crate::roomeq::types::MixedPhaseSerdeConfig,
    sample_rate: f64,
    output_dir: Option<&Path>,
) {
    let phase = match ch.initial_curve.phase.as_ref() {
        Some(p) if !p.is_empty() => p,
        _ => return,
    };
    let _ = phase; // used via initial_curve below

    let mp_config = crate::roomeq::mixed_phase::MixedPhaseConfig {
        max_fir_length_ms: config.max_fir_length_ms,
        pre_ringing_threshold_db: config.pre_ringing_threshold_db,
        min_spatial_depth: config.min_spatial_depth,
        phase_smoothing_octaves: config.phase_smoothing_octaves,
    };

    let phase_fir = match crate::roomeq::mixed_phase::decompose_phase(&ch.initial_curve, &mp_config)
    {
        Ok((_min, _excess, delay_ms, residual)) => {
            info!(
                "  Phase correction '{}': delay={:.2} ms, generating phase-only FIR",
                name, delay_ms
            );
            crate::roomeq::mixed_phase::generate_excess_phase_fir(
                &ch.initial_curve.freq,
                &residual,
                &mp_config,
                sample_rate,
            )
        }
        Err(e) => {
            warn!("  Phase correction failed for '{}': {}", name, e);
            return;
        }
    };

    // Save phase FIR WAV and add convolution plugin
    let mut filename = crate::roomeq::artifacts::convolution_artifact_filename(
        name,
        crate::roomeq::artifacts::ConvolutionArtifactKind::PhaseCorrection,
        sample_rate,
    );
    if let Some(out_dir) = output_dir {
        let reserved = crate::roomeq::artifacts::reserve_convolution_artifact_path(
            out_dir,
            name,
            crate::roomeq::artifacts::ConvolutionArtifactKind::PhaseCorrection,
            sample_rate,
        );
        filename = reserved.0;
        let wav_path = reserved.1;
        if let Err(e) = crate::fir::save_fir_to_wav(&phase_fir, sample_rate as u32, &wav_path) {
            warn!("Failed to save phase correction FIR for {}: {}", name, e);
        } else {
            info!("  Saved phase correction FIR to {}", wav_path.display());
        }
    }
    chain
        .plugins
        .push(crate::roomeq::output::create_convolution_plugin(&filename));

    let phase_response = crate::response::compute_fir_complex_response(
        &phase_fir,
        &ch.final_curve.freq,
        sample_rate,
    );
    ch.final_curve = crate::response::apply_complex_response(&ch.final_curve, &phase_response);
    chain.final_curve = Some((&ch.final_curve).into());

    // Combine with existing FIR for IR computation (convolve the two)
    if let Some(ref existing) = ch.fir_coeffs {
        ch.fir_coeffs = Some(convolve(existing, &phase_fir));
    } else {
        ch.fir_coeffs = Some(phase_fir);
    }
}

pub(in super::super) fn apply_phase_alignment_delay_schedule(
    phase_alignment_results: &HashMap<String, (f64, bool, String)>,
    channel_results: &mut HashMap<String, ChannelOptimizationResult>,
    channel_chains: &mut HashMap<String, ChannelDspChain>,
) -> HashMap<String, f64> {
    let schedule = compute_phase_alignment_delay_schedule(phase_alignment_results);

    for (channel_name, delay_ms) in &schedule {
        let applied = if let Some(chain) = channel_chains.get_mut(channel_name.as_str()) {
            output::add_delay_plugin(chain, *delay_ms);
            true
        } else {
            false
        };

        if applied {
            sync_reported_phase_adjustment(
                channel_name,
                channel_results,
                channel_chains,
                *delay_ms,
                false,
            );
            info!(
                "  Applied {:.3} ms phase alignment delay to '{}'",
                delay_ms, channel_name
            );
        }
    }

    schedule
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roomeq::ChannelOptimizationResult;
    use crate::roomeq::types::{ChannelDspChain, CurveData};
    use ndarray::Array1;
    use std::collections::HashMap;

    fn small_curve() -> crate::Curve {
        crate::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 16),
            spl: Array1::from_elem(16, 80.0),
            phase: Some(Array1::from_elem(16, 0.0)),
            ..Default::default()
        }
    }

    fn small_curve_no_phase() -> crate::Curve {
        crate::Curve {
            freq: Array1::logspace(10.0, f64::log10(20.0), f64::log10(20_000.0), 16),
            spl: Array1::from_elem(16, 80.0),
            phase: None,
            ..Default::default()
        }
    }

    fn curve_data(curve: &crate::Curve) -> CurveData {
        CurveData {
            freq: curve.freq.to_vec(),
            spl: curve.spl.to_vec(),
            phase: curve.phase.as_ref().map(|p| p.to_vec()),
            norm_range: None,
        }
    }

    fn make_channel(
        name: &str,
        curve: crate::Curve,
    ) -> (ChannelOptimizationResult, ChannelDspChain) {
        let ch = ChannelOptimizationResult {
            name: name.to_string(),
            pre_score: 0.0,
            post_score: 0.0,
            initial_curve: curve.clone(),
            final_curve: curve.clone(),
            biquads: Vec::new(),
            fir_coeffs: None,
            optimizer_evidence: Vec::new(),
        };
        let chain = ChannelDspChain {
            channel: name.to_string(),
            plugins: Vec::new(),
            drivers: None,
            initial_curve: None,
            final_curve: Some(curve_data(&curve)),
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        (ch, chain)
    }

    #[test]
    fn apply_phase_correction_skips_without_phase() {
        let (mut ch, mut chain) = make_channel("left", small_curve_no_phase());
        let config = crate::roomeq::types::MixedPhaseSerdeConfig {
            max_fir_length_ms: 10.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
        };
        apply_phase_correction("left", &mut ch, &mut chain, &config, 48_000.0, None);
        assert!(chain.plugins.is_empty());
        assert!(ch.fir_coeffs.is_none());
    }

    #[test]
    fn apply_phase_alignment_delay_schedule_adds_delay_plugins() {
        let (l_ch, l_chain) = make_channel("L", small_curve());
        let (r_ch, r_chain) = make_channel("R", small_curve());
        let (sub_ch, sub_chain) = make_channel("Sub", small_curve());
        let mut results = HashMap::from([
            ("L".to_string(), l_ch),
            ("R".to_string(), r_ch),
            ("Sub".to_string(), sub_ch),
        ]);
        let mut chains = HashMap::from([
            ("L".to_string(), l_chain),
            ("R".to_string(), r_chain),
            ("Sub".to_string(), sub_chain),
        ]);
        let phase_results = HashMap::from([
            ("L".to_string(), (-2.0, false, "Sub".to_string())),
            ("R".to_string(), (1.0, false, "Sub".to_string())),
        ]);
        let schedule =
            apply_phase_alignment_delay_schedule(&phase_results, &mut results, &mut chains);
        assert!(schedule.contains_key("Sub"));
        assert!(schedule.contains_key("R"));
        assert!(!schedule.contains_key("L"));
        assert!(
            chains["Sub"]
                .plugins
                .iter()
                .any(|p| p.plugin_type == "delay")
        );
        assert!(chains["R"].plugins.iter().any(|p| p.plugin_type == "delay"));
        assert!(!chains["L"].plugins.iter().any(|p| p.plugin_type == "delay"));
    }

    #[test]
    fn apply_phase_alignment_delay_schedule_empty_results_empty_schedule() {
        let mut results = HashMap::<String, ChannelOptimizationResult>::new();
        let mut chains = HashMap::<String, ChannelDspChain>::new();
        let schedule =
            apply_phase_alignment_delay_schedule(&HashMap::new(), &mut results, &mut chains);
        assert!(schedule.is_empty());
    }

    #[test]
    fn apply_phase_correction_generates_fir_without_output_dir() {
        let (mut ch, mut chain) = make_channel("left", small_curve());
        let config = crate::roomeq::types::MixedPhaseSerdeConfig {
            max_fir_length_ms: 5.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
        };
        apply_phase_correction("left", &mut ch, &mut chain, &config, 48_000.0, None);
        assert!(ch.fir_coeffs.is_some());
        assert!(
            chain.plugins.iter().any(|p| p.plugin_type == "convolution"),
            "phase correction should add a convolution plugin"
        );
    }

    #[test]
    fn apply_phase_correction_saves_wav_when_output_dir_provided() {
        let tmp = tempfile::TempDir::new().unwrap();
        let (mut ch, mut chain) = make_channel("left", small_curve());
        let config = crate::roomeq::types::MixedPhaseSerdeConfig {
            max_fir_length_ms: 5.0,
            pre_ringing_threshold_db: -30.0,
            min_spatial_depth: 0.5,
            phase_smoothing_octaves: 1.0 / 6.0,
        };
        apply_phase_correction(
            "left",
            &mut ch,
            &mut chain,
            &config,
            48_000.0,
            Some(tmp.path()),
        );
        assert!(ch.fir_coeffs.is_some());
        let wav_files: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "wav"))
            .collect();
        assert!(!wav_files.is_empty(), "phase FIR WAV should be saved");
    }
}
