use super::super::room_optimization_result::RoomOptimizationResult;
use super::super::*;
use super::build::build_bootstrap_uncertainty_report;
use super::build::build_perceptual_policy_report;
use super::misc::applied_bass_crossover_hz;
use super::misc::direct_early_late_correction_metrics;
use super::misc::excursion_hpf_hz_from_chain;
use super::misc::final_score_band_for_channel;
use super::misc::recompute_curve_flatness_score;
use super::role::update_perceptual_metrics;

pub(in super::super) fn refresh_final_reports(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
    sample_rate: f64,
    sidecar_dir: &Path,
) {
    let applied_crossover_hz = applied_bass_crossover_hz(result);
    let excursion_floors: HashMap<String, f64> = result
        .channels
        .iter()
        .filter_map(|(name, chain)| excursion_hpf_hz_from_chain(chain).map(|hz| (name.clone(), hz)))
        .collect();
    for ch_result in result.channel_results.values_mut() {
        let (mut score_min_freq, score_max_freq) =
            final_score_band_for_channel(config, &ch_result.name, applied_crossover_hz);
        if let Some(hpf_hz) = excursion_floors.get(&ch_result.name) {
            score_min_freq = score_min_freq.max(*hpf_hz).min(score_max_freq);
        }
        let (topology_pre, topology_post) = (ch_result.pre_score, ch_result.post_score);
        // Report an honest improvement baseline: raw input versus final
        // deployed response, evaluated over the same role-aware band. Routed
        // pre-EQ/alignment must not be folded into the reported "before".
        ch_result.pre_score = recompute_curve_flatness_score(
            &ch_result.initial_curve,
            score_min_freq,
            score_max_freq,
        );
        // Symmetrically, routing-only transfers (crossover high-pass, gain,
        // delays) must not be folded into the reported "after": they are
        // deployment, not correction. Evaluate the de-routed final curve, the
        // same basis the acceptance gate uses, so an identity fallback scores
        // post == pre instead of showing a phantom routing regression.
        let post_basis = result
            .channels
            .get(&ch_result.name)
            .and_then(|chain| {
                let baseline = super::super::room_optimization_result::routed_baseline_curve(
                    chain,
                    &ch_result.initial_curve,
                    sample_rate,
                )?;
                super::super::room_optimization_result::remove_routing_transfer(
                    &ch_result.initial_curve,
                    &baseline,
                    &ch_result.final_curve,
                )
            })
            .unwrap_or_else(|| ch_result.final_curve.clone());
        ch_result.post_score =
            recompute_curve_flatness_score(&post_basis, score_min_freq, score_max_freq);
        log::debug!(
            "refresh_final_reports '{}': band=[{:.0},{:.0}] topology {:.4}->{:.4}, refreshed {:.4}->{:.4}",
            ch_result.name,
            score_min_freq,
            score_max_freq,
            topology_pre,
            topology_post,
            ch_result.pre_score,
            ch_result.post_score,
        );
        if let Some(chain) = result.channels.get_mut(&ch_result.name) {
            let reported = super::super::reported_curve_with_user_preferences(
                &ch_result.final_curve,
                chain,
                sample_rate,
            );
            chain.final_curve = Some((&reported).into());
        }
    }

    let count = result.channel_results.len().max(1) as f64;
    let avg_pre = result
        .channel_results
        .values()
        .map(|ch| ch.pre_score)
        .sum::<f64>()
        / count;
    let avg_post = result
        .channel_results
        .values()
        .map(|ch| ch.post_score)
        .sum::<f64>()
        / count;
    result.combined_pre_score = avg_pre;
    result.combined_post_score = avg_post;
    result.metadata.pre_score = avg_pre;
    result.metadata.post_score = avg_post;
    result.metadata.home_cinema_layout = Some(roomeq_engine::home_cinema::analyze_layout(config));
    result.metadata.multi_seat_coverage = Some(crate::home_cinema::multi_seat_coverage(config));
    if result.metadata.multi_seat_correction.is_none() && config.optimizer.multi_seat.is_some() {
        // Non-HomeCinema topology routes (e.g. Generic multi-sub) run the
        // multi-seat objective but never built the correction report; derive
        // it from the optimized channel results here.
        result.metadata.multi_seat_correction = Some(
            crate::home_cinema::multi_seat_correction_report(config, &result.channel_results, None),
        );
    }
    let existing_bass_management = result.metadata.bass_management.clone();
    result.metadata.bass_management = if let Some(existing) = existing_bass_management {
        let mut refreshed =
            roomeq_engine::home_cinema::bass_management_report_with_optimization_and_sample_rate(
                config,
                existing.applied_sub_gain_db,
                existing.gain_limited,
                existing.optimization.clone(),
                sample_rate,
            );
        // Topology workflows may calibrate the finalized per-input routing
        // graph after optimization. Rebuilding from the optimization summary
        // loses those source-specific trims, so preserve authoritative graph
        // and headroom evidence across report refresh.
        if let Some(report) = refreshed.as_mut() {
            if existing.routing_graph.is_some() {
                report.routing_graph = existing.routing_graph;
            }
            if existing.headroom_simulation.is_some() {
                report.headroom_simulation = existing.headroom_simulation;
            }
            report.advisory = existing.advisory;
        }
        refreshed
    } else {
        roomeq_engine::home_cinema::bass_management_report(config, None, false)
    };

    let epa_cfg = config.optimizer.epa_config.clone().unwrap_or_default();
    result.metadata.epa_per_channel =
        roomeq_engine::output::compute_epa_per_channel(&result.channels, &epa_cfg);
    result.metadata.epa_multichannel =
        roomeq_engine::output::compute_epa_multichannel(&result.channels, &epa_cfg);

    refresh_temporal_ir_evidence(result, config, sample_rate, sidecar_dir);

    refresh_direct_early_late_reports(result, config);
    refresh_perceptual_policy_reports(result, config);

    update_perceptual_metrics(&mut result.metadata, Some(&result.channels), Some(config));
}

/// (Re)compute per-channel impulse-response waveforms and FIR temporal
/// masking evidence from the currently deployed chains.
///
/// This must run *before* `apply_final_correction_safety_gate`: the runtime
/// acceptance policy reads `fir_temporal_masking`, and stages that add FIR
/// taps late in the pipeline (e.g. redirected bass) otherwise reach the gate
/// with `pre_ringing_evidence_missing`. `refresh_final_reports` calls it again
/// after the gate so the published evidence reflects any reverted stages.
pub(in super::super) fn refresh_temporal_ir_evidence(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
    sample_rate: f64,
    sidecar_dir: &Path,
) {
    let epa_cfg = config.optimizer.epa_config.clone().unwrap_or_default();
    let runtime_epa_cfg = roomeq_engine::config_adapter::to_optimizer_epa(&epa_cfg);

    // Evidence describes the currently deployed chain. Clear values left by
    // an earlier pre-gate refresh before rebuilding them; otherwise a safety
    // revert can leave a channel claiming FIR latency/pre-ringing even after
    // its convolution stage has been removed.
    for chain in result.channels.values_mut() {
        chain.fir_temporal_masking = None;
    }

    let ir_inputs: Vec<_> = result
        .channel_results
        .iter()
        .map(|(name, ch)| {
            let delay_ms = result
                .channels
                .get(name)
                .map(total_chain_delay_ms)
                .unwrap_or(0.0);
            let fir_coeffs = ch.fir_coeffs.clone().or_else(|| {
                deployed_fir_coefficients(result.channels.get(name), sidecar_dir, sample_rate)
            });
            (
                name.clone(),
                ch.initial_curve.clone(),
                ch.biquads.clone(),
                fir_coeffs,
                delay_ms,
            )
        })
        .collect();

    for (channel_name, initial_curve, biquads, fir_coeffs, delay_ms) in ir_inputs {
        if let Some((pre_ir, post_ir)) =
            roomeq_engine::analysis::ir_waveform::compute_channel_ir_waveforms(
                &initial_curve,
                &biquads,
                fir_coeffs.as_deref(),
                delay_ms,
                sample_rate,
            )
            && let Some(chain) = result.channels.get_mut(&channel_name)
        {
            chain.pre_ir = Some(pre_ir);
            chain.post_ir = Some(post_ir);
        }

        if let Some(coeffs) = fir_coeffs.as_deref()
            && let Some(metrics) = roomeq_engine::loss::epa::score::temporal_ir_masking_metrics(
                coeffs,
                sample_rate,
                &runtime_epa_cfg.temporal_masking,
            )
            && let Some(chain) = result.channels.get_mut(&channel_name)
        {
            chain.fir_temporal_masking = Some(
                roomeq_engine::report_adapter::to_temporal_ir_masking(metrics),
            );
        }
    }
}

/// Load a deployed convolution sidecar when the optimization result did not
/// retain its in-memory coefficients. This occurs for FIRs introduced while
/// assembling topology/workflow output, and must not make runtime evidence
/// classify the channel as IIR-only.
fn deployed_fir_coefficients(
    chain: Option<&roomeq_model::ChannelDspChain>,
    sidecar_dir: &Path,
    sample_rate: f64,
) -> Option<Vec<f64>> {
    chain?
        .plugins
        .iter()
        .filter(|plugin| plugin.plugin_type == "convolution")
        .filter_map(|plugin| {
            let ir_file = plugin.parameters.get("ir_file")?.as_str()?;
            let path = Path::new(ir_file);
            let path = if path.is_relative() {
                sidecar_dir.join(path)
            } else {
                path.to_path_buf()
            };
            let decoded = crate::wav::decode_first_channel(&path).ok()?;
            let expected_rate = sample_rate.round() as u32;
            if decoded.sample_rate != expected_rate {
                log::warn!(
                    "Ignoring FIR temporal sidecar '{}' at {} Hz; expected {} Hz",
                    path.display(),
                    decoded.sample_rate,
                    expected_rate,
                );
                return None;
            }
            Some(
                decoded
                    .samples
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>(),
            )
        })
        .max_by_key(Vec::len)
}

pub(in super::super) fn refresh_direct_early_late_reports(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
) {
    let Some(early_late_cfg) = config.optimizer.early_late_correction_config() else {
        return;
    };
    for chain in result.channels.values_mut() {
        chain.direct_early_late_correction = match (&chain.pre_ir, &chain.post_ir) {
            (Some(pre), Some(post)) => {
                direct_early_late_correction_metrics(pre, post, &early_late_cfg)
            }
            _ => None,
        };
    }
}

pub(in super::super) fn refresh_perceptual_policy_reports(
    result: &mut RoomOptimizationResult,
    config: &RoomConfig,
) {
    result.metadata.perceptual_policy = build_perceptual_policy_report(config);
    result.metadata.bootstrap_uncertainty = build_bootstrap_uncertainty_report(config);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::single_channel_room_result;

    #[test]
    fn temporal_ir_evidence_populated_for_fir_chain_before_gate() {
        // Regression test: stages that add FIR taps late in the pipeline must
        // have temporal masking evidence available before the final safety
        // gate, otherwise runtime acceptance fails with
        // `pre_ringing_evidence_missing`.
        let mut result = single_channel_room_result("L");
        let ch_result = result.channel_results.get_mut("L").expect("channel result");
        // IR waveforms require phase data on the measurement.
        let n = ch_result.initial_curve.freq.len();
        ch_result.initial_curve.phase = Some(ndarray::Array1::zeros(n));
        ch_result.fir_coeffs = Some(vec![0.0, 1.0, 0.0]);
        assert!(result.channels["L"].fir_temporal_masking.is_none());

        refresh_temporal_ir_evidence(
            &mut result,
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
        );

        let chain = &result.channels["L"];
        assert!(chain.fir_temporal_masking.is_some());
        assert!(chain.pre_ir.is_some());
        assert!(chain.post_ir.is_some());

        result.channel_results.get_mut("L").unwrap().fir_coeffs = None;
        refresh_temporal_ir_evidence(
            &mut result,
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
        );
        assert!(
            result.channels["L"].fir_temporal_masking.is_none(),
            "evidence from a removed FIR stage must not survive refresh"
        );
    }

    #[test]
    fn temporal_ir_evidence_loads_deployed_convolution_sidecar() {
        let directory = tempfile::tempdir().unwrap();
        let filename = "L_fir_48000hz.wav";
        let path = directory.path().join(filename);
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for index in 0..4096 {
            writer
                .write_sample::<f32>(if index == 2048 { 1.0 } else { 0.0 })
                .unwrap();
        }
        writer.finalize().unwrap();

        let mut result = single_channel_room_result("L");
        let ch_result = result.channel_results.get_mut("L").unwrap();
        let n = ch_result.initial_curve.freq.len();
        ch_result.initial_curve.phase = Some(ndarray::Array1::zeros(n));
        assert!(ch_result.fir_coeffs.is_none());
        result
            .channels
            .get_mut("L")
            .unwrap()
            .plugins
            .push(roomeq_model::PluginConfigWrapper {
                plugin_type: "convolution".to_string(),
                parameters: serde_json::json!({"ir_file": filename}),
            });

        refresh_temporal_ir_evidence(
            &mut result,
            &RoomConfig::default(),
            48_000.0,
            directory.path(),
        );

        let masking = result.channels["L"]
            .fir_temporal_masking
            .as_ref()
            .expect("deployed FIR temporal evidence");
        assert_eq!(masking.main_index, 2048);
        assert!((masking.main_time_ms - 2048.0 / 48.0).abs() < 1e-12);
    }

    #[test]
    fn temporal_ir_evidence_absent_without_fir() {
        let mut result = single_channel_room_result("L");

        refresh_temporal_ir_evidence(
            &mut result,
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
        );

        assert!(result.channels["L"].fir_temporal_masking.is_none());
    }

    #[test]
    fn direct_early_late_reports_populated_when_enabled_with_irs() {
        let pre_ir = roomeq_model::IrWaveform {
            time_ms: (0..16).map(|i| i as f64 * 0.1).collect(),
            amplitude: (0..16).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect(),
        };
        let mut post_ir = pre_ir.clone();
        post_ir.amplitude[0] = 0.5;
        post_ir.amplitude[10] = 0.25;
        let mut result = single_channel_room_result("L");
        let chain = result.channels.get_mut("L").expect("chain");
        chain.pre_ir = Some(pre_ir);
        chain.post_ir = Some(post_ir);

        let mut config = RoomConfig::default();
        config.optimizer.early_late_correction = Some(roomeq_model::EarlyLateCorrectionConfig {
            enabled: true,
            ..Default::default()
        });

        refresh_direct_early_late_reports(&mut result, &config);

        assert!(result.channels["L"].direct_early_late_correction.is_some());
    }

    #[test]
    fn direct_early_late_reports_absent_without_irs() {
        let mut result = single_channel_room_result("L");

        let mut config = RoomConfig::default();
        config.optimizer.early_late_correction = Some(roomeq_model::EarlyLateCorrectionConfig {
            enabled: true,
            ..Default::default()
        });

        refresh_direct_early_late_reports(&mut result, &config);

        assert!(result.channels["L"].direct_early_late_correction.is_none());
    }

    #[test]
    fn direct_early_late_reports_untouched_when_disabled() {
        let impulse_ir = || roomeq_model::IrWaveform {
            time_ms: (0..16).map(|i| i as f64 * 0.1).collect(),
            amplitude: (0..16).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect(),
        };
        let mut result = single_channel_room_result("L");
        let chain = result.channels.get_mut("L").expect("chain");
        chain.pre_ir = Some(impulse_ir());
        chain.post_ir = Some(impulse_ir());

        refresh_direct_early_late_reports(&mut result, &RoomConfig::default());

        assert!(result.channels["L"].direct_early_late_correction.is_none());
    }
}

#[cfg(test)]
mod routing_basis_tests {
    use super::*;
    use crate::test_fixtures::single_channel_room_result;

    #[test]
    fn refreshed_post_score_excludes_routing_transfer() {
        // Routing-only transfers (here an excursion-protection high-pass)
        // are deployment, not correction: an identity correction must report
        // post == pre instead of a phantom routing regression.
        let mut result = single_channel_room_result("L");
        let initial = result.channel_results["L"].initial_curve.clone();
        let chain = result.channels.get_mut("L").unwrap();
        chain.plugins = vec![roomeq_model::PluginConfigWrapper {
            plugin_type: "eq".to_string(),
            parameters: serde_json::json!({
            "label": "excursion_protection",
                "filters": [{"filter_type": "highpass", "freq": 80.0, "q": 0.707, "db_gain": 0.0}],
            }),
        }];
        let routed = crate::ctc::apply_channel_dsp_chain_to_curve(
            result.channels.get("L").unwrap(),
            &initial,
            48_000.0,
        )
        .expect("routed curve");
        // Identity correction: deployed final curve is exactly the routed
        // input, so pre and post must agree after the refresh.
        result.channel_results.get_mut("L").unwrap().final_curve = routed;

        refresh_final_reports(
            &mut result,
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
        );

        let ch = &result.channel_results["L"];
        assert!(
            (ch.post_score - ch.pre_score).abs() < 1e-9,
            "identity correction must score post == pre, got {} -> {}",
            ch.pre_score,
            ch.post_score
        );
    }
}
