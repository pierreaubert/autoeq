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

#[cfg(test)]
mod serialized_waveform_tests {
    use super::*;
    use math_audio_iir_fir::KautzFilter;

    #[test]
    fn serial_fir_composition_matches_direct_oracle_and_checks_resources() {
        let dir = tempfile::tempdir().unwrap();
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let chain = result.channels.get_mut("L").unwrap();
        chain.plugins.clear();
        for (name, taps) in [("a.wav", vec![1.0, 2.0, 3.0]), ("b.wav", vec![4.0, 5.0])] {
            let mut writer = hound::WavWriter::create(
                dir.path().join(name),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 48_000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for tap in taps {
                writer.write_sample(tap as f32).unwrap();
            }
            writer.finalize().unwrap();
            chain
                .plugins
                .push(roomeq_engine::output::create_convolution_plugin(name));
        }
        let taps = deployed_fir_coefficients(Some(chain), None, dir.path(), 48_000.0).unwrap();
        // Direct polynomial multiplication, including the last nonzero tail.
        assert_eq!(taps.len(), 4);
        for (actual, expected) in taps.iter().zip([4.0, 13.0, 22.0, 15.0]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!(deployed_fir_coefficients(Some(chain), None, dir.path(), 96_000.0).is_none());
        chain.plugins.push(roomeq_model::PluginConfigWrapper {
            plugin_type: "band_split".into(),
            parameters: serde_json::json!({}),
        });
        assert!(deployed_fir_coefficients(Some(chain), None, dir.path(), 48_000.0).is_none());
    }

    #[test]
    fn serial_fir_temporal_evidence_composes_all_resources_and_rejects_partial_sets() {
        let dir = tempfile::tempdir().unwrap();
        for rate in [48_000_u32, 96_000] {
            let mut result = crate::test_fixtures::single_channel_room_result("L");
            let channel = result.channel_results.get_mut("L").unwrap();
            channel.initial_curve.phase =
                Some(ndarray::Array1::zeros(channel.initial_curve.freq.len()));
            // A retained single-stage kernel must not override two emitted resources.
            channel.fir_coeffs = Some(vec![1.0]);
            let delay = rate as usize / 1000;
            let mut taps = vec![0.0; delay + 1];
            taps[delay] = 1.0;
            for name in ["a.wav", "b.wav"] {
                let mut writer = hound::WavWriter::create(
                    dir.path().join(name),
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: rate,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    },
                )
                .unwrap();
                for &value in &taps {
                    writer.write_sample(value as f32).unwrap();
                }
                writer.finalize().unwrap();
            }
            result.channels.get_mut("L").unwrap().plugins = ["a.wav", "b.wav"]
                .into_iter()
                .map(|name| roomeq_model::PluginConfigWrapper {
                    plugin_type: "convolution".into(),
                    parameters: serde_json::json!({"ir_file": name}),
                })
                .collect();
            refresh_temporal_ir_evidence(
                &mut result,
                &RoomConfig::default(),
                f64::from(rate),
                dir.path(),
            );
            let metrics = result.channels["L"].fir_temporal_masking.as_ref().unwrap();
            assert!(
                (metrics.main_time_ms - 2.0).abs() < 0.05,
                "{rate}: {metrics:?}"
            );

            // Resolve every resource: missing later stages cannot leave a partial pass.
            result.channels.get_mut("L").unwrap().plugins[1].parameters["ir_file"] =
                serde_json::json!("missing.wav");
            refresh_temporal_ir_evidence(
                &mut result,
                &RoomConfig::default(),
                f64::from(rate),
                dir.path(),
            );
            assert!(result.channels["L"].fir_temporal_masking.is_none());
            assert!(result.channels["L"].post_ir.is_none());
        }
    }

    #[test]
    fn serialized_kautz_waveform_matches_streamed_bank_gain_and_delay() {
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            let mut result = crate::test_fixtures::single_channel_room_result("L");
            let channel = result.channel_results.get_mut("L").unwrap();
            channel.initial_curve.spl.fill(80.0);
            channel.initial_curve.phase =
                Some(ndarray::Array1::zeros(channel.initial_curve.freq.len()));
            // No PEQ surrogate is necessary to describe this emitted bank.
            channel.biquads.clear();
            let filter = serde_json::json!({
                "topology": "kautz_filter", "filter_type": "peak",
                "freq": 75.0, "q": 2.0, "db_gain": 0.0,
                "kautz_sections": [
                    {"pole_freq": 75.0, "q": 2.0, "gain": -0.025},
                    {"pole_freq": 135.0, "q": 3.0, "gain": 0.018}
                ]
            });
            result.channels.get_mut("L").unwrap().plugins = vec![
                roomeq_engine::output::create_labeled_eq_plugin_from_filter_configs(
                    vec![filter],
                    "kautz_modal",
                ),
                roomeq_engine::output::create_gain_plugin(-6.0),
                roomeq_engine::output::create_delay_plugin(128_000.0 / rate),
            ];
            refresh_temporal_ir_evidence(&mut result, &RoomConfig::default(), rate, Path::new("."));
            let reported = result.channels["L"]
                .post_ir
                .as_ref()
                .expect("phase-supported serialized waveform");
            let mut bank = KautzFilter::from_room_modes(&[(75.0, 2.0), (135.0, 3.0)], rate);
            bank.sections[0].gain = -0.025;
            bank.sections[1].gain = 0.018;
            let mut expected = vec![0.0; reported.amplitude.len()];
            for (sample, value) in expected.iter_mut().enumerate().skip(128) {
                let input = if sample == 128 { 1.0 } else { 0.0 };
                *value = 10.0_f64.powf(-6.0 / 20.0) * (input + bank.process(input));
            }
            let error = reported
                .amplitude
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                error < 1e-9,
                "{rate} Hz: waveform differs from streamed playback by {error}"
            );
        }
    }

    #[test]
    fn serialized_waveform_failure_clears_stale_views_without_peq_fallback() {
        let mut result = crate::test_fixtures::single_channel_room_result("L");
        let initial = &mut result.channel_results.get_mut("L").unwrap().initial_curve;
        initial.phase = Some(ndarray::Array1::zeros(initial.freq.len()));
        result.channels.get_mut("L").unwrap().plugins =
            vec![roomeq_engine::output::create_gain_plugin(-3.0)];
        refresh_temporal_ir_evidence(
            &mut result,
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
        );
        assert!(result.channels["L"].post_ir.is_some());
        result.channels.get_mut("L").unwrap().plugins = vec![
            roomeq_engine::output::create_labeled_eq_plugin_from_filter_configs(
                vec![serde_json::json!({
                    "topology": "kautz_filter",
                    "kautz_sections": [{"pole_freq": 75.0, "q": 2.0, "gain": "invalid"}]
                })],
                "kautz_modal",
            ),
        ];
        refresh_temporal_ir_evidence(
            &mut result,
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
        );
        assert!(result.channels["L"].pre_ir.is_none());
        assert!(result.channels["L"].post_ir.is_none());
    }
}

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
                super::super::room_optimization_result::correction_only_curve(
                    chain,
                    &ch_result.initial_curve,
                    &ch_result.final_curve,
                    sample_rate,
                    sidecar_dir,
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
    let mut waveform_errors = std::collections::BTreeMap::new();

    // Evidence describes the currently deployed chain. Clear values left by
    // an earlier pre-gate refresh before rebuilding them; otherwise a safety
    // revert can leave a channel claiming FIR latency/pre-ringing even after
    // its convolution stage has been removed.
    for chain in result.channels.values_mut() {
        chain.fir_temporal_masking = None;
        chain.pre_ir = None;
        chain.post_ir = None;
        chain.direct_early_late_correction = None;
    }

    let ir_inputs: Vec<_> = result
        .channel_results
        .iter()
        .map(|(name, ch)| {
            let fir_coeffs = deployed_fir_coefficients(
                result.channels.get(name),
                ch.fir_coeffs.as_deref(),
                sidecar_dir,
                sample_rate,
            );
            (name.clone(), ch.initial_curve.clone(), fir_coeffs)
        })
        .collect();

    for (channel_name, initial_curve, fir_coeffs) in ir_inputs {
        if result
            .channels
            .get(&channel_name)
            .is_some_and(super::super::per_driver_fir::has_physical_fir)
        {
            // There is no common FIR to multiply onto the summed capture.
            // Render the acoustic post IR from the replayed complex response;
            // use the worst physical FIR for conservative temporal acceptance.
            let chain = result.channels.get(&channel_name).unwrap();
            let mut metrics = Vec::new();
            for driver in chain.drivers.as_deref().unwrap_or_default() {
                for plugin in &driver.plugins {
                    if plugin.plugin_type != "convolution" {
                        continue;
                    }
                    if let Some(path) = plugin.parameters.get("ir_file").and_then(|v| v.as_str())
                        && let Ok(wav) = crate::wav::decode_first_channel(&sidecar_dir.join(path))
                        && wav.sample_rate == sample_rate.round() as u32
                    {
                        let taps: Vec<f64> = wav.samples.into_iter().map(f64::from).collect();
                        if let Some(m) =
                            roomeq_engine::loss::epa::score::temporal_ir_masking_metrics(
                                &taps,
                                sample_rate,
                                &runtime_epa_cfg.temporal_masking,
                            )
                        {
                            metrics.push(roomeq_engine::report_adapter::to_temporal_ir_masking(m));
                        }
                    }
                }
            }
            let timing = super::super::parallel_timing::validate(chain, config, &initial_curve);
            let timing_available = timing.is_ok();
            let post = timing
                .map_err(|message| AutoeqError::InvalidConfiguration { message })
                .and_then(|()| {
                    super::super::per_driver_fir::replay(
                        chain,
                        &initial_curve,
                        sample_rate,
                        sidecar_dir,
                    )
                })
                .inspect_err(|error| {
                    waveform_errors.insert(channel_name.clone(), error.to_string());
                })
                .ok();
            // The group reference may itself be a synthesized coherent sum.
            // Without timing provenance, it is not an independent fallback.
            let pre_ir = timing_available
                .then(|| {
                    roomeq_engine::analysis::ir_waveform::compute_channel_ir_waveforms(
                        &initial_curve,
                        &[],
                        None,
                        0.0,
                        sample_rate,
                    )
                })
                .flatten()
                .map(|pair| pair.0);
            let pair = post.and_then(|curve| {
                roomeq_engine::analysis::ir_waveform::compute_channel_ir_waveforms_from_curves(
                    &initial_curve,
                    &curve,
                    sample_rate,
                )
            });
            let chain = result.channels.get_mut(&channel_name).unwrap();
            if let Some((pre, post)) = pair {
                chain.pre_ir = Some(pre);
                chain.post_ir = Some(post);
            } else {
                chain.pre_ir = pre_ir;
                chain.post_ir = None;
            }
            // This branch does not rebuild direct/early/late decomposition.
            // Never retain a result from a previous serial-chain realization.
            chain.direct_early_late_correction = None;
            chain.fir_temporal_masking = metrics.into_iter().max_by(|a, b| {
                a.pre_ringing_audible_db
                    .total_cmp(&b.pre_ringing_audible_db)
            });
            continue;
        }
        // Rebuild this channel's waveform pair or report it as unavailable.
        // In particular, missing phase must not retain an earlier pre/post IR
        // as if it were evidence for the current measurement and correction.
        if let Some(chain) = result.channels.get_mut(&channel_name) {
            chain.pre_ir = None;
            chain.post_ir = None;
            chain.direct_early_late_correction = None;
        }
        let waveforms = result.channels.get(&channel_name).and_then(|chain| {
            if chain
                .drivers
                .as_ref()
                .is_some_and(|drivers| !drivers.is_empty())
            {
                // Branch gains, delays, and EQ cannot be recovered from a PEQ
                // summary applied to a combined reference. Replay each capture
                // through its actual branch, then the common chain, exactly once.
                let post = super::super::parallel_timing::validate(
                    chain, config, &initial_curve,
                ).map_err(|message| AutoeqError::InvalidConfiguration { message }).and_then(|()| super::super::per_driver_fir::replay(
                    chain,
                    &initial_curve,
                    sample_rate,
                    sidecar_dir,
                ))
                .inspect_err(|error| {
                    waveform_errors.insert(channel_name.clone(), error.to_string());
                })
                .ok()?;
                return roomeq_engine::analysis::ir_waveform::compute_channel_ir_waveforms_from_curves(
                    &initial_curve,
                    &post,
                    sample_rate,
                );
            }
            let mut embedded = HashMap::new();
            let convolutions: Vec<_> = chain
                .plugins
                .iter()
                .filter(|plugin| plugin.plugin_type == "convolution")
                .collect();
            // A retained common kernel has an unambiguous owner only for a
            // single convolution. Otherwise resolve each emitted resource.
            if convolutions.len() == 1
                && let Some(taps) = fir_coeffs.as_ref()
                && let Some(path) = convolutions[0]
                    .parameters
                    .get("ir_file")
                    .and_then(|value| value.as_str())
            {
                embedded.insert(path.to_owned(), taps.clone());
            }
            roomeq_engine::analysis::ir_waveform::compute_channel_ir_waveforms_with_transfer(
                &initial_curve,
                sample_rate,
                |frequencies| match crate::ctc::channel_electrical_response_with_embedded_irs(
                    chain,
                    frequencies,
                    sample_rate,
                    sidecar_dir,
                    &embedded,
                ) {
                    Ok(response) => Some(response),
                    Err(error) => {
                        log::warn!("Serialized waveform unavailable for {channel_name}: {error}");
                        waveform_errors.insert(channel_name.clone(), error.to_string());
                        None
                    }
                },
            )
        });
        if let Some((pre_ir, post_ir)) = waveforms
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
    stamp_convolution_identities(result, sidecar_dir);
    super::waveform_status::record(result, &waveform_errors);
}

/// Stamp actual tap count and sample rate on every resolvable convolution.
///
/// WP6 reports the shipped bytes' own dimensions, not design-time intent:
/// each `ir_file` is decoded and its length and rate recorded alongside
/// the design delay. Unresolvable references stay unstamped (an explicit
/// absence later stages can see), never zero-filled.
fn stamp_convolution_identities(result: &mut RoomOptimizationResult, sidecar_dir: &Path) {
    for chain in result.channels.values_mut() {
        for plugin in chain
            .plugins
            .iter_mut()
            .chain(chain.drivers.iter_mut().flat_map(|drivers| {
                drivers
                    .iter_mut()
                    .flat_map(|driver| driver.plugins.iter_mut())
            }))
        {
            if plugin.plugin_type != "convolution" {
                continue;
            }
            let Some(ir_file) = plugin.parameters.get("ir_file").and_then(|v| v.as_str()) else {
                continue;
            };
            let path = Path::new(ir_file);
            let path = if path.is_relative() {
                sidecar_dir.join(path)
            } else {
                path.to_path_buf()
            };
            let Ok(wav) = crate::wav::decode_first_channel(&path) else {
                continue;
            };
            plugin.parameters["taps"] = serde_json::json!(wav.samples.len());
            plugin.parameters["sample_rate_hz"] = serde_json::json!(wav.sample_rate);
        }
    }
}

/// Compose every serial convolution, without substituting a partial resource set.
///
/// Retained taps have an unambiguous owner only for a single convolution.
/// This is FIR-only evidence: other DSP stages and alignment delays are separate.
fn deployed_fir_coefficients(
    chain: Option<&roomeq_model::ChannelDspChain>,
    retained: Option<&[f64]>,
    sidecar_dir: &Path,
    sample_rate: f64,
) -> Option<Vec<f64>> {
    let chain = chain?;
    let convolutions: Vec<_> = chain
        .plugins
        .iter()
        .filter(|plugin| plugin.plugin_type == "convolution")
        .collect();
    if convolutions.is_empty() {
        return None;
    }
    if convolutions.len() > 1
        && chain
            .plugins
            .iter()
            .any(|plugin| matches!(plugin.plugin_type.as_str(), "band_split" | "band_merge"))
    {
        // A parallel split must be replayed with its crossovers, not treated
        // as a cascade of every FIR appearing in its serialized plugin list.
        return None;
    }
    let mut kernels = Vec::with_capacity(convolutions.len());
    for plugin in &convolutions {
        let ir_file = plugin.parameters.get("ir_file")?.as_str()?;
        let taps = if convolutions.len() == 1
            && let Some(retained) = retained
        {
            retained.to_vec()
        } else {
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
            decoded.samples.into_iter().map(f64::from).collect()
        };
        if taps.is_empty() || taps.iter().any(|value| !value.is_finite()) {
            return None;
        }
        kernels.push(taps);
    }
    if kernels.len() == 1 {
        return kernels.pop();
    }
    // Bound analysis memory, not allowed acoustic latency. Oversized chains
    // have unavailable evidence rather than a truncated or wrapped response.
    const MAX_ANALYSIS_FFT: usize = 1 << 22;
    let length = kernels
        .iter()
        .try_fold(1_usize, |length, taps| length.checked_add(taps.len() - 1))?;
    let fft_size = length.checked_next_power_of_two()?;
    if fft_size > MAX_ANALYSIS_FFT {
        return None;
    }
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let forward = planner.plan_fft_forward(fft_size);
    let inverse = planner.plan_fft_inverse(fft_size);
    let mut product = vec![num_complex::Complex64::new(1.0, 0.0); fft_size];
    let mut buffer = vec![num_complex::Complex64::new(0.0, 0.0); fft_size];
    for taps in kernels {
        buffer.fill(num_complex::Complex64::new(0.0, 0.0));
        for (value, tap) in buffer.iter_mut().zip(taps) {
            value.re = tap;
        }
        forward.process(&mut buffer);
        for (value, factor) in product.iter_mut().zip(&buffer) {
            *value *= factor;
        }
    }
    inverse.process(&mut product);
    let taps: Vec<_> = product[..length]
        .iter()
        .map(|value| value.re / fft_size as f64)
        .collect();
    taps.iter().all(|value| value.is_finite()).then_some(taps)
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
    fn report_refresh_scores_routed_bass_correction_not_composite_programme_energy() {
        let mut result = single_channel_room_result("LFE");
        let initial = result.channel_results["LFE"].initial_curve.clone();
        let chain = result.channels.get_mut("LFE").unwrap();
        chain.plugins = vec![roomeq_engine::topology::mark_route_owned_plugin(
            roomeq_engine::output::create_crossover_plugin("LR24", 120.0, "low"),
        )];
        let mut composite =
            crate::room_optimization::room_optimization_result::routed_baseline_curve(
                chain, &initial, 48_000.0,
            )
            .unwrap();
        // Redirected inputs can add non-flat programme energy at the physical
        // sub. It is not the LFE input's correction transfer.
        for (f, level) in composite.freq.iter().zip(composite.spl.iter_mut()) {
            *level += 6.0 * (-(f / 80.0).ln().powi(2) / 0.3).exp();
        }
        result.channel_results.get_mut("LFE").unwrap().final_curve = composite;
        let mut config = RoomConfig::default();
        config.optimizer.min_freq = 20.0;
        config.optimizer.max_freq = 120.0;
        refresh_final_reports(&mut result, &config, 48_000.0, Path::new("."));
        let channel = &result.channel_results["LFE"];
        assert!(
            (channel.pre_score - channel.post_score).abs() < 1e-10,
            "route-only correction must remain identity after report refresh: {} -> {}",
            channel.pre_score,
            channel.post_score
        );
    }

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
        result.channels.get_mut("L").unwrap().plugins.push(
            roomeq_engine::output::create_convolution_plugin("retained.wav"),
        );
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
        result
            .channels
            .get_mut("L")
            .unwrap()
            .plugins
            .retain(|plugin| plugin.plugin_type != "convolution");
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
    fn temporal_ir_refresh_clears_waveforms_when_phase_evidence_is_unavailable() {
        let mut result = single_channel_room_result("L");
        let initial = &mut result.channel_results.get_mut("L").unwrap().initial_curve;
        initial.phase = Some(ndarray::Array1::zeros(initial.freq.len()));
        refresh_temporal_ir_evidence(
            &mut result,
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
        );
        assert!(result.channels["L"].pre_ir.is_some());
        assert!(result.channels["L"].post_ir.is_some());
        result
            .channel_results
            .get_mut("L")
            .unwrap()
            .initial_curve
            .phase = None;
        refresh_temporal_ir_evidence(
            &mut result,
            &RoomConfig::default(),
            48_000.0,
            Path::new("."),
        );
        assert!(
            result.channels["L"].pre_ir.is_none(),
            "stale measured-phase pre-IR survived"
        );
        assert!(
            result.channels["L"].post_ir.is_none(),
            "stale measured-phase post-IR survived"
        );
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
