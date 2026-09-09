use super::super::*;
use super::misc::tag_group_delay_plugin;
use roomeq_engine::fir::{gd_delay_padding_samples, realize_gd_fir_delay};

/// Realize the whole delay schedule before changing any live chain. Common
/// padding also reaches channels without measured phase, so it cannot change
/// their timing relative to the alignment group. Each delay is a separate
/// pre-route stage, shared by all branches of its logical input.
pub(super) fn apply_phase_linear_delay_target(
    results: &mut HashMap<String, ChannelOptimizationResult>,
    chains: &mut HashMap<String, ChannelDspChain>,
    names: &[String],
    requested_ms: &[f64],
    polarities: &[bool],
    sample_rate: f64,
    directory: &Path,
) -> std::result::Result<Vec<(String, f64, bool)>, String> {
    if names.len() != requested_ms.len()
        || names.len() != polarities.len()
        || !sample_rate.is_finite()
        || sample_rate <= 0.0
        || requested_ms.iter().any(|d| !d.is_finite())
    {
        return Err("invalid group delay schedule".into());
    }
    let padding = gd_delay_padding_samples(requested_ms, sample_rate);
    for (index, name) in names.iter().enumerate() {
        if !results.contains_key(name) || names[..index].contains(name) {
            return Err(format!("unknown or duplicate GD channel '{name}'"));
        }
    }
    if padding > 0 && chains.keys().any(|name| !results.contains_key(name)) {
        return Err("common GD latency requires every playback channel result".into());
    }
    let mut channels: Vec<_> = results.keys().cloned().collect();
    channels.sort();
    let mut prepared = Vec::new();
    for name in channels {
        let index = names.iter().position(|n| n == &name);
        let delay = index.map_or(0.0, |i| requested_ms[i]);
        let inverted = index.is_some_and(|i| polarities[i]);
        if delay.abs() < 1e-9 && padding == 0 && !inverted {
            continue;
        }
        let chain = chains
            .get(&name)
            .ok_or_else(|| format!("missing DSP chain '{name}'"))?;
        let result = &results[&name];
        let shift = delay * sample_rate / 1000.0;
        if (shift - shift.round()).abs() > 1e-9
            && result.final_curve.freq.last().is_some_and(|frequency| {
                *frequency > sample_rate * roomeq_engine::fir::GD_DELAY_MAX_NORMALIZED_FREQUENCY
            })
        {
            return Err(format!(
                "'{name}' playback band exceeds supported fractional-delay bandwidth"
            ));
        }
        let mut kernel = realize_gd_fir_delay(&[1.0], delay, sample_rate, padding)?;
        if inverted {
            kernel.coefficients.iter_mut().for_each(|c| *c = -*c);
        }
        let combined = result
            .fir_coeffs
            .as_ref()
            .map(|coefficients| {
                let mut shifted = realize_gd_fir_delay(coefficients, delay, sample_rate, padding)?;
                if inverted {
                    shifted.coefficients.iter_mut().for_each(|c| *c = -*c);
                }
                Ok::<_, String>(shifted.coefficients)
            })
            .transpose()?;
        let existing_convolution = chain.plugins.iter().any(|p| p.plugin_type == "convolution");
        let payload = if existing_convolution {
            &kernel.coefficients
        } else {
            combined.as_ref().unwrap_or(&kernel.coefficients)
        }
        .clone();
        let (filename, path) = autoeq_artifacts::roomeq::reserve_convolution_artifact_path(
            directory,
            &name,
            autoeq_artifacts::roomeq::ConvolutionArtifactKind::Fir,
            sample_rate,
        );
        let mut plugin = tag_group_delay_plugin(
            roomeq_engine::output::create_convolution_plugin(&filename),
            "group_delay_phase_linear",
        );
        plugin.parameters["gd_requested_delay_ms"] = serde_json::json!(delay);
        plugin.parameters["gd_effective_delay_ms"] = serde_json::json!(kernel.effective_delay_ms);
        plugin.parameters["gd_common_padding_samples"] = serde_json::json!(padding);
        plugin.parameters["gd_usable_band_max_hz"] =
            serde_json::json!(sample_rate * roomeq_engine::fir::GD_DELAY_MAX_NORMALIZED_FREQUENCY);
        plugin.parameters["polarity_inverted"] = serde_json::json!(inverted);
        prepared.push((
            name,
            kernel,
            combined,
            existing_convolution,
            plugin,
            path,
            payload,
            inverted,
        ));
    }
    // Fresh sidecars leave the previous serialized artifact intact if any
    // write fails. No chain or reported response changes until all writes pass.
    for (_, _, _, _, _, path, payload, _) in &prepared {
        math_audio_iir_fir::save_fir_to_wav(payload, sample_rate as u32, path)
            .map_err(|error| format!("cannot write GD FIR '{}': {error}", path.display()))?;
    }
    let mut applied = Vec::new();
    for (name, kernel, combined, existing_convolution, plugin, _, _, inverted) in prepared {
        let result = results.get_mut(&name).expect("prepared result");
        let phase_known = result.final_curve.phase.is_some();
        result.fir_coeffs =
            combined.or_else(|| (!existing_convolution).then(|| kernel.coefficients.clone()));
        chains
            .get_mut(&name)
            .expect("prepared chain")
            .plugins
            .push(plugin);
        sync_reported_fir_adjustment(&name, results, chains, &kernel.coefficients, sample_rate);
        if !phase_known {
            results.get_mut(&name).unwrap().final_curve.phase = None;
            if let Some(curve) = chains.get_mut(&name).unwrap().final_curve.as_mut() {
                curve.phase = None;
            }
        }
        applied.push((name, kernel.effective_delay_ms, inverted));
    }
    Ok(applied)
}
