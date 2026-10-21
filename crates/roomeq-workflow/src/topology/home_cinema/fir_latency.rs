//! Match declared FIR scheduling delays before optimizing routed crossovers.

// Rust guideline compliant 2026-02-21
use roomeq_model::PluginConfigWrapper;
use std::collections::HashMap;

/// Add only the missing design delay; retain acoustic and steering delays.
///
/// Unknown convolution scheduling is left unassessed. No delay
/// is inferred from tap length, impulse peaks, or measured acoustic phase.
pub(super) fn align(
    outputs: &mut HashMap<String, Vec<PluginConfigWrapper>>,
    sample_rate: f64,
) -> Option<Vec<(String, f64)>> {
    let delays: Option<Vec<_>> = outputs
        .iter()
        .map(|(name, plugins)| Some((name.clone(), output_delay(plugins, sample_rate)?)))
        .collect();
    let delays = delays?;
    let maximum = delays.iter().map(|(_, delay)| *delay).fold(0.0, f64::max);
    let mut alignment = Vec::new();
    for (name, delay) in delays {
        let extra = maximum - delay;
        if extra <= 1e-9 {
            continue;
        }
        let mut plugin = roomeq_engine::output::create_delay_plugin(extra);
        plugin.parameters["label"] = serde_json::json!("fir_design_delay");
        plugin.parameters["room_eq_stage"] = serde_json::json!("post_route");
        outputs.get_mut(&name)?.push(plugin);
        alignment.push((name, extra));
    }
    alignment.sort_by(|a, b| a.0.cmp(&b.0));
    Some(alignment)
}

fn declared_delay(plugin: &PluginConfigWrapper, fs: f64) -> Option<f64> {
    let delay = if plugin.plugin_type == "convolution" {
        plugin.parameters["correction_design_delay_ms"]
            .as_f64()
            .or_else(|| {
                plugin.parameters["latency_samples"]
                    .as_u64()
                    .map(|samples| samples as f64 * 1000.0 / fs)
            })?
    } else {
        plugin.parameters["delay_ms"].as_f64()?
    };
    (delay.is_finite() && delay >= 0.0).then_some(delay)
}

fn serial_delay(plugins: &[PluginConfigWrapper], fs: f64) -> Option<f64> {
    let mut delay = 0.0;
    for plugin in plugins {
        if matches!(plugin.plugin_type.as_str(), "band_split" | "band_merge") {
            return None;
        }
        if plugin.plugin_type == "convolution"
            || (plugin.plugin_type == "delay" && plugin.parameters["label"] == "fir_design_delay")
        {
            if plugin.parameters.get("channels").is_some() {
                return None;
            }
            delay += declared_delay(plugin, fs)?;
        }
    }
    delay.is_finite().then_some(delay)
}

fn output_delay(plugins: &[PluginConfigWrapper], fs: f64) -> Option<f64> {
    let Some(start) = plugins
        .iter()
        .position(|plugin| plugin.plugin_type == "band_split")
    else {
        return serial_delay(plugins, fs);
    };
    let end = plugins
        .iter()
        .position(|plugin| plugin.plugin_type == "band_merge")?;
    if end <= start {
        return None;
    }
    let band = &plugins[start + 1..end];
    if band
        .iter()
        .any(|plugin| matches!(plugin.plugin_type.as_str(), "band_split" | "band_merge"))
    {
        return None;
    }
    let firs: Vec<_> = band
        .iter()
        .filter(|plugin| plugin.plugin_type == "convolution")
        .collect();
    let [fir] = firs.as_slice() else {
        return None;
    };
    let complement = match fir.parameters["channels"].as_array()?.as_slice() {
        [a, b] if a == 0 && b == 1 => serde_json::json!([2, 3]),
        [a, b] if a == 2 && b == 3 => serde_json::json!([0, 1]),
        _ => return None,
    };
    let delays: Vec<_> = band
        .iter()
        .filter(|plugin| plugin.plugin_type == "delay")
        .collect();
    let opposite = match delays.as_slice() {
        [] => 0.0,
        [delay]
            if delay.parameters["label"] == "hybrid_fir_latency_alignment"
                && delay.parameters["channels"] == complement =>
        {
            declared_delay(delay, fs)?
        }
        _ => return None,
    };
    // The two split bands are parallel, so their delays do not add. Both
    // serial sections still contribute, including a previous output match.
    let total = serial_delay(&plugins[..start], fs)?
        + declared_delay(fir, fs)?.max(opposite)
        + serial_delay(&plugins[end + 1..], fs)?;
    total.is_finite().then_some(total)
}

/// Remove a common known delay only from optimizer samples, before interpolation.
pub(super) fn optimizer_curves(
    curves: &HashMap<String, roomeq_model::Curve>,
    outputs: &HashMap<String, Vec<PluginConfigWrapper>>,
    fs: f64,
) -> Option<HashMap<String, roomeq_model::Curve>> {
    let delays: Option<Vec<_>> = curves
        .keys()
        .map(|name| output_delay(outputs.get(name)?, fs))
        .collect();
    let delays = delays?;
    let delay = delays.first().copied().unwrap_or(0.0);
    if delays.iter().any(|other| (other - delay).abs() > 1e-6) {
        return None;
    }
    if delay <= 1e-9 {
        return Some(curves.clone());
    }
    let mut corrected = curves.clone();
    for curve in corrected.values_mut() {
        if let Some(phase) = &mut curve.phase {
            for (phase, frequency) in phase.iter_mut().zip(&curve.freq) {
                *phase =
                    (*phase + 360.0 * frequency * delay / 1000.0 + 180.0).rem_euclid(360.0) - 180.0;
            }
        }
    }
    Some(corrected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_output_delay_preserves_the_realized_main_sub_phase() {
        let fs = 48_000.0;
        let mut fir = roomeq_engine::output::create_convolution_plugin("main.wav");
        fir.parameters["correction_design_delay_ms"] = serde_json::json!(32.0 * 1000.0 / fs);
        let mut outputs = HashMap::from([("L".into(), vec![fir]), ("Sub".into(), vec![])]);
        let added = align(&mut outputs, fs).unwrap();
        assert_eq!(added, vec![("Sub".into(), 32.0 * 1000.0 / fs)]);
        assert!(align(&mut outputs, fs).unwrap().is_empty());
        let input = roomeq_model::Curve {
            freq: ndarray::array![20.0, 80.0, 123.0, 1000.0],
            spl: ndarray::Array1::zeros(4),
            phase: Some(ndarray::Array1::zeros(4)),
            ..Default::default()
        };
        let mut taps = vec![0.0; 64];
        taps[32] = 1.0;
        let mut realized = Vec::new();
        for name in ["L", "Sub"] {
            let mut result = crate::test_fixtures::single_channel_room_result(name);
            let chain = result.channels.get_mut(name).unwrap();
            chain.plugins = outputs[name].clone();
            let irs = super::super::embedded_convolution_irs(
                &chain.plugins,
                (name == "L").then_some(taps.as_slice()),
            )
            .unwrap();
            realized.push(
                crate::ctc::apply_channel_dsp_chain_to_curve_with_embedded_irs(
                    chain,
                    &input,
                    fs,
                    std::path::Path::new("/unused"),
                    &irs,
                )
                .unwrap(),
            );
        }
        for (a, b) in realized[0]
            .phase
            .as_ref()
            .unwrap()
            .iter()
            .zip(realized[1].phase.as_ref().unwrap())
        {
            assert!((a - b).to_radians().sin().abs() < 1e-9);
            assert!(((a - b).to_radians().cos() - 1.0).abs() < 1e-9);
        }
        assert!(
            (&realized[0].spl - &realized[1].spl)
                .iter()
                .all(|db| db.abs() < 1e-9)
        );
    }

    #[test]
    fn physical_delay_is_kept_and_equal_or_minimum_phase_outputs_add_nothing() {
        let physical = roomeq_engine::output::create_delay_plugin(2.0);
        let mut outputs =
            HashMap::from([("L".into(), vec![physical.clone()]), ("Sub".into(), vec![])]);
        assert!(align(&mut outputs, 48_000.0).unwrap().is_empty());
        assert_eq!(outputs["L"][0].parameters, physical.parameters);
        assert!(outputs["Sub"].is_empty());
    }

    #[test]
    fn canonical_split_bands_contribute_their_maximum_not_summed_delay() {
        let chain = roomeq_engine::output::build_mixed_mode_crossover_chain(
            "L",
            &roomeq_model::MixedModeConfig::default(),
            &[],
            "main.wav",
            false,
            10.0,
            None,
        );
        let mut plugins = chain.plugins;
        let fir = plugins
            .iter_mut()
            .find(|plugin| plugin.plugin_type == "convolution")
            .unwrap();
        fir.parameters["correction_design_delay_ms"] = serde_json::json!(10.0);
        let mut outputs = HashMap::from([("L".into(), plugins), ("Sub".into(), vec![])]);
        assert_eq!(
            align(&mut outputs, 48_000.0).unwrap(),
            vec![("Sub".into(), 10.0)]
        );
        assert!(align(&mut outputs, 48_000.0).unwrap().is_empty());
        outputs
            .get_mut("L")
            .unwrap()
            .iter_mut()
            .find(|plugin| plugin.parameters["label"] == "hybrid_fir_latency_alignment")
            .unwrap()
            .parameters["channels"] = serde_json::json!([2, 3]);
        assert!(align(&mut outputs, 48_000.0).is_none());
    }

    #[test]
    fn optimizer_demodulates_known_common_delay_without_changing_emitted_curves() {
        let fs = 48_000.0;
        let delay = 85.33333333333333;
        let freq = ndarray::array![20.0, 80.0, 300.0, 1000.0];
        let mut outputs = HashMap::new();
        let mut curves = HashMap::new();
        for (name, residual) in [("L", vec![10.0, 0.0, -90.0, 90.0]), ("Sub", vec![0.0; 4])] {
            let mut fir = roomeq_engine::output::create_convolution_plugin("known.wav");
            fir.parameters["correction_design_delay_ms"] = serde_json::json!(delay);
            outputs.insert(name.into(), vec![fir]);
            curves.insert(
                name.into(),
                roomeq_model::Curve {
                    freq: freq.clone(),
                    spl: ndarray::Array1::zeros(4),
                    phase: Some(ndarray::Array1::from_iter(freq.iter().zip(residual).map(
                        |(f, phase)| {
                            (phase - 360.0 * f * delay / 1000.0 + 180.0).rem_euclid(360.0) - 180.0
                        },
                    ))),
                    ..Default::default()
                },
            );
        }
        let original = curves["L"].phase.clone();
        let corrected = optimizer_curves(&curves, &outputs, fs).unwrap();
        for (actual, expected) in corrected["L"]
            .phase
            .as_ref()
            .unwrap()
            .iter()
            .zip([10.0, 0.0, -90.0, 90.0])
        {
            assert!((actual - expected).abs() < 1e-9);
        }
        assert!(
            corrected["Sub"]
                .phase
                .as_ref()
                .unwrap()
                .iter()
                .all(|phase| phase.abs() < 1e-9)
        );
        assert_eq!(curves["L"].phase, original);
    }

    #[test]
    fn zero_design_delay_preserves_measured_unwrapped_phase() {
        let curve = roomeq_model::Curve {
            freq: ndarray::array![20.0, 80.0, 300.0],
            spl: ndarray::Array1::zeros(3),
            phase: Some(ndarray::array![0.0, -500.0, -2000.0]),
            ..Default::default()
        };
        let curves = HashMap::from([("L".into(), curve.clone())]);
        let outputs = HashMap::from([("L".into(), vec![])]);
        assert_eq!(
            optimizer_curves(&curves, &outputs, 48_000.0).unwrap()["L"].phase,
            curve.phase
        );
    }

    #[test]
    fn missing_or_parallel_fir_delay_cannot_invent_alignment() {
        for parallel in [false, true] {
            let mut fir = roomeq_engine::output::create_convolution_plugin("main.wav");
            if parallel {
                fir.parameters["correction_design_delay_ms"] = serde_json::json!(10.0);
                fir.parameters["channels"] = serde_json::json!([0, 1]);
            }
            let mut outputs = HashMap::from([("L".into(), vec![fir]), ("Sub".into(), vec![])]);
            assert!(align(&mut outputs, 48_000.0).is_none());
            assert!(outputs["Sub"].is_empty());
        }
    }
}
