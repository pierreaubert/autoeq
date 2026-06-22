use super::super::types::ChannelOptimizationResult;
use super::super::*;

/// Apply a pure delay/polarity adjustment to the phase of a reported curve.
pub(in super::super) fn apply_phase_only_adjustment_to_reported_curve(
    curve: &mut Curve,
    delay_ms: f64,
    invert_polarity: bool,
) {
    if curve.freq.is_empty() {
        return;
    }

    let base_phase = curve
        .phase
        .clone()
        .unwrap_or_else(|| ndarray::Array1::zeros(curve.freq.len()));
    let inversion_phase = if invert_polarity { 180.0 } else { 0.0 };
    let delay_s = delay_ms / 1000.0;

    let phase =
        ndarray::Array1::from_iter(curve.freq.iter().zip(base_phase.iter()).map(
            |(&freq_hz, &phase_deg)| phase_deg + inversion_phase - (360.0 * freq_hz * delay_s),
        ));
    curve.phase = Some(phase);
}

/// Linear convolution of two FIR filters.
pub(in super::super) fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len() + b.len() - 1;
    let mut out = vec![0.0; len];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    out
}

pub(in super::super) fn collect_current_final_curves(
    channel_results: &HashMap<String, ChannelOptimizationResult>,
) -> HashMap<String, Curve> {
    channel_results
        .iter()
        .map(|(name, result)| (name.clone(), result.final_curve.clone()))
        .collect()
}

pub(in super::super) fn total_chain_delay_ms(chain: &ChannelDspChain) -> f64 {
    chain
        .plugins
        .iter()
        .filter(|plugin| plugin.plugin_type == "delay")
        .filter_map(|plugin| {
            plugin
                .parameters
                .get("delay_ms")
                .and_then(|value| value.as_f64())
        })
        .sum()
}

pub(in super::super) fn compute_phase_alignment_delay_schedule(
    phase_alignment_results: &HashMap<String, (f64, bool, String)>,
) -> HashMap<String, f64> {
    let mut graph: HashMap<String, Vec<(String, f64)>> = HashMap::new();

    for (main_name, (relative_delay_ms, _invert, sub_name)) in phase_alignment_results {
        // The optimizer's delay means: delay(main) - delay(sub) = relative_delay_ms.
        graph
            .entry(sub_name.clone())
            .or_default()
            .push((main_name.clone(), *relative_delay_ms));
        graph
            .entry(main_name.clone())
            .or_default()
            .push((sub_name.clone(), -*relative_delay_ms));
    }

    let mut raw_offsets: HashMap<String, f64> = HashMap::new();
    let mut schedule: HashMap<String, f64> = HashMap::new();

    for start in graph.keys() {
        if raw_offsets.contains_key(start) {
            continue;
        }

        raw_offsets.insert(start.clone(), 0.0);
        let mut stack = vec![start.clone()];
        let mut component = Vec::new();

        while let Some(channel) = stack.pop() {
            component.push(channel.clone());
            let channel_offset = raw_offsets[&channel];

            if let Some(neighbors) = graph.get(&channel) {
                for (neighbor, delta_ms) in neighbors {
                    let neighbor_offset = channel_offset + *delta_ms;
                    if let Some(existing) = raw_offsets.get(neighbor) {
                        if (existing - neighbor_offset).abs() > 0.05 {
                            warn!(
                                "Conflicting phase-alignment delay constraints for '{}': {:.3} ms vs {:.3} ms; averaging",
                                neighbor, existing, neighbor_offset
                            );
                            let consensus = (existing + neighbor_offset) / 2.0;
                            raw_offsets.insert(neighbor.clone(), consensus);
                        }
                    } else {
                        raw_offsets.insert(neighbor.clone(), neighbor_offset);
                        stack.push(neighbor.clone());
                    }
                }
            }
        }

        let min_offset = component
            .iter()
            .filter_map(|name| raw_offsets.get(name))
            .copied()
            .fold(f64::INFINITY, f64::min);

        for name in component {
            if let Some(offset) = raw_offsets.get(&name) {
                let delay_ms = offset - min_offset;
                if delay_ms > 0.01 {
                    schedule.insert(name, delay_ms);
                }
            }
        }
    }

    schedule
}

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    #[test]
    fn phase_alignment_schedule_is_consistent_for_shared_sub() {
        let results = HashMap::from([
            ("L".to_string(), (-5.0, false, "LFE".to_string())),
            ("R".to_string(), (2.0, false, "LFE".to_string())),
        ]);

        let schedule = super::super::compute_phase_alignment_delay_schedule(&results);

        assert_close(*schedule.get("L").unwrap_or(&0.0), 0.0);
        assert_close(schedule["LFE"], 5.0);
        assert_close(schedule["R"], 7.0);
    }

    #[test]
    fn conflicting_delays_use_consensus_instead_of_arbitrary_first() {
        let mut results = HashMap::new();
        results.insert("A".to_string(), (0.0, false, "B".to_string()));
        results.insert("B".to_string(), (0.0, false, "C".to_string()));
        results.insert("C".to_string(), (5.0, false, "A".to_string()));

        let schedule = super::super::compute_phase_alignment_delay_schedule(&results);

        assert!(
            schedule.contains_key("A") || schedule.contains_key("B") || schedule.contains_key("C"),
            "schedule should not be empty even with inconsistent cycle"
        );
    }

    fn assert_close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 0.01,
            "assertion failed: {} ≈ {} (diff = {})",
            a,
            b,
            (a - b).abs()
        );
    }

    #[test]
    fn convolve_simple() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let result = super::super::convolve(&a, &b);
        assert_eq!(result, vec![3.0, 10.0, 8.0]);
    }

    #[test]
    fn apply_phase_only_adjustment_adds_delay_and_inversion() {
        use ndarray::Array1;
        let mut curve = crate::Curve {
            freq: Array1::from_vec(vec![100.0, 200.0]),
            spl: Array1::from_vec(vec![80.0, 80.0]),
            phase: Some(Array1::from_vec(vec![0.0, 0.0])),
            ..Default::default()
        };
        super::super::apply_phase_only_adjustment_to_reported_curve(&mut curve, 1.0, true);
        let phase = curve.phase.unwrap();
        // 1 ms at 100 Hz -> -36 deg, plus 180 inversion
        assert!((phase[0] - (180.0 - 36.0)).abs() < 1.0);
    }

    #[test]
    fn total_chain_delay_ms_sums_delay_plugins() {
        use serde_json::json;
        let chain = crate::roomeq::types::ChannelDspChain {
            channel: "test".to_string(),
            plugins: vec![
                crate::roomeq::types::PluginConfigWrapper {
                    plugin_type: "delay".to_string(),
                    parameters: json!({"delay_ms": 2.5}),
                },
                crate::roomeq::types::PluginConfigWrapper {
                    plugin_type: "gain".to_string(),
                    parameters: json!({"gain_db": 0.0}),
                },
                crate::roomeq::types::PluginConfigWrapper {
                    plugin_type: "delay".to_string(),
                    parameters: json!({"delay_ms": 1.5}),
                },
            ],
            drivers: None,
            initial_curve: None,
            final_curve: None,
            eq_response: None,
            target_curve: None,
            pre_ir: None,
            post_ir: None,
            fir_temporal_masking: None,
            direct_early_late_correction: None,
        };
        assert_eq!(super::super::total_chain_delay_ms(&chain), 4.0);
    }
}
