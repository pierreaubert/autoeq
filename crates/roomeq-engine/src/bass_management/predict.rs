use crate::Curve;
use crate::home_cinema::BassManagementRoutingGraph;
use crate::topology::{
    complex_sum_mains, compute_crossover_complex_response, curve_has_usable_phase,
    same_frequency_grid,
};
use autoeq_core::interpolate_log_space;
use std::collections::HashMap;

/// Predict the physical-sub contribution made by one logical input channel.
///
/// Unlike [`predict_bass_output_curve_from_routes`], this does not sum the
/// shared bass bus. It is the response needed to evaluate the acoustic output
/// a microphone records when only `source_channel` is driven.
pub fn predict_bass_source_curve_from_routes(
    sub_curve: &Curve,
    source_pre_route_transfer: Option<&Curve>,
    graph: &BassManagementRoutingGraph,
    source_channel: &str,
    sample_rate: f64,
) -> Option<Curve> {
    use num_complex::Complex;

    let pre_route_transfer_realized = source_pre_route_transfer.is_some();
    let source_sub_curve = apply_source_pre_route_transfer(sub_curve, source_pre_route_transfer)?;
    if !curve_has_usable_phase(&source_sub_curve) {
        return None;
    }
    let phase = source_sub_curve.phase.as_ref()?;
    let mut complex_sum = vec![Complex::new(0.0, 0.0); source_sub_curve.freq.len()];
    let mut any_route = false;

    for route in graph.routes.iter().filter(|route| {
        route.source_channel == source_channel
            && (route.route_kind == "redirected_bass_lowpass_to_sub"
                || route.route_kind == "lfe_lowpass_to_sub")
    }) {
        any_route = true;
        let response = if let Some(freq) = route.low_pass_hz {
            compute_crossover_complex_response(
                &route.crossover_type,
                freq,
                sample_rate,
                true,
                &source_sub_curve.freq,
            )
        } else {
            vec![Complex::new(1.0, 0.0); source_sub_curve.freq.len()]
        };
        let polarity_phase = if route.polarity_inverted { 180.0 } else { 0.0 };

        for idx in 0..source_sub_curve.freq.len() {
            let delay_phase = -360.0 * source_sub_curve.freq[idx] * route.delay_ms / 1000.0;
            let input_trim_db = if pre_route_transfer_realized {
                0.0
            } else {
                graph
                    .input_trim_db
                    .get(&route.source_channel)
                    .copied()
                    .unwrap_or(0.0)
            };
            let magnitude =
                10.0_f64.powf((source_sub_curve.spl[idx] + route.gain_db + input_trim_db) / 20.0);
            let phase_rad = (phase[idx] + delay_phase + polarity_phase).to_radians();
            complex_sum[idx] += Complex::from_polar(magnitude, phase_rad) * response[idx];
        }
    }

    if !any_route {
        return None;
    }
    let mut spl = ndarray::Array1::<f64>::zeros(source_sub_curve.freq.len());
    let mut output_phase = ndarray::Array1::<f64>::zeros(source_sub_curve.freq.len());
    for (idx, value) in complex_sum.iter().enumerate() {
        spl[idx] = 20.0 * value.norm().max(1e-12).log10();
        output_phase[idx] = value.arg().to_degrees();
    }
    Some(Curve {
        freq: source_sub_curve.freq.clone(),
        spl,
        phase: Some(output_phase),
        ..source_sub_curve
    })
}

/// Predict the physical-sub contribution made by one logical input when each
/// physical output has its own measured acoustic transfer.
///
/// Route gain, delay, polarity, and low-pass are applied once per matrix edge
/// before the destination responses are coherently summed.  This is the
/// authoritative multi-output counterpart of
/// [`predict_bass_source_curve_from_routes`]; using one already-summed sub
/// curve for every edge would duplicate physical outputs and erase the
/// distinction between direct, crossed, and dual-mono stereo routing.
pub fn predict_bass_source_curve_from_output_routes(
    reference_curve: &Curve,
    output_base_curves: &HashMap<String, Curve>,
    fallback_curve: &Curve,
    source_pre_route_transfer: Option<&Curve>,
    graph: &BassManagementRoutingGraph,
    source_channel: &str,
    sample_rate: f64,
) -> Option<Curve> {
    use num_complex::Complex;

    let pre_route_transfer_realized = source_pre_route_transfer.is_some();
    let mut complex_sum = vec![Complex::new(0.0, 0.0); reference_curve.freq.len()];
    let mut any_route = false;

    for route in graph.routes.iter().filter(|route| {
        route.source_channel == source_channel
            && matches!(
                route.route_kind.as_str(),
                "redirected_bass_lowpass_to_sub" | "lfe_lowpass_to_sub"
            )
    }) {
        let base_curve = match output_base_curves.get(&route.destination) {
            Some(curve) => curve,
            None if graph.physical_sub_outputs.len() <= 1 => fallback_curve,
            None => return None,
        };
        let base_curve = if same_frequency_grid(&reference_curve.freq, &base_curve.freq) {
            base_curve.clone()
        } else {
            interpolate_log_space(&reference_curve.freq, base_curve)
        };
        let route_curve = apply_source_pre_route_transfer(&base_curve, source_pre_route_transfer)?;
        if !curve_has_usable_phase(&route_curve) {
            return None;
        }
        let phase = route_curve.phase.as_ref()?;
        let response = if let Some(freq) = route.low_pass_hz {
            compute_crossover_complex_response(
                &route.crossover_type,
                freq,
                sample_rate,
                true,
                &reference_curve.freq,
            )
        } else {
            vec![Complex::new(1.0, 0.0); reference_curve.freq.len()]
        };
        let polarity_phase = if route.polarity_inverted { 180.0 } else { 0.0 };
        let input_trim_db = if pre_route_transfer_realized {
            0.0
        } else {
            graph
                .input_trim_db
                .get(&route.source_channel)
                .copied()
                .unwrap_or(0.0)
        };
        any_route = true;
        for idx in 0..reference_curve.freq.len() {
            let delay_phase = -360.0 * reference_curve.freq[idx] * route.delay_ms / 1000.0;
            let magnitude =
                10.0_f64.powf((route_curve.spl[idx] + route.gain_db + input_trim_db) / 20.0);
            let phase_rad = (phase[idx] + delay_phase + polarity_phase).to_radians();
            complex_sum[idx] += Complex::from_polar(magnitude, phase_rad) * response[idx];
        }
    }

    if !any_route {
        return None;
    }
    let mut spl = ndarray::Array1::<f64>::zeros(reference_curve.freq.len());
    let mut phase = ndarray::Array1::<f64>::zeros(reference_curve.freq.len());
    for (idx, value) in complex_sum.iter().enumerate() {
        spl[idx] = 20.0 * value.norm().max(1e-12).log10();
        phase[idx] = value.arg().to_degrees();
    }
    Some(Curve {
        freq: reference_curve.freq.clone(),
        spl,
        phase: Some(phase),
        ..Default::default()
    })
}

/// Apply the logical input's pre-route DSP transfer to the physical-sub
/// acoustic response before route-owned crossover, gain, delay and polarity.
///
/// The realized main branch already contains this transfer. Applying it to the
/// sub branch as well preserves the signal graph's split-point semantics.
pub fn apply_source_pre_route_transfer(
    sub_curve: &Curve,
    source_pre_route_transfer: Option<&Curve>,
) -> Option<Curve> {
    let Some(transfer) = source_pre_route_transfer else {
        return Some(sub_curve.clone());
    };
    let aligned_transfer;
    let transfer = if same_frequency_grid(&sub_curve.freq, &transfer.freq)
        && sub_curve.spl.len() == transfer.spl.len()
    {
        transfer
    } else {
        aligned_transfer = interpolate_log_space(&sub_curve.freq, transfer);
        &aligned_transfer
    };
    if sub_curve.spl.len() != transfer.spl.len() {
        return None;
    }

    let mut conditioned = sub_curve.clone();
    conditioned.spl = &conditioned.spl + &transfer.spl;
    // The product H_acoustic x H_electrical has known phase only when both
    // factors do. An unknown acoustic phase must stay unknown even when the
    // electrical transfer contributes phase (F03).
    match (conditioned.phase.as_mut(), transfer.phase.as_ref()) {
        (Some(phase), Some(transfer_phase)) if phase.len() == transfer_phase.len() => {
            *phase = &*phase + transfer_phase;
        }
        (None, Some(_)) => {}
        (Some(_), Some(_)) => return None,
        (Some(_), None) | (None, None) => {}
    }
    Some(conditioned)
}

/// Predict the complete acoustic response of one logical input after routing.
///
/// A main input is the coherent sum of its high-pass main branch and only its
/// own low-pass physical-sub branch. LFE has no main branch and therefore
/// returns its routed sub contribution alone. This intentionally never sums
/// unrelated logical inputs from the shared physical bass bus.
pub fn predict_deployed_source_curve_from_routes(
    main_curve: Option<&Curve>,
    sub_curve: &Curve,
    source_pre_route_transfer: Option<&Curve>,
    graph: &BassManagementRoutingGraph,
    source_channel: &str,
    sample_rate: f64,
) -> Option<Curve> {
    match (
        main_curve,
        predict_bass_source_curve_from_routes(
            sub_curve,
            source_pre_route_transfer,
            graph,
            source_channel,
            sample_rate,
        ),
    ) {
        (Some(main), Some(sub)) => {
            let sub = crate::topology::interpolate_bass_response(&main.freq, &sub);
            Some(complex_sum_mains(&[main, &sub]))
        }
        (Some(main), None) => Some(main.clone()),
        (None, Some(sub)) => Some(sub),
        (None, None) => None,
    }
}

pub fn predict_bass_output_curve_from_routes(
    sub_curve: &Curve,
    source_pre_route_transfers: Option<&HashMap<String, Curve>>,
    graph: &BassManagementRoutingGraph,
    output_role: &str,
    sample_rate: f64,
) -> Option<Curve> {
    use num_complex::Complex;

    if !curve_has_usable_phase(sub_curve) {
        return None;
    }
    let mut complex_sum = vec![Complex::new(0.0, 0.0); sub_curve.freq.len()];
    let mut any_route = false;
    for route in graph.routes.iter().filter(|route| {
        route.destination == output_role
            && (route.route_kind == "redirected_bass_lowpass_to_sub"
                || route.route_kind == "lfe_lowpass_to_sub")
    }) {
        any_route = true;
        let source_pre_route_transfer =
            source_pre_route_transfers.and_then(|transfers| transfers.get(&route.source_channel));
        let route_curve = apply_source_pre_route_transfer(sub_curve, source_pre_route_transfer)?;
        let phase = route_curve.phase.as_ref()?;
        let response = if let Some(freq) = route.low_pass_hz {
            compute_crossover_complex_response(
                &route.crossover_type,
                freq,
                sample_rate,
                true,
                &sub_curve.freq,
            )
        } else {
            vec![Complex::new(1.0, 0.0); sub_curve.freq.len()]
        };
        let polarity_phase = if route.polarity_inverted { 180.0 } else { 0.0 };
        for idx in 0..sub_curve.freq.len() {
            let freq_hz = sub_curve.freq[idx];
            let delay_phase = -360.0 * freq_hz * route.delay_ms / 1000.0;
            let input_trim_db = if source_pre_route_transfer.is_some() {
                0.0
            } else {
                graph
                    .input_trim_db
                    .get(&route.source_channel)
                    .copied()
                    .unwrap_or(0.0)
            };
            let mag = 10.0_f64.powf((route_curve.spl[idx] + route.gain_db + input_trim_db) / 20.0);
            let phase_rad = (phase[idx] + delay_phase + polarity_phase).to_radians();
            complex_sum[idx] += Complex::from_polar(mag, phase_rad) * response[idx];
        }
    }
    if !any_route {
        return None;
    }

    let mut spl = ndarray::Array1::<f64>::zeros(sub_curve.freq.len());
    let mut output_phase = ndarray::Array1::<f64>::zeros(sub_curve.freq.len());
    for (idx, value) in complex_sum.iter().enumerate() {
        spl[idx] = 20.0 * value.norm().max(1e-12).log10();
        output_phase[idx] = value.arg().to_degrees();
    }
    Some(Curve {
        freq: sub_curve.freq.clone(),
        spl,
        phase: Some(output_phase),
        ..Default::default()
    })
}

pub fn predict_bass_bus_curve_from_routes(
    reference_curve: &Curve,
    graph: &BassManagementRoutingGraph,
    output_base_curves: &HashMap<String, Curve>,
    fallback_curve: &Curve,
    sample_rate: f64,
) -> Option<Curve> {
    use num_complex::Complex;

    if !curve_has_usable_phase(reference_curve) {
        return None;
    }

    let mut complex_sum = vec![Complex::new(0.0, 0.0); reference_curve.freq.len()];
    let mut any_route = false;
    for route in graph.routes.iter().filter(|route| {
        route.route_kind == "redirected_bass_lowpass_to_sub"
            || route.route_kind == "lfe_lowpass_to_sub"
    }) {
        let base_curve = output_base_curves
            .get(&route.destination)
            .unwrap_or(fallback_curve);
        if !curve_has_usable_phase(base_curve) {
            continue;
        }
        let curve = if same_frequency_grid(&reference_curve.freq, &base_curve.freq) {
            base_curve.clone()
        } else {
            interpolate_log_space(&reference_curve.freq, base_curve)
        };
        let Some(phase) = curve.phase.as_ref() else {
            continue;
        };
        any_route = true;
        let response = if let Some(freq) = route.low_pass_hz {
            compute_crossover_complex_response(
                &route.crossover_type,
                freq,
                sample_rate,
                true,
                &reference_curve.freq,
            )
        } else {
            vec![Complex::new(1.0, 0.0); reference_curve.freq.len()]
        };
        let polarity_phase = if route.polarity_inverted { 180.0 } else { 0.0 };
        for idx in 0..reference_curve.freq.len() {
            let freq_hz = reference_curve.freq[idx];
            let delay_phase = -360.0 * freq_hz * route.delay_ms / 1000.0;
            let input_trim_db = graph
                .input_trim_db
                .get(&route.source_channel)
                .copied()
                .unwrap_or(0.0);
            let mag = 10.0_f64.powf((curve.spl[idx] + route.gain_db + input_trim_db) / 20.0);
            let phase_rad = (phase[idx] + delay_phase + polarity_phase).to_radians();
            complex_sum[idx] += Complex::from_polar(mag, phase_rad) * response[idx];
        }
    }
    if !any_route {
        return None;
    }

    let mut spl = ndarray::Array1::<f64>::zeros(reference_curve.freq.len());
    let mut output_phase = ndarray::Array1::<f64>::zeros(reference_curve.freq.len());
    for (idx, value) in complex_sum.iter().enumerate() {
        spl[idx] = 20.0 * value.norm().max(1e-12).log10();
        output_phase[idx] = value.arg().to_degrees();
    }
    Some(Curve {
        freq: reference_curve.freq.clone(),
        spl,
        phase: Some(output_phase),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::apply_crossover_response_to_curve;
    use ndarray::Array1;
    use roomeq_model::BassManagementRoute;

    fn flat_curve() -> Curve {
        let freq = Array1::linspace(20.0, 500.0, 481);
        Curve {
            spl: Array1::zeros(freq.len()),
            phase: Some(Array1::zeros(freq.len())),
            freq,
            ..Curve::default()
        }
    }

    fn transfer_curve(reference: &Curve, gain_db: f64, delay_ms: f64) -> Curve {
        Curve {
            freq: reference.freq.clone(),
            spl: Array1::from_elem(reference.freq.len(), gain_db),
            phase: Some(
                reference
                    .freq
                    .mapv(|frequency| -360.0 * frequency * delay_ms / 1_000.0),
            ),
            ..Curve::default()
        }
    }

    fn routing_graph(crossover_type: &str, input_trim_db: f64) -> BassManagementRoutingGraph {
        BassManagementRoutingGraph {
            physical_sub_output: "LFE".to_string(),
            physical_sub_outputs: Vec::new(),
            stereo_routing: None,
            input_channels: vec!["L".to_string()],
            output_channels: vec!["L".to_string(), "LFE".to_string()],
            routes: vec![BassManagementRoute {
                group_id: Some("lcr".to_string()),
                source_channel: "L".to_string(),
                source_index: 0,
                destination: "LFE".to_string(),
                destination_index: 1,
                pre_chain_channel: Some("LFE".to_string()),
                post_chain_channel: Some("LFE".to_string()),
                route_kind: "redirected_bass_lowpass_to_sub".to_string(),
                crossover_type: crossover_type.to_string(),
                high_pass_hz: None,
                low_pass_hz: Some(80.0),
                gain_db: 0.0,
                gain_linear: 1.0,
                matrix_gain: 1.0,
                delay_ms: 0.0,
                polarity_inverted: false,
            }],
            matrix: None,
            input_trim_db: HashMap::from([("L".to_string(), input_trim_db)]),
            post_dsp_main_alignment_band_hz: None,
            advisories: Vec::new(),
        }
    }

    #[test]
    fn source_pre_route_transfer_is_common_to_both_crossover_branches() {
        for sample_rate in [48_000.0, 96_000.0] {
            for crossover_type in ["LR24", "LR48"] {
                let base = flat_curve();
                // The canonical serialized transfer includes both correction
                // and the logical-input trim. routing_graph.input_trim_db is
                // metadata and must not apply that trim a second time.
                let transfer = transfer_curve(&base, -5.5, 4.25);
                let main_highpass = apply_crossover_response_to_curve(
                    &base,
                    crossover_type,
                    80.0,
                    sample_rate,
                    false,
                );
                let main = apply_source_pre_route_transfer(&main_highpass, Some(&transfer))
                    .expect("shared transfer grid");

                let deployed = predict_deployed_source_curve_from_routes(
                    Some(&main),
                    &base,
                    Some(&transfer),
                    &routing_graph(crossover_type, -3.0),
                    "L",
                    sample_rate,
                )
                .expect("phase-coherent routed source");

                for (frequency, level) in deployed.freq.iter().zip(deployed.spl.iter()) {
                    if (40.0..=160.0).contains(frequency) {
                        assert!(
                            (*level + 5.5).abs() <= 0.02,
                            "{crossover_type} at {sample_rate} Hz reconstructed {level:.4} dB at {frequency:.1} Hz"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn output_route_replay_uses_each_physical_measurement_once() {
        let reference = flat_curve();
        let sub1 = flat_curve();
        let mut sub2 = flat_curve();
        sub2.spl.fill(-20.0);

        let mut graph = routing_graph("LR24", 0.0);
        graph.physical_sub_output = "Sub1".to_string();
        graph.physical_sub_outputs = vec!["Sub1".to_string(), "Sub2".to_string()];
        graph.output_channels = vec!["L".to_string(), "Sub1".to_string(), "Sub2".to_string()];
        let mut sub1_route = graph.routes[0].clone();
        sub1_route.destination = "Sub1".to_string();
        sub1_route.destination_index = 1;
        sub1_route.pre_chain_channel = Some("L".to_string());
        sub1_route.post_chain_channel = Some("Sub1".to_string());
        sub1_route.low_pass_hz = None;
        sub1_route.gain_db = 20.0 * 0.5_f64.log10();
        sub1_route.gain_linear = 0.5;
        sub1_route.matrix_gain = 0.5;
        let mut sub2_route = sub1_route.clone();
        sub2_route.destination = "Sub2".to_string();
        sub2_route.destination_index = 2;
        sub2_route.post_chain_channel = Some("Sub2".to_string());
        graph.routes = vec![sub1_route, sub2_route];

        let outputs = HashMap::from([("Sub1".to_string(), sub1), ("Sub2".to_string(), sub2)]);
        let replay = predict_bass_source_curve_from_output_routes(
            &reference, &outputs, &reference, None, &graph, "L", 48_000.0,
        )
        .expect("both physical outputs have measured responses");
        let expected = 20.0 * (0.5_f64 + 0.05).log10();
        assert!(
            replay
                .spl
                .iter()
                .all(|level| (*level - expected).abs() < 1.0e-9),
            "dual-mono replay must sum 0.5*Sub1 + 0.5*Sub2 exactly once"
        );

        let missing = HashMap::from([("Sub1".to_string(), flat_curve())]);
        assert!(
            predict_bass_source_curve_from_output_routes(
                &reference, &missing, &reference, None, &graph, "L", 48_000.0,
            )
            .is_none(),
            "multi-output replay must not substitute an aggregate fallback for a missing output"
        );
    }

    #[test]
    fn magnitude_only_sub_stays_ineligible_after_pre_route_transfer() {
        // F03: unknown acoustic phase must not be upgraded to known phase by
        // the electrical pre-route transfer.
        use ndarray::Array1;
        let freq = Array1::from_vec(vec![40.0, 80.0, 160.0]);
        let sub = Curve {
            freq: freq.clone(),
            spl: Array1::from_elem(3, 80.0),
            phase: None,
            ..Curve::default()
        };
        let transfer = Curve {
            freq: freq.clone(),
            spl: Array1::zeros(3),
            phase: Some(Array1::from_elem(3, -90.0)),
            ..Curve::default()
        };
        let conditioned = apply_source_pre_route_transfer(&sub, Some(&transfer))
            .expect("conditioning should succeed");
        assert!(
            conditioned.phase.is_none(),
            "magnitude-only acoustic phase was invented: {:?}",
            conditioned.phase
        );
        let graph = routing_graph("LR24", 0.0);
        assert!(
            predict_bass_source_curve_from_routes(&sub, Some(&transfer), &graph, "L", 48_000.0)
                .is_none(),
            "magnitude-only sub must remain ineligible for coherent prediction"
        );
    }

    #[test]
    fn mismatched_source_transfer_grid_is_aligned() {
        let base = flat_curve();
        let transfer_freq = Array1::linspace(20.0, 500.0, 241);
        let transfer = Curve {
            spl: Array1::from_elem(transfer_freq.len(), 3.0),
            phase: Some(Array1::zeros(transfer_freq.len())),
            freq: transfer_freq,
            ..Curve::default()
        };
        let conditioned = apply_source_pre_route_transfer(&base, Some(&transfer))
            .expect("transfer should be aligned to the sub grid");

        assert!(same_frequency_grid(&conditioned.freq, &base.freq));
        assert_eq!(conditioned.spl.len(), base.spl.len());
        assert!(
            conditioned
                .spl
                .iter()
                .all(|level| (*level - 3.0).abs() < 1.0e-9)
        );
        assert!(
            conditioned
                .phase
                .as_ref()
                .is_some_and(|phase| phase.iter().all(|value| value.abs() < 1.0e-9))
        );
    }
}
