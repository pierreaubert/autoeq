//! Cross-checks between physical routing (roomeq-dsp) and realized
//! electrical response (engine DSP realization).

use crate::physical_routing::resolve_physical_routing;
use roomeq_model::{
    BassManagementRoute, BassManagementRoutingGraph, ChannelDspChain, DriverDspChain,
    PluginConfigWrapper,
};
use serde_json::json;
use std::collections::HashMap;

fn tagged(kind: &str, mut parameters: serde_json::Value, owner: &str) -> PluginConfigWrapper {
    parameters["room_eq_stage"] = json!(owner);
    PluginConfigWrapper {
        plugin_type: kind.into(),
        parameters,
    }
}

fn fixture() -> (HashMap<String, ChannelDspChain>, BassManagementRoutingGraph) {
    let mut channels: HashMap<String, ChannelDspChain> = ["L", "R", "LFE"]
        .into_iter()
        .map(|name| {
            let chain = serde_json::from_value(json!({"channel": name, "plugins": []})).unwrap();
            (name.into(), chain)
        })
        .collect();
    channels.get_mut("L").unwrap().plugins.push(tagged(
        "gain",
        json!({"gain_db": -3.0}),
        "pre_route",
    ));
    let sub = channels.get_mut("LFE").unwrap();
    sub.plugins
        .push(tagged("gain", json!({"gain_db": -7.0}), "pre_route"));
    for count in [9, 4] {
        sub.plugins.push(tagged(
            "eq",
            json!({"filters": vec![
                json!({"filter_type": "peak", "freq": 70.0, "q": 1.0, "db_gain": 0.0}); count
            ]}),
            "post_route",
        ));
    }
    sub.plugins
        .push(tagged("gain", json!({"gain_db": -6.0}), "post_route"));
    sub.plugins
        .push(tagged("gain", json!({"gain_db": 4.0}), "route_owned"));
    sub.drivers = Some(vec![
        DriverDspChain {
            measured_acoustics: None,
            name: "subs_1".into(),
            index: 0,
            initial_curve: None,
            measured_band_hz: None,
            plugins: vec![tagged("delay", json!({"delay_ms": 9.0}), "post_route")],
        },
        DriverDspChain {
            measured_acoustics: None,
            name: "subs_2".into(),
            index: 1,
            initial_curve: None,
            measured_band_hz: None,
            plugins: vec![tagged("gain", json!({"gain_db": -5.0}), "post_route")],
        },
    ]);
    let mut graph = BassManagementRoutingGraph {
        physical_sub_output: "LFE".into(),
        physical_sub_outputs: Vec::new(),
        stereo_routing: None,
        input_channels: vec!["L".into(), "R".into(), "LFE".into()],
        output_channels: vec![
            "L".into(),
            "R".into(),
            "LFE".into(),
            "subs_1".into(),
            "subs_2".into(),
        ],
        routes: vec![],
        matrix: None,
        input_trim_db: HashMap::from([("L".into(), -3.0)]),
        post_dsp_main_alignment_band_hz: None,
        advisories: vec![],
    };
    for (source_index, source) in ["L", "R", "LFE"].into_iter().enumerate() {
        for (destination_index, destination, gain_db, delay_ms) in
            [(3, "subs_1", 4.0, 9.0), (4, "subs_2", -1.0, 0.0)]
        {
            graph.routes.push(BassManagementRoute {
                group_id: None,
                source_channel: source.into(),
                source_index,
                destination: destination.into(),
                destination_index,
                pre_chain_channel: Some("LFE".into()),
                post_chain_channel: Some(destination.into()),
                route_kind: if source == "LFE" {
                    "lfe_lowpass_to_sub"
                } else {
                    "redirected_bass_lowpass_to_sub"
                }
                .into(),
                crossover_type: "LR48".into(),
                high_pass_hz: None,
                low_pass_hz: Some(if source == "LFE" { 120.0 } else { 80.0 }),
                gain_db,
                gain_linear: 10.0_f64.powf(gain_db / 20.0),
                matrix_gain: 10.0_f64.powf(gain_db / 20.0),
                delay_ms,
                polarity_inverted: destination_index == 4,
            });
        }
    }
    for index in 0..2 {
        let mut route = graph.routes[index * 2].clone();
        route.destination = route.source_channel.clone();
        route.destination_index = index;
        route.pre_chain_channel = Some(route.source_channel.clone());
        route.post_chain_channel = Some(route.source_channel.clone());
        route.route_kind = "main_highpass_to_self".into();
        route.high_pass_hz = route.low_pass_hz.take();
        route.gain_db = 0.0;
        route.gain_linear = 1.0;
        route.matrix_gain = 1.0;
        route.delay_ms = 0.0;
        graph.routes.push(route);
    }
    (channels, graph)
}

#[test]
fn physical_routing_preserves_correction_owned_output_delay_once() {
    let (mut channels, graph) = fixture();
    let baseline = resolve_physical_routing(&channels, &graph).unwrap();
    channels.get_mut("LFE").unwrap().drivers.as_mut().unwrap()[0]
        .plugins
        .push(tagged(
            "delay",
            json!({"delay_ms": 0.5, "room_eq_correction_delay": true}),
            "post_route",
        ));
    let physical = resolve_physical_routing(&channels, &graph).unwrap();
    let owned: Vec<_> = physical.outputs[3]
        .plugins
        .iter()
        .filter(|plugin| {
            plugin.plugin_type == "delay" && plugin.parameters["room_eq_correction_delay"] == true
        })
        .collect();
    assert_eq!(owned.len(), 1);
    assert_eq!(owned[0].parameters["delay_ms"], 0.5);
    assert!(
        physical.outputs[4]
            .plugins
            .iter()
            .all(|plugin| plugin.parameters["room_eq_correction_delay"] != true)
    );
    assert_eq!(physical.routes[3].delay_ms, graph.routes[3].delay_ms);

    let response = |plugins: &[PluginConfigWrapper]| {
        let mut chain = channels["LFE"].clone();
        chain.plugins = plugins.to_vec();
        chain.drivers = None;
        let mut sidecars = crate::dsp_realization::NoConvolutionIr;
        let mut realized =
            crate::dsp_realization::RealizedDsp::new(&chain, 48_000.0, &mut sidecars).unwrap();
        realized.response_at(100.0).unwrap()
    };
    let before = response(&baseline.outputs[3].plugins);
    let after = response(&physical.outputs[3].plugins);
    let expected = num_complex::Complex64::from_polar(
        1.0,
        -2.0 * std::f64::consts::PI * 100.0 * 0.5 / 1_000.0,
    );
    assert!(((after / before) - expected).norm() < 1e-9);

    for bad_delay in [json!(-0.5), json!("invalid")] {
        let mut changed = channels.clone();
        let driver = &mut changed.get_mut("LFE").unwrap().drivers.as_mut().unwrap()[0];
        driver
            .plugins
            .iter_mut()
            .find(|plugin| plugin.parameters["room_eq_correction_delay"] == true)
            .unwrap()
            .parameters["delay_ms"] = bad_delay;
        assert!(resolve_physical_routing(&changed, &graph).is_err());
    }
}
