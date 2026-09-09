//! Backend delay realization. Padding belongs to a parallel stage, never an
//! individual branch: every path through that stage receives the same latency.
use super::conformance::{ExportArtifactManifest, ExportNodeKind};
use roomeq_model::PluginConfigWrapper;
use std::fmt::Write;

/// Additional latency introduced solely by backend fractional-delay support.
/// This excludes requested delays, pre-existing FIR latency and device buffers.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CamillaDspDelayRealization {
    pub sample_rate_hz: u32,
    pub serial_padding_samples: usize,
    pub pre_route_padding_samples: usize,
    pub route_padding_samples: usize,
    pub post_route_padding_samples: usize,
    pub common_padding_samples: usize,
    pub additional_latency_ms: f64,
    pub fractional_delay_present: bool,
    pub usable_band_upper_hz: f64,
    pub magnitude_tolerance_db: f64,
}

pub(super) fn report(graph: &roomeq_model::DspGraph, rate: f64) -> CamillaDspDelayRealization {
    let mut groups = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    if let Some(routing) = super::camilladsp_routing_graph(graph).filter(|r| !r.routes.is_empty()) {
        let (inputs, outputs) = super::routed_channel_names(graph, &routing);
        groups[1] = inputs.iter().map(|name| total_delay(&super::plugins_for_stage(&graph.channels[name], "pre_route"))).collect();
        groups[2] = routing.routes.iter().map(|route| route.delay_ms).collect();
        groups[3] = outputs.iter().map(|name| total_delay(&super::plugins_for_stage(&graph.channels[name], "post_route"))).collect();
    } else {
        groups[0] = graph.channels.values().map(|chain| total_delay(&chain.plugins)).collect();
    }
    let fractional = groups.iter().flatten().any(|ms| {
        let samples = ms * rate / 1000.0;
        (samples - samples.round()).abs() > 1e-9
    });
    let pads = groups.map(|group| padding(group, rate));
    let common_padding_samples = pads.iter().sum();
    CamillaDspDelayRealization {
        sample_rate_hz: rate as u32,
        serial_padding_samples: pads[0], pre_route_padding_samples: pads[1],
        route_padding_samples: pads[2], post_route_padding_samples: pads[3],
        common_padding_samples,
        additional_latency_ms: common_padding_samples as f64 * 1000.0 / rate,
        fractional_delay_present: fractional,
        usable_band_upper_hz: if fractional { roomeq_engine::fir::GD_DELAY_MAX_NORMALIZED_FREQUENCY * rate } else { rate / 2.0 },
        magnitude_tolerance_db: if fractional { roomeq_engine::fir::GD_DELAY_MAGNITUDE_TOLERANCE_DB } else { 0.0 },
    }
}

pub(super) fn total_delay(plugins: &[PluginConfigWrapper]) -> f64 {
    plugins.iter().filter(|p| p.plugin_type == "delay")
        .map(|p| p.parameters["delay_ms"].as_f64().expect("validated delay"))
        .sum()
}

pub(super) fn padding(delays: impl IntoIterator<Item = f64>, rate: f64) -> usize {
    roomeq_engine::fir::gd_delay_padding_samples(&delays.into_iter().collect::<Vec<_>>(), rate)
}

pub(super) fn write_delay(
    out: &mut String, manifest: &mut ExportArtifactManifest, name: &str,
    delay_ms: f64, rate: f64, padding: usize,
) -> anyhow::Result<()> {
    manifest.define_node(ExportNodeKind::Processor, name)?;
    let samples = delay_ms * rate / 1000.0 + padding as f64;
    if (samples - samples.round()).abs() <= 1e-9 {
        writeln!(out, "  {name}:\n    type: Delay\n    parameters:\n      delay: {:.0}\n      unit: samples", samples.round())?;
    } else {
        let realized = roomeq_engine::fir::realize_gd_fir_delay(&[1.0], delay_ms, rate, padding)
            .map_err(anyhow::Error::msg)?;
        writeln!(out, "  {name}:\n    type: Conv\n    parameters:\n      type: Values\n      values:")?;
        for coefficient in realized.coefficients {
            writeln!(out, "      - {coefficient:.17e}")?;
        }
    }
    Ok(())
}

pub(super) fn write_stage(
    out: &mut String, manifest: &mut ExportArtifactManifest, prefix: &str,
    plugins: &[PluginConfigWrapper], rate: f64, padding: usize,
) -> anyhow::Result<Vec<String>> {
    let delay = total_delay(plugins);
    // All supported scalar LTI processors commute with a pure delay. Collapse
    // repeated delay ownership before realizing its fractional component.
    let others: Vec<_> = plugins.iter().filter(|p| p.plugin_type != "delay").cloned().collect();
    let mut names = super::write::write_camilladsp_filters_for_plugins(out, manifest, prefix, &others)?;
    if plugins.iter().any(|p| p.plugin_type == "delay") || padding != 0 {
        let name = format!("{prefix}_delay");
        write_delay(out, manifest, &name, delay, rate, padding)?;
        // Preserve the first delay's position in the serialized chain. Later
        // delays commute into it, but unrelated processors retain their order.
        let index = if let Some(first) = plugins.iter().position(|p| p.plugin_type == "delay") {
            let mut scratch = String::new();
            let mut scratch_manifest = ExportArtifactManifest::new(crate::ExportFormat::CamillaDsp);
            super::write::write_camilladsp_filters_for_plugins(
                &mut scratch, &mut scratch_manifest, prefix, &plugins[..first])?.len()
        } else { names.len() };
        names.insert(index, name);
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn shared_padding_includes_zero_delay_reference() {
        let rate = 48_000.0;
        let delay = 0.10478137820238476;
        let common = padding([0.0, delay], rate);
        assert_eq!(common, 59);
        let mut manifest = ExportArtifactManifest::new(crate::ExportFormat::CamillaDsp);
        let mut out = String::new();
        let reference = write_stage(&mut out, &mut manifest, "reference", &[], rate, common).unwrap();
        assert_eq!(reference, ["reference_delay"]);
        assert!(out.contains("delay: 59\n      unit: samples"));
        let plugins = [PluginConfigWrapper {
            plugin_type: "delay".into(), parameters: json!({"delay_ms": delay}),
        }];
        let delayed = write_stage(&mut out, &mut manifest, "delayed", &plugins, rate, common).unwrap();
        assert_eq!(delayed, ["delayed_delay"]);
        assert!(out.contains("type: Values"));
    }

    #[test]
    fn exact_integer_delay_needs_no_padding_or_decimal_ms_rounding() {
        for rate in [44_100.0, 48_000.0, 96_000.0] {
            assert_eq!(padding([0.0, 5.0 * 1000.0 / rate], rate), 0);
            let mut manifest = ExportArtifactManifest::new(crate::ExportFormat::CamillaDsp);
            let mut out = String::new();
            write_delay(&mut out, &mut manifest, "delay", 5.0 * 1000.0 / rate, rate, 0).unwrap();
            assert!(out.contains("delay: 5\n      unit: samples"));
        }
    }
}
