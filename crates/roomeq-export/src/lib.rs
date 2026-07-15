//! Experimental exporters for the canonical `roomeq-model` DSP graph.
//!
//! Production RoomEQ formats remain in the root `autoeq` crate while they are
//! migrated. These exporters validate the graph and never emit an empty
//! placeholder artifact.

#![forbid(unsafe_code)]

use roomeq_model::{DspGraph, Plugin};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Json,
    EqualizerApo,
}

pub fn export(graph: &DspGraph, format: ExportFormat) -> Result<String, String> {
    graph.validate()?;
    match format {
        ExportFormat::Json => serde_json::to_string_pretty(graph).map_err(|e| e.to_string()),
        ExportFormat::EqualizerApo => export_equalizer_apo(graph),
    }
}

fn export_equalizer_apo(graph: &DspGraph) -> Result<String, String> {
    let mut out = format!("# RoomEQ export v{}\n", graph.version);
    for (channel, chain) in &graph.channels {
        out.push_str(&format!("\nChannel: {channel}\n"));
        for plugin in &chain.plugins {
            write_plugin(&mut out, plugin)?;
        }
    }
    Ok(out)
}

fn write_plugin(out: &mut String, plugin: &Plugin) -> Result<(), String> {
    match plugin.kind.as_str() {
        "gain" => {
            let gain = plugin
                .parameters
                .get("gain_db")
                .and_then(|v| v.as_f64())
                .ok_or("gain plugin requires gain_db")?;
            out.push_str(&format!("Preamp: {gain:+.2} dB\n"));
        }
        "eq" => {
            let p = &plugin.parameters;
            let freq = p
                .get("freq")
                .and_then(|v| v.as_f64())
                .ok_or("eq plugin requires freq")?;
            let gain = p.get("gain_db").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let q = p.get("q").and_then(|v| v.as_f64()).unwrap_or(1.0);
            out.push_str(&format!(
                "Filter: ON PK Fc {freq:.2} Hz Gain {gain:+.2} dB Q {q:.4}\n"
            ));
        }
        "delay" => {
            let delay = plugin
                .parameters
                .get("delay_ms")
                .and_then(|v| v.as_f64())
                .ok_or("delay plugin requires delay_ms")?;
            out.push_str(&format!("Delay: {delay:.3} ms\n"));
        }
        other => return Err(format!("unsupported plugin kind '{other}'")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_graph_for_every_format() {
        let graph = DspGraph::new("1");
        for format in [ExportFormat::Json, ExportFormat::EqualizerApo] {
            let error = export(&graph, format).expect_err("empty graph must not export");
            assert!(
                error.contains("at least one channel"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn exports_json_and_apo() {
        let mut graph = DspGraph::new("1");
        graph.add_channel(
            "L",
            vec![Plugin {
                kind: "gain".into(),
                parameters: serde_json::json!({"gain_db": -3.0}),
            }],
        );
        assert!(
            export(&graph, ExportFormat::Json)
                .unwrap()
                .contains("channels")
        );
        assert!(
            export(&graph, ExportFormat::EqualizerApo)
                .unwrap()
                .contains("Preamp")
        );
    }
}
